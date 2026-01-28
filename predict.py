import argparse
import os
import yaml
import numpy as np
import torch
import rasterio
from tqdm import tqdm
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model.smp_models import create_model, MODEL_FACTORY


def get_device(gpus_str):
    """Sets the device for PyTorch operations."""
    if torch.cuda.is_available() and gpus_str:
        gpu_ids = [int(gpu) for gpu in gpus_str.split(',')]
        device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"Using GPU: {gpu_ids[0]}")
    else:
        device = torch.device("cpu")
        print("CUDA not available or no GPUs specified. Using CPU.")
    return device


def get_patch_size_from_config(config):
    """
    Determine appropriate patch size from model config.
    Models with '224' in encoder name require 224x224 input.
    """
    encoder_name = config['model']['args'].get('encoder_name', '')
    model_name = config['model']['name']
    
    # DPT and UperNet with ViT/Swin typically need 224
    if '224' in encoder_name or model_name in ['DPT', 'UperNet']:
        return 224
    return 256


def load_model_from_log_dir(log_dir, checkpoint_name, device):
    """
    Load model from a log directory.
    
    Args:
        log_dir: Path to the model's log directory
        checkpoint_name: 'best.pt' or 'last.pt'
        device: Device to load model on
        
    Returns:
        Tuple of (model, config) or (None, None) if failed
    """
    config_path = os.path.join(log_dir, 'config.yaml')
    checkpoint_path = os.path.join(log_dir, 'weights', checkpoint_name)
    
    if not os.path.exists(config_path):
        print(f"  Config not found: {config_path}")
        return None, None
    
    if not os.path.exists(checkpoint_path):
        print(f"  Checkpoint not found: {checkpoint_path}")
        return None, None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_name = config['model']['name']
        model_args = config['model']['args']
        
        # Create model using factory
        model = create_model(
            model_name=model_name,
            encoder_name=model_args.get('encoder_name', 'resnet34'),
            in_channels=model_args.get('in_channels', 13),
            classes=model_args.get('classes', 1),
            encoder_weights=None  # Don't load pretrained weights
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        return model, config
        
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None, None


def create_gaussian_window(size, sigma_ratio=0.25):
    """
    Create a 2D Gaussian window for smooth patch blending.
    
    Args:
        size: Window size (assumes square)
        sigma_ratio: Ratio of sigma to size (smaller = more peaked center)
        
    Returns:
        2D Gaussian weight array normalized to [0, 1]
    """
    sigma = size * sigma_ratio
    
    # Create 1D Gaussian
    x = np.arange(size) - (size - 1) / 2
    gauss_1d = np.exp(-x**2 / (2 * sigma**2))
    
    # Create 2D Gaussian via outer product
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    
    # Normalize to [0, 1] with minimum value to avoid zero weights
    gauss_2d = gauss_2d / gauss_2d.max()
    gauss_2d = np.clip(gauss_2d, 0.1, 1.0)  # Minimum 0.1 weight at edges
    
    return gauss_2d.astype(np.float32)


def create_cosine_window(size):
    """
    Create a 2D Cosine (Hann) window for smooth patch blending.
    
    Args:
        size: Window size (assumes square)
        
    Returns:
        2D Cosine weight array
    """
    # Create 1D Hann window
    hann_1d = np.hanning(size)
    
    # Create 2D window via outer product
    hann_2d = np.outer(hann_1d, hann_1d)
    
    # Add minimum value to avoid zero weights at corners
    hann_2d = np.clip(hann_2d, 0.1, 1.0)
    
    return hann_2d.astype(np.float32)


def predict_image(model, image, patch_size, overlap, device, expected_channels=13, use_tta=False):
    """
    Predict on an image using sliding window approach with smooth blending.
    
    Uses Gaussian weighted averaging for smooth transitions between patches,
    which eliminates visible seams at patch boundaries.
    
    Args:
        model: The trained model
        image: Input image (C, H, W) as numpy array
        patch_size: Size of patches
        overlap: Overlap ratio
        device: Device for inference
        expected_channels: Number of channels the model expects
        use_tta: Use Test Time Augmentation (flip averaging)
        
    Returns:
        Binary prediction mask
    """
    n_bands, height, width = image.shape
    
    # Handle channel mismatch - pad with zeros if needed
    if n_bands < expected_channels:
        padding = np.zeros((expected_channels - n_bands, height, width), dtype=image.dtype)
        image = np.concatenate([image, padding], axis=0)
    elif n_bands > expected_channels:
        image = image[:expected_channels]
    
    # Normalize
    image = image.astype(np.float32) / 10000.0
    
    stride = int(patch_size * (1 - overlap))
    if stride <= 0:
        stride = 1
    
    # Calculate necessary padding
    pad_h = (stride - (height - patch_size) % stride) % stride
    pad_w = (stride - (width - patch_size) % stride) % stride
    
    # Pad the image using reflect mode for better edge handling
    padded_image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    padded_height, padded_width = padded_image.shape[1], padded_image.shape[2]
    
    # Create Gaussian weight window for smooth blending
    weight_window = create_gaussian_window(patch_size, sigma_ratio=0.3)
    weight_tensor = torch.from_numpy(weight_window).to(device).unsqueeze(0).unsqueeze(0)
    
    # Create placeholders
    prediction_map = torch.zeros((1, 1, padded_height, padded_width), device=device, dtype=torch.float32)
    weight_map = torch.zeros((1, 1, padded_height, padded_width), device=device, dtype=torch.float32)
    
    with torch.no_grad():
        for y in range(0, padded_height - patch_size + 1, stride):
            for x in range(0, padded_width - patch_size + 1, stride):
                patch = padded_image[:, y:y+patch_size, x:x+patch_size]
                patch_tensor = torch.from_numpy(patch).float().to(device).unsqueeze(0)
                
                # Basic prediction
                output = model(patch_tensor)
                prediction = torch.sigmoid(output)
                
                # Optional: Test Time Augmentation (horizontal + vertical flip)
                if use_tta:
                    # Horizontal flip
                    patch_hflip = torch.flip(patch_tensor, dims=[3])
                    pred_hflip = torch.sigmoid(model(patch_hflip))
                    pred_hflip = torch.flip(pred_hflip, dims=[3])
                    
                    # Vertical flip
                    patch_vflip = torch.flip(patch_tensor, dims=[2])
                    pred_vflip = torch.sigmoid(model(patch_vflip))
                    pred_vflip = torch.flip(pred_vflip, dims=[2])
                    
                    # Both flips
                    patch_hvflip = torch.flip(patch_tensor, dims=[2, 3])
                    pred_hvflip = torch.sigmoid(model(patch_hvflip))
                    pred_hvflip = torch.flip(pred_hvflip, dims=[2, 3])
                    
                    # Average all predictions
                    prediction = (prediction + pred_hflip + pred_vflip + pred_hvflip) / 4.0
                
                # Apply Gaussian weight to prediction
                weighted_pred = prediction * weight_tensor
                
                # Accumulate weighted predictions and weights
                prediction_map[:, :, y:y+patch_size, x:x+patch_size] += weighted_pred
                weight_map[:, :, y:y+patch_size, x:x+patch_size] += weight_tensor
    
    # Weighted average
    averaged_prediction = prediction_map / (weight_map + 1e-6)
    
    # Crop back to original size
    final_prediction = averaged_prediction[:, :, :height, :width]
    
    # Convert to binary mask
    binary_mask = (final_prediction > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    
    return binary_mask


def normalize_for_display(image, bands=(3, 2, 1)):
    """Normalize image for RGB display."""
    r_idx, g_idx, b_idx = bands
    max_band = max(bands)
    
    if image.shape[0] <= max_band:
        r_idx, g_idx, b_idx = min(2, image.shape[0]-1), min(1, image.shape[0]-1), 0
    
    rgb = np.stack([
        image[r_idx],
        image[g_idx],
        image[b_idx]
    ], axis=-1).astype(np.float32)
    
    # Normalize for Sentinel-2
    rgb = (rgb / 10000.0) * (255 * 5)
    rgb = np.clip(rgb / 255.0, 0, 1)
    
    return rgb


def create_all_models_comparison(
    image_path: str,
    logs_dir: str,
    output_path: str,
    patch_size: int = 256,
    overlap: float = 0.5,
    device: torch.device = None,
    use_tta: bool = False
):
    """
    Run prediction with all models (best and last checkpoints) and create comparison figure.
    
    Args:
        image_path: Path to input image
        logs_dir: Path to logs directory containing all model folders
        output_path: Path to save the comparison figure
        patch_size: Patch size for prediction
        overlap: Overlap ratio
        device: Device for inference
    """
    # Load image
    print(f"Loading image: {image_path}")
    with rasterio.open(image_path) as src:
        image = src.read()
    
    # Fix coordinate system mismatch between rasterio (geographic) and display (screen)
    # Flip vertically to match expected orientation
    image = image[:, ::-1, :]  # Flip height axis
    
    n_bands, height, width = image.shape
    print(f"  Shape: {image.shape} (bands, height, width)")
    
    # Get RGB for display
    rgb = normalize_for_display(image)
    
    # Find all model directories
    model_dirs = []
    for name in sorted(os.listdir(logs_dir)):
        model_path = os.path.join(logs_dir, name)
        weights_path = os.path.join(model_path, 'weights')
        if os.path.isdir(model_path) and os.path.exists(weights_path):
            model_dirs.append((name, model_path))
    
    print(f"Found {len(model_dirs)} model directories")
    
    # Collect all predictions
    predictions = []  # List of (model_name, checkpoint_type, prediction_mask)
    
    for model_name, model_path in tqdm(model_dirs, desc="Processing models"):
        for checkpoint in ['best.pt', 'last.pt']:
            checkpoint_type = checkpoint.replace('.pt', '')
            
            print(f"\nLoading {model_name} ({checkpoint_type})...")
            model, config = load_model_from_log_dir(model_path, checkpoint, device)
            
            if model is None:
                print(f"  Skipped")
                continue
            
            expected_channels = config['model']['args'].get('in_channels', 13)
            model_patch_size = get_patch_size_from_config(config)
            
            print(f"  Running prediction (expects {expected_channels} channels, patch {model_patch_size}, TTA={use_tta})...")
            try:
                pred_mask = predict_image(
                    model=model,
                    image=image.copy(),
                    patch_size=model_patch_size,
                    overlap=overlap,
                    device=device,
                    expected_channels=expected_channels,
                    use_tta=use_tta
                )
                
                predictions.append({
                    'model_name': model_name,
                    'checkpoint': checkpoint_type,
                    'mask': pred_mask
                })
                print(f"  Done. Positive pixels: {pred_mask.sum()}")
                
            except Exception as e:
                print(f"  Error during prediction: {e}")
                continue
            
            # Free memory
            del model
            torch.cuda.empty_cache()
    
    if len(predictions) == 0:
        print("No predictions generated!")
        return
    
    # Create comparison figure
    print(f"\nCreating comparison figure with {len(predictions)} predictions...")
    
    # Group by model name
    model_groups = {}
    for pred in predictions:
        name = pred['model_name']
        if name not in model_groups:
            model_groups[name] = {}
        model_groups[name][pred['checkpoint']] = pred['mask']
    
    n_models = len(model_groups)
    n_cols = 4  # RGB, best, last, overlay
    n_rows = n_models + 1  # +1 for header row with just RGB
    
    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.1)
    
    # Header row - show RGB large
    ax_rgb_main = fig.add_subplot(gs[0, :2])
    ax_rgb_main.imshow(rgb)
    ax_rgb_main.set_title(f'Input Image\n({n_bands} bands, {height}x{width})', fontsize=12, fontweight='bold')
    ax_rgb_main.axis('off')
    
    # Legend for header row
    ax_legend = fig.add_subplot(gs[0, 2:])
    ax_legend.text(0.1, 0.8, 'Model Comparison', fontsize=14, fontweight='bold', transform=ax_legend.transAxes)
    ax_legend.text(0.1, 0.6, f'Total Models: {n_models}', fontsize=11, transform=ax_legend.transAxes)
    ax_legend.text(0.1, 0.4, f'Checkpoints: best, last', fontsize=11, transform=ax_legend.transAxes)
    ax_legend.text(0.1, 0.2, f'Patch size: {patch_size}, Overlap: {overlap}', fontsize=10, transform=ax_legend.transAxes)
    ax_legend.axis('off')
    
    # Model rows
    for row_idx, (model_name, checkpoints) in enumerate(sorted(model_groups.items())):
        row = row_idx + 1  # Skip header row
        
        # Column 0: Model name with RGB thumbnail
        ax_name = fig.add_subplot(gs[row, 0])
        ax_name.imshow(rgb)
        ax_name.set_title(model_name.replace('_', '\n'), fontsize=9, fontweight='bold')
        ax_name.axis('off')
        
        # Column 1: Best checkpoint
        ax_best = fig.add_subplot(gs[row, 1])
        if 'best' in checkpoints:
            mask = checkpoints['best']
            ax_best.imshow(mask, cmap='gray', vmin=0, vmax=1)
            pos_ratio = mask.sum() / mask.size * 100
            ax_best.set_title(f'Best\n({pos_ratio:.2f}%)', fontsize=9)
        else:
            ax_best.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            ax_best.set_title('Best', fontsize=9)
        ax_best.axis('off')
        
        # Column 2: Last checkpoint
        ax_last = fig.add_subplot(gs[row, 2])
        if 'last' in checkpoints:
            mask = checkpoints['last']
            ax_last.imshow(mask, cmap='gray', vmin=0, vmax=1)
            pos_ratio = mask.sum() / mask.size * 100
            ax_last.set_title(f'Last\n({pos_ratio:.2f}%)', fontsize=9)
        else:
            ax_last.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            ax_last.set_title('Last', fontsize=9)
        ax_last.axis('off')
        
        # Column 3: Overlay (best prediction on RGB)
        ax_overlay = fig.add_subplot(gs[row, 3])
        if 'best' in checkpoints:
            mask = checkpoints['best']
            # Create overlay
            overlay = rgb.copy()
            # Add red tint where prediction is positive
            overlay[mask > 0, 0] = np.clip(overlay[mask > 0, 0] + 0.5, 0, 1)
            overlay[mask > 0, 1] = overlay[mask > 0, 1] * 0.5
            overlay[mask > 0, 2] = overlay[mask > 0, 2] * 0.5
            ax_overlay.imshow(overlay)
            ax_overlay.set_title('Overlay (Best)', fontsize=9)
        else:
            ax_overlay.imshow(rgb)
            ax_overlay.set_title('Overlay', fontsize=9)
        ax_overlay.axis('off')
    
    # Save figure
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    fig.suptitle(f'All Models Comparison: {image_name}', fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\nComparison figure saved to: {output_path}")
    
    # Also save individual predictions
    individual_dir = os.path.join(os.path.dirname(output_path), 'individual_predictions')
    os.makedirs(individual_dir, exist_ok=True)
    
    for pred in predictions:
        pred_path = os.path.join(individual_dir, f"{pred['model_name']}_{pred['checkpoint']}.png")
        plt.imsave(pred_path, pred['mask'], cmap='gray')
    
    print(f"Individual predictions saved to: {individual_dir}")


def main():
    parser = argparse.ArgumentParser(description="Mangrove Segmentation Prediction - All Models Comparison")
    parser.add_argument(
        '--image_path', type=str, 
        default='S2_SR_HARMONIZED_20251112T021829_20251112T023015_T50PQS_13bands.tif',
        help='Path to the input image file for prediction.'
    )
    parser.add_argument(
        '--logs_dir', type=str, default='logs',
        help='Path to the logs directory containing all trained models.'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save the comparison figure. Default: results/all_models_comparison.png'
    )
    parser.add_argument(
        '--patch_size', type=int, default=256,
        help='Patch size for sliding window prediction. Default: 256'
    )
    parser.add_argument(
        '--overlap', type=float, default=0.5,
        help='Overlap ratio between patches (0 to 1). Default: 0.5 (higher = smoother but slower)'
    )
    parser.add_argument(
        '--tta', action='store_true',
        help='Use Test Time Augmentation (flip averaging) for better results'
    )
    parser.add_argument(
        '--gpus', type=str, default='0',
        help='GPU IDs to use for inference. Uses CPU if empty or not available.'
    )
    args = parser.parse_args()
    
    # Check input image exists
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found: {args.image_path}")
    
    # Check logs directory exists
    if not os.path.exists(args.logs_dir):
        raise FileNotFoundError(f"Logs directory not found: {args.logs_dir}")
    
    # Set default output path
    if args.output_path is None:
        os.makedirs('results', exist_ok=True)
        image_name = os.path.splitext(os.path.basename(args.image_path))[0]
        args.output_path = f'results/{image_name}_all_models_comparison.png'
    
    # Setup device
    device = get_device(args.gpus)
    
    print("=" * 60)
    print("Mangrove Segmentation - All Models Comparison")
    print("=" * 60)
    print(f"Input image: {args.image_path}")
    print(f"Logs directory: {args.logs_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Patch size: {args.patch_size}")
    print(f"Overlap: {args.overlap}")
    print(f"TTA: {args.tta}")
    print("=" * 60)
    
    # Run comparison
    create_all_models_comparison(
        image_path=args.image_path,
        logs_dir=args.logs_dir,
        output_path=args.output_path,
        patch_size=args.patch_size,
        overlap=args.overlap,
        device=device,
        use_tta=args.tta
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
