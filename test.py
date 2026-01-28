import os
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp

from model.unet_plus_plus import UnetPlusPlus
from model.smp_models import create_model
from data.mangrove_dataset import MangroveDataset

# Import AE components conditionally
try:
    from data.ae_dataset import AEDataset
    from model.ae_enhanced_smp import create_ae_enhanced_model
    AE_AVAILABLE = True
except ImportError as e:
    AE_AVAILABLE = False
    print(f"AE components not available: {e}")


def needs_224_input(model_name: str) -> bool:
    """
    Check if a model requires 224x224 input size instead of 256x256.
    
    Args:
        model_name: Name of the model (with or without AE prefix)
    
    Returns:
        True if model needs 224x224 input, False otherwise
    """
    # Remove AE prefix if present
    base_name = model_name[2:] if model_name.startswith('AE') else model_name
    
    # Models that require 224x224 input
    models_224 = ['DPT', 'UperNet']
    
    return base_name in models_224


def get_device(gpu_id=None):
    """Sets the device for PyTorch operations."""
    if torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {gpu_id}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU: 0")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate various metrics including IoU, F1 score, precision, and recall.
    
    Args:
        pred_mask (torch.Tensor): Predicted mask (probabilities)
        gt_mask (torch.Tensor): Ground truth mask (binary)
        threshold (float): Threshold for converting probabilities to binary
        
    Returns:
        dict: Dictionary containing various metrics
    """
    # Convert predictions to binary
    pred_binary = (pred_mask > threshold).float()
    gt_binary = gt_mask.float()
    
    # Calculate using segmentation_models_pytorch metrics
    tp, fp, fn, tn = smp.metrics.get_stats(
        pred_binary.long(), gt_binary.long(), mode='binary', threshold=threshold
    )
    
    # Calculate metrics
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro-imagewise')
    f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro-imagewise')
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction='micro-imagewise')
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction='micro-imagewise')
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro-imagewise')
    
    # Sum the stats if they are tensors (for pixel-wise statistics)
    tp_sum = tp.sum().item() if tp.numel() > 1 else tp.item()
    fp_sum = fp.sum().item() if fp.numel() > 1 else fp.item()
    fn_sum = fn.sum().item() if fn.numel() > 1 else fn.item()
    tn_sum = tn.sum().item() if tn.numel() > 1 else tn.item()
    
    return {
        'iou': iou.item(),
        'f1': f1.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': accuracy.item(),
        'tp': tp_sum,
        'fp': fp_sum,
        'fn': fn_sum,
        'tn': tn_sum
    }


def create_test_visualization(input_image, gt_mask, pred_mask, metrics, image_name, ae_projection=None):
    """
    Create a visualization for test results with metrics overlay.
    
    Args:
        input_image (torch.Tensor): Original input image (C, H, W)
        gt_mask (torch.Tensor): Ground truth mask (H, W)
        pred_mask (torch.Tensor): Predicted mask probabilities (H, W)
        metrics (dict): Dictionary of calculated metrics
        image_name (str): Name of the image
        ae_projection (torch.Tensor, optional): AE projection features (D, H, W)
        
    Returns:
        np.ndarray: Combined visualization image
    """
    # Convert tensors to numpy for visualization
    if input_image.shape[0] > 3:
        # Take first 3 channels for RGB visualization (B, G, R -> R, G, B)
        vis_image = input_image[[2, 1, 0], :, :].permute(1, 2, 0).cpu().numpy()
    else:
        vis_image = input_image.permute(1, 2, 0).cpu().numpy()
    
    # Clip and normalize for display
    vis_image = np.clip(vis_image * 3.0, 0, 1)  # Enhance brightness
    
    gt_mask_np = gt_mask.cpu().numpy()
    pred_mask_np = pred_mask.cpu().numpy()
    
    # Determine number of subplots based on whether AE projection is provided
    if ae_projection is not None:
        # AE model: 4 subplots (input, gt, prediction, ae_projection)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{image_name}\nIoU: {metrics["iou"]:.4f} | F1: {metrics["f1"]:.4f} | Precision: {metrics["precision"]:.4f} | Recall: {metrics["recall"]:.4f}', 
                     fontsize=14, fontweight='bold')
        
        # Input image
        axes[0, 0].imshow(vis_image)
        axes[0, 0].set_title('Input Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(gt_mask_np, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Ground Truth', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Prediction (binary)
        pred_binary_np = (pred_mask_np > 0.5).astype(np.float32)
        axes[1, 0].imshow(pred_binary_np, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Prediction', fontweight='bold')
        axes[1, 0].axis('off')
        
        # AE Projection (show as RGB using first 3 channels)
        ae_projection_np = ae_projection.cpu().numpy()
        if ae_projection_np.shape[0] >= 3:
            # Take first 3 channels for RGB visualization
            ae_rgb = ae_projection_np[:3].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
            # Normalize to 0-1 range for display
            ae_rgb = (ae_rgb - ae_rgb.min()) / (ae_rgb.max() - ae_rgb.min() + 1e-8)
        else:
            # If less than 3 channels, duplicate to make RGB
            ae_single = ae_projection_np[0]
            ae_single = (ae_single - ae_single.min()) / (ae_single.max() - ae_single.min() + 1e-8)
            ae_rgb = np.stack([ae_single, ae_single, ae_single], axis=-1)
        
        axes[1, 1].imshow(ae_rgb)
        axes[1, 1].set_title('AE Projection', fontweight='bold')
        axes[1, 1].axis('off')
    else:
        # Regular model: 3 subplots (input, gt, prediction)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{image_name}\nIoU: {metrics["iou"]:.4f} | F1: {metrics["f1"]:.4f} | Precision: {metrics["precision"]:.4f} | Recall: {metrics["recall"]:.4f}', 
                     fontsize=14, fontweight='bold')
        
        # Input image
        axes[0].imshow(vis_image)
        axes[0].set_title('Input Image', fontweight='bold')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(gt_mask_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth', fontweight='bold')
        axes[1].axis('off')
        
        # Prediction (binary)
        pred_binary_np = (pred_mask_np > 0.5).astype(np.float32)
        axes[2].imshow(pred_binary_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Prediction', fontweight='bold')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to numpy array
    fig.canvas.draw()
    vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return vis_array


def load_model_and_config(log_dir, weight_type='best'):
    """
    Load model and config from log directory.
    
    Args:
        log_dir (str): Path to log directory
        weight_type (str): Type of weights to load ('best' or 'last')
        
    Returns:
        tuple: (model, config, device)
    """
    # Load config
    config_path = os.path.join(log_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = get_device()
    
    # Dynamic model loading based on config
    model_name = config['model']['name']
    
    # Check if it's an AE-enhanced model
    if model_name.startswith('AE'):
        # AE-enhanced versions of all models (including UNet++)
        if not AE_AVAILABLE:
            raise RuntimeError("AE components are not available. Please check your installation.")
        
        # Extract base model name (e.g., "AEMAnet" -> "MAnet", "AEUnetPlusPlus" -> "UnetPlusPlus")
        base_model_name = model_name[2:]  # Remove "AE" prefix
        if base_model_name in ['UnetPlusPlus', 'MAnet', 'PAN', 'DeepLabV3Plus', 'Segformer', 'FPN', 'DPT', 'UperNet']:
            # Create AE-enhanced version using unified SMP approach
            model = create_ae_enhanced_model(base_model_name, **config['model']['args'])
        else:
            raise ValueError(f"Unknown AE model base: {base_model_name}")
    elif model_name in ['UnetPlusPlus', 'MAnet', 'PAN', 'DeepLabV3Plus', 'Segformer', 'FPN', 'DPT', 'UperNet']:
        # Standard SMP models
        if model_name == 'UnetPlusPlus':
            # Use original implementation for backward compatibility
            model = UnetPlusPlus(**config['model']['args'])
        else:
            # Use SMP factory for new models
            model = create_model(model_name, **config['model']['args'])
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: UnetPlusPlus, MAnet, PAN, DeepLabV3Plus, Segformer, FPN, DPT, UperNet, and their AE variants (prefix with 'AE').")
    
    # Load weights
    weight_path = os.path.join(log_dir, 'weights', f'{weight_type}.pt')
    if not os.path.exists(weight_path):
        # Try the other weight type
        alternative_weight = 'last' if weight_type == 'best' else 'best'
        weight_path = os.path.join(log_dir, 'weights', f'{alternative_weight}.pt')
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found at {weight_path}")
        print(f"Using {alternative_weight} weights instead of {weight_type}")
    
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from {weight_path}")
    return model, config, device


def test_model(log_dir, weight_type='best', batch_size=1, num_workers=4):
    """
    Test the model on test dataset and save results.
    
    Args:
        log_dir (str): Path to log directory
        weight_type (str): Type of weights to load ('best' or 'last')
        batch_size (int): Batch size for testing
        num_workers (int): Number of workers for data loading
    """
    print(f"Starting testing with log directory: {log_dir}")
    
    # Load model and config
    model, config, device = load_model_and_config(log_dir, weight_type)
    
    # Check if AE model or regular model
    model_name = config['model']['name']
    is_ae_model = model_name.startswith('AE')
    
    # Create directories for results
    results_dir = os.path.join(log_dir, 'results')
    visualization_dir = os.path.join(results_dir, 'visualization')
    predictions_dir = os.path.join(results_dir, 'predictions')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    if is_ae_model:
        ae_projections_dir = os.path.join(results_dir, 'ae_projections')
        os.makedirs(ae_projections_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    print(f"Visualizations will be saved to: {visualization_dir}")
    print(f"Predictions will be saved to: {predictions_dir}")
    if is_ae_model:
        print(f"AE projections will be saved to: {ae_projections_dir}")
    
    # Select appropriate dataset class and determine target size
    dataset_class = AEDataset if is_ae_model else MangroveDataset
    target_size = 224 if needs_224_input(model_name) else 256
    print(f"Using target size: {target_size}x{target_size} for model: {model_name}")
    
    try:
        test_dataset = dataset_class(
            root_dir=config['data']['root_dir'], 
            split='test', 
            shuffle=False,
            target_size=target_size
        )
        print(f"Test dataset loaded: {len(test_dataset)} images")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        print("Trying 'validation' split instead...")
        test_dataset = dataset_class(
            root_dir=config['data']['root_dir'], 
            split='validation', 
            shuffle=False,
            target_size=target_size
        )
        print(f"Validation dataset loaded: {len(test_dataset)} images")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize results storage
    all_results = []
    
    # Test loop
    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
            if is_ae_model:
                # AE models expect (images, ae_features, labels)
                images, ae_features, labels = batch_data
                images = images.to(device)
                ae_features = ae_features.to(device)
                labels = labels.to(device)
                
                # Forward pass - AE models return (logits, Z)
                outputs, ae_projection = model(images, ae_features)
            else:
                # Regular models expect (images, labels)
                images, labels = batch_data
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                ae_projection = None
            
            predictions = torch.sigmoid(outputs)
            
            # Process each image in the batch
            for i in range(images.size(0)):
                image_idx = batch_idx * batch_size + i
                
                # Get actual image name from dataset
                actual_image_name = os.path.splitext(test_dataset.image_files[image_idx])[0]
                
                # Get individual tensors
                img = images[i]  # (C, H, W)
                gt = labels[i].squeeze(0)  # (H, W)
                pred = predictions[i].squeeze(0)  # (H, W)
                
                # Get AE projection for this image if available
                ae_proj_single = ae_projection[i] if ae_projection is not None else None
                
                # If model used 224 input, resize outputs to 256 for consistent storage
                if target_size == 224:
                    # Resize prediction and ground truth to 256x256 for storage
                    pred = TF.resize(pred.unsqueeze(0), [256, 256], antialias=True).squeeze(0)
                    gt = TF.resize(gt.unsqueeze(0), [256, 256], antialias=False).squeeze(0)
                    img = TF.resize(img, [256, 256], antialias=True)
                    if ae_proj_single is not None:
                        ae_proj_single = TF.resize(ae_proj_single, [256, 256], antialias=True)
                
                # Calculate metrics
                metrics = calculate_metrics(pred, gt)
                
                # Create and save visualization
                visualization = create_test_visualization(img, gt, pred, metrics, actual_image_name, ae_proj_single)
                vis_path = os.path.join(visualization_dir, f"{actual_image_name}.png")
                Image.fromarray(visualization).save(vis_path)
                
                # Save individual prediction as binary PNG (0 or 255)
                pred_binary = (pred.cpu().numpy() > 0.5).astype(np.uint8)
                pred_img = Image.fromarray(pred_binary * 255, mode='L')
                pred_path = os.path.join(predictions_dir, f"{actual_image_name}.png")
                pred_img.save(pred_path)
                
                # Save AE projection if available (as RGB PNG)
                if ae_proj_single is not None:
                    ae_proj_np = ae_proj_single.cpu().numpy()
                    if ae_proj_np.shape[0] >= 3:
                        # Use first 3 channels for RGB
                        ae_rgb = ae_proj_np[:3].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
                        # Normalize to 0-255 range
                        ae_rgb = (ae_rgb - ae_rgb.min()) / (ae_rgb.max() - ae_rgb.min() + 1e-8)
                        ae_rgb = (ae_rgb * 255).astype(np.uint8)
                        ae_img = Image.fromarray(ae_rgb, mode='RGB')
                    else:
                        # If less than 3 channels, duplicate to make RGB
                        ae_single = ae_proj_np[0]
                        ae_single = (ae_single - ae_single.min()) / (ae_single.max() - ae_single.min() + 1e-8)
                        ae_single = (ae_single * 255).astype(np.uint8)
                        ae_rgb = np.stack([ae_single, ae_single, ae_single], axis=-1)
                        ae_img = Image.fromarray(ae_rgb, mode='RGB')
                    
                    ae_path = os.path.join(ae_projections_dir, f"{actual_image_name}.png")
                    ae_img.save(ae_path)
                
                # Store results
                result_row = {
                    'image_name': actual_image_name,
                    'image_index': image_idx,
                    **metrics
                }
                all_results.append(result_row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate summary statistics
    summary_stats = {
        'metric': ['mean', 'std', 'min', 'max', 'median'],
        'iou': [
            results_df['iou'].mean(),
            results_df['iou'].std(),
            results_df['iou'].min(),
            results_df['iou'].max(),
            results_df['iou'].median()
        ],
        'f1': [
            results_df['f1'].mean(),
            results_df['f1'].std(),
            results_df['f1'].min(),
            results_df['f1'].max(),
            results_df['f1'].median()
        ],
        'precision': [
            results_df['precision'].mean(),
            results_df['precision'].std(),
            results_df['precision'].min(),
            results_df['precision'].max(),
            results_df['precision'].median()
        ],
        'recall': [
            results_df['recall'].mean(),
            results_df['recall'].std(),
            results_df['recall'].min(),
            results_df['recall'].max(),
            results_df['recall'].median()
        ],
        'accuracy': [
            results_df['accuracy'].mean(),
            results_df['accuracy'].std(),
            results_df['accuracy'].min(),
            results_df['accuracy'].max(),
            results_df['accuracy'].median()
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save results
    detailed_results_path = os.path.join(results_dir, 'detailed_results.csv')
    summary_results_path = os.path.join(results_dir, 'summary_results.csv')
    
    results_df.to_csv(detailed_results_path, index=False)
    summary_df.to_csv(summary_results_path, index=False)
    
    print(f"\nTesting completed!")
    print(f"Processed {len(results_df)} images")
    print(f"Detailed results saved to: {detailed_results_path}")
    print(f"Summary results saved to: {summary_results_path}")
    
    # Print summary to console
    print(f"\n=== Summary Statistics ===")
    print(f"Average IoU: {results_df['iou'].mean():.4f} ± {results_df['iou'].std():.4f}")
    print(f"Average F1:  {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    print(f"Average Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"Average Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    print(f"Average Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    
    return results_df, summary_df


def main():
    parser = argparse.ArgumentParser(description="Test Mangrove Segmentation Model")
    parser.add_argument(
        'log_dir', type=str,
        help='Path to the log directory containing trained model and config'
    )
    parser.add_argument(
        '--weight-type', type=str, default='last', choices=['best', 'last'],
        help='Type of weights to load (default: best)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=12,
        help='Batch size for testing (default: 1)'
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of workers for data loading (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Validate log directory
    if not os.path.exists(args.log_dir):
        raise FileNotFoundError(f"Log directory not found: {args.log_dir}")
    
    # Run testing
    test_model(
        log_dir=args.log_dir,
        weight_type=args.weight_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
