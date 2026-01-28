import torch
import numpy as np
import torchvision.transforms.functional as F
import matplotlib.cm as cm

def safe_divide(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Safely divides two tensors, avoiding division by zero."""
    den_clone = den.clone()
    den_clone[den_clone == 0] = 1e-6
    return num / den_clone

def colorize_index(index_tensor: torch.Tensor, cmap_name: str, vmin: float = None, vmax: float = None) -> torch.Tensor:
    """Applies a matplotlib colormap to a single-channel tensor."""
    if vmin is not None and vmax is not None:
        normalized_tensor = (index_tensor - vmin) / (vmax - vmin)
    else:
        min_val, max_val = torch.min(index_tensor), torch.max(index_tensor)
        if max_val > min_val:
            normalized_tensor = (index_tensor - min_val) / (max_val - min_val)
        else:
            normalized_tensor = torch.zeros_like(index_tensor)

    normalized_tensor = torch.clamp(normalized_tensor, 0, 1)
    
    colormap = cm.get_cmap(cmap_name)
    colored_image_np = colormap(normalized_tensor.numpy())[:, :, :3]
    colored_image_tensor = torch.from_numpy(colored_image_np).permute(2, 0, 1)
    
    return colored_image_tensor.float()

def create_comparison_image(
    image_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    ae_projected_tensor: torch.Tensor = None,
) -> torch.Tensor:
    """
    Creates a comparison image grid from an input image, ground truth, prediction, spectral indices, and optional AE features.
    If ae_projected_tensor is provided: 3x3 grid with AE projections
    If not provided: 2x3 grid with original spectral indices
    """
    image_tensor = image_tensor.detach().cpu()
    label_tensor = label_tensor.detach().cpu()
    pred_tensor = pred_tensor.detach().cpu()
    
    # --- Top Row: RGB, Ground Truth, Prediction ---
    rgb_image = image_tensor[[3, 2, 1], :, :] * 3.0
    rgb_image = torch.clamp(rgb_image, 0, 1)

    label_rgb = label_tensor.repeat(3, 1, 1)
    label_rgb[0, :, :], label_rgb[2, :, :] = 0, 0



    pred_mask = (pred_tensor > 0.5).float()
    pred_rgb = torch.zeros_like(label_rgb)
    true_positives = (pred_mask * label_tensor).bool()
    false_positives = (pred_mask * (1 - label_tensor)).bool()
    pred_rgb[0, false_positives[0]], pred_rgb[1, true_positives[0]] = 1, 1
    
    top_row = torch.cat([rgb_image, label_rgb, pred_rgb], dim=2)

    # --- Bottom Row: Spectral Indices or AE Features ---
    if ae_projected_tensor is not None:
        # Show AE projected features (typically 3 channels: Band 0, Band 1, Band 2
        ae_projected_tensor = ae_projected_tensor.detach().cpu()
        normalized_tensor = (ae_projected_tensor - torch.min(ae_projected_tensor)) / (torch.max(ae_projected_tensor) - torch.min(ae_projected_tensor))

        bottom_row = torch.cat([normalized_tensor, normalized_tensor, normalized_tensor], dim=2)
        # Third row: Traditional spectral indices
        blue, green, red, nir, swir1, swir2 = image_tensor[0], image_tensor[1], image_tensor[2], image_tensor[3], image_tensor[4], image_tensor[5]
        
        ndvi = safe_divide(nir - red, nir + red)
        ndmi = safe_divide(swir2 - green, swir2 + green)
        mvi = safe_divide(nir - green, swir1 - green)

        ndvi_rgb = colorize_index(ndvi, cmap_name='RdYlGn', vmin=-1, vmax=1)
        ndmi_rgb = colorize_index(ndmi, cmap_name='RdYlGn', vmin=-1, vmax=1)
        mvi_rgb = colorize_index(mvi, cmap_name='RdYlGn', vmin=-1, vmax=8)
        
        third_row = torch.cat([ndvi_rgb, ndmi_rgb, mvi_rgb], dim=2)
        
        final_image = torch.cat([top_row, bottom_row, third_row], dim=1)
    else:
        # Traditional 2x3 grid with spectral indices
        blue, green, red, nir, swir1, swir2 = image_tensor[0], image_tensor[1], image_tensor[2], image_tensor[3], image_tensor[4], image_tensor[5]

        ndvi = safe_divide(nir - red, nir + red)
        ndmi = safe_divide(swir2 - green, swir2 + green)
        mvi = safe_divide(nir - green, swir1 - green)

        ndvi_rgb = colorize_index(ndvi, cmap_name='RdYlGn', vmin=-1, vmax=1)
        ndmi_rgb = colorize_index(ndmi, cmap_name='RdYlGn', vmin=-1, vmax=1)
        mvi_rgb = colorize_index(mvi, cmap_name='RdYlGn', vmin=-1, vmax=8)
        
        bottom_row = torch.cat([ndvi_rgb, ndmi_rgb, mvi_rgb], dim=2)
        
        final_image = torch.cat([top_row, bottom_row], dim=1)

    return final_image 