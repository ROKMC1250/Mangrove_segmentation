import os
import torch
import rasterio
import numpy as np
import random
from torch.utils.data import Dataset
from typing import Optional, List, Tuple
from torchvision.transforms import functional as F

class AEDataset(Dataset):
    """
    Custom PyTorch Dataset for AE-enhanced mangrove segmentation.

    Loads 6-band satellite images, their corresponding AE features (64-band), 
    and single-band masks.
    """
    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[callable] = None, shuffle: bool = True, target_size: int = 256):
        """
        Args:
            root_dir (str): The root directory of the processed dataset.
                            Structure expected:
                            root_dir/
                            ├── train/
                            │   ├── images/      # 6-band TIFF files
                            │   ├── alphaearth/ # 64-band AE output TIFF files
                            │   └── masks/       # 1-band mask files
                            └── validation/
                                ├── images/
                                ├── alphaearth/
                                └── masks/
            split (str): The dataset split, 'train' or 'validation'.
            transform (callable, optional): Optional transform to be applied on a sample.
            shuffle (bool): Whether to shuffle the dataset order. Default: True.
            target_size (int): Target image size for resizing. Default: 256.
        """
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.ae_dir = os.path.join(root_dir, split, 'alphaearth')
        self.label_dir = os.path.join(root_dir, split, 'masks')
        self.transform = transform
        self.shuffle = shuffle
        self.target_size = target_size

        # Get sorted list of image files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.tif')])
        
        # Verify that each image has corresponding AE features and labels
        self.ae_files = []
        self.label_files = []
        
        for img_file in self.image_files:
            # AE features file (assuming same name as image)
            ae_file = img_file.replace('.tif', '_alphaearth_embed_vector.tif')  # or adjust naming convention
            ae_path = os.path.join(self.ae_dir, ae_file)
            if not os.path.exists(ae_path):
                # Try alternative naming
                ae_file = img_file  # same name as image
                ae_path = os.path.join(self.ae_dir, ae_file)
                if not os.path.exists(ae_path):
                    raise FileNotFoundError(f"AE features for image {img_file} not found at {ae_path}")
            self.ae_files.append(ae_file)
            
            # Label file
            label_file = img_file.replace('.tif', '_masks.tif')
            label_path = os.path.join(self.label_dir, label_file)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label for image {img_file} not found at {label_path}")
            self.label_files.append(label_file)
        
        # Create triplet list and shuffle if requested
        self.file_triplets = list(zip(self.image_files, self.ae_files, self.label_files))
        if self.shuffle:
            random.shuffle(self.file_triplets)
            # Update individual lists after shuffling
            self.image_files, self.ae_files, self.label_files = zip(*self.file_triplets)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get file paths
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        ae_path = os.path.join(self.ae_dir, self.ae_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Open image using rasterio
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32) / 10000 # Shape: (6, H, W), normalize
            
            # Handle -inf, inf, and nan values in satellite images (conditional for speed)
            if np.any(~np.isfinite(image)):
                image = np.where(np.isneginf(image), 0.0, image)
                image = np.where(np.isposinf(image), 0.0, image)
                image = np.where(np.isnan(image), 0.0, image)
        
        # Open AE features using rasterio (with explicit cache clearing to prevent memory buildup)
        with rasterio.open(ae_path) as src:
            alphaearth = src.read().astype(np.float32)  # Shape: (64, H, W)
            
            # Handle -inf and inf values that can occur at image edges or nodata regions (conditional for speed)
            if np.any(~np.isfinite(alphaearth)):
                alphaearth = np.where(np.isneginf(alphaearth), 0.0, alphaearth)
                alphaearth = np.where(np.isposinf(alphaearth), 0.0, alphaearth)
                alphaearth = np.where(np.isnan(alphaearth), 0.0, alphaearth)
            
            # Force garbage collection of large arrays to prevent memory accumulation
            import gc
            if idx % 50 == 0:  # Every 50 samples, force cleanup
                gc.collect()
            
            # AE features might need normalization depending on their range
            # Uncomment and adjust if needed:
            # alphaearth = alphaearth / alphaearth.max()  # normalize to [0,1]
        
        # Open label using rasterio
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.float32) # Shape: (H, W)
            
        # Binarize the label: 0 for non-mangrove, 1 for mangrove
        label = (label > 0).astype(np.float32)

        # Add a channel dimension to the label
        label = np.expand_dims(label, axis=0) # Shape: (1, H, W)

        # Convert to PyTorch tensors before resizing
        image_tensor = torch.from_numpy(image)
        ae_tensor = torch.from_numpy(alphaearth)
        label_tensor = torch.from_numpy(label)

        # Resize all tensors to target size
        # Using 'bilinear' for image/ae and 'nearest' for mask to preserve label values
        image_tensor = F.resize(image_tensor, [self.target_size, self.target_size], antialias=True)
        ae_tensor = F.resize(ae_tensor, [self.target_size, self.target_size], antialias=True)
        label_tensor = F.resize(label_tensor, [self.target_size, self.target_size], antialias=False)

        # Apply transformations if any
        if self.transform:
            # Note: Current transforms might be expecting numpy arrays.
            # This part may need adjustment if issues arise.
            transformed = self.transform(image=image_tensor, alphaearth=ae_tensor, mask=label_tensor)
            image_tensor = transformed['image']
            ae_tensor = transformed['alphaearth']
            label_tensor = transformed['mask']
        
        return image_tensor, ae_tensor, label_tensor
