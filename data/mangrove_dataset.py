import os
import torch
import rasterio
import numpy as np
import random
import warnings
import logging
from torch.utils.data import Dataset
from typing import Optional, List, Tuple
from torchvision.transforms import functional as F

# Suppress rasterio TIFF warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
logging.getLogger('rasterio').setLevel(logging.ERROR)
logging.getLogger('rasterio._env').setLevel(logging.ERROR)

class MangroveDataset(Dataset):
    """
    Custom PyTorch Dataset for mangrove segmentation.

    Loads 13-band Sentinel-2 satellite images and their corresponding single-band masks.
    Bands: B1(Aerosol), B2(Blue), B3(Green), B4(Red), B5-B7(Red Edge), B8(NIR), 
           B8A(Narrow NIR), B9(Water Vapor), B11(SWIR1), B12(SWIR2), SCL
    """
    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[callable] = None, shuffle: bool = True, target_size: int = 256):
        """
        Args:
            root_dir (str): The root directory of the processed GMW dataset.
                            (e.g., 'datasets/GMW')
            split (str): The dataset split, 'train' or 'validation'.
            transform (callable, optional): Optional transform to be applied on a sample.
            shuffle (bool): Whether to shuffle the dataset order. Default: True.
            target_size (int): Target image size for resizing. Default: 256.
        """
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'masks')
        self.transform = transform
        self.shuffle = shuffle
        self.target_size = target_size

        # Get sorted list of image files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.tif')])
        
        # Verify that each image has a corresponding label
        self.label_files = []
        for img_file in self.image_files:
            label_file = img_file.replace('.tif', '_masks.tif')
            label_path = os.path.join(self.label_dir, label_file)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label for image {img_file} not found at {label_path}")
            self.label_files.append(label_file)
        
        # Create paired list and shuffle if requested
        self.file_pairs = list(zip(self.image_files, self.label_files))
        if self.shuffle:
            random.shuffle(self.file_pairs)
            # Update individual lists after shuffling
            self.image_files, self.label_files = zip(*self.file_pairs)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get file paths
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Open image and label using rasterio
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32) / 10000  # Shape: (13, H, W)
        
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.float32) # Shape: (H, W)
            
        # Binarize the label: 0 for non-mangrove, 1 for mangrove
        label = (label > 0).astype(np.float32)

        # Add a channel dimension to the label
        label = np.expand_dims(label, axis=0) # Shape: (1, H, W)

        # Convert to PyTorch tensors before resizing
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label)

        # Resize image and label to target size
        # Using 'bilinear' for image and 'nearest' for mask to preserve label values
        image_tensor = F.resize(image_tensor, [self.target_size, self.target_size], antialias=True)
        label_tensor = F.resize(label_tensor, [self.target_size, self.target_size], antialias=False)

        # Apply transformations if any
        if self.transform:
            # Note: Current transforms might be expecting numpy arrays.
            # This part may need adjustment if issues arise.
            transformed = self.transform(image=image_tensor, mask=label_tensor)
            image_tensor = transformed['image']
            label_tensor = transformed['mask']
        
        return image_tensor, label_tensor 