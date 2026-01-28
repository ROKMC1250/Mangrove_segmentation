"""
AE-Enhanced SMP models for mangrove segmentation.

This module provides AE-enhanced versions of various SMP models,
combining satellite images with AlphaEarth (AE) features.

Author: Assistant
"""

import torch
import torch.nn as nn
from model.smp_models import create_model


class AEEnhancedSMP(nn.Module):
    """
    Generic AE-Enhanced SMP model that combines satellite images with AE features.
    
    This model projects AE features to a lower dimension and concatenates them
    with satellite images before feeding to any SMP model.
    """
    
    def __init__(self, base_model_name, encoder_name, in_img=6, ae_dim=64, D=3, classes=1, bce_weight=2.0, **kwargs):
        super().__init__()
        
        # Store bce_weight for trainer (not passed to base model)
        self.bce_weight = bce_weight
        
        # Projection layer for AE features
        self.proj = nn.Conv2d(ae_dim, D, kernel_size=1, bias=True)
        self.proj_activation = nn.ReLU(inplace=True)  # Ensure positive outputs
        
        # Initialize projection layer
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj.bias)
        
        # Create the base SMP model with combined input channels
        # Remove bce_weight from kwargs as it's not a model parameter
        model_kwargs = {k: v for k, v in kwargs.items() if k != 'bce_weight'}
        
        self.seg = create_model(
            base_model_name,
            encoder_name=encoder_name,
            in_channels=in_img + D,  # 6 satellite + D projected AE channels
            classes=classes,
            encoder_weights=None,
            **model_kwargs
        )
    
    def forward(self, img, ae):
        """
        Forward pass combining satellite images and AE features.
        
        Args:
            img: Satellite images (B, 6, H, W)
            ae: AE features (B, 64, H, W)
        
        Returns:
            logits: Segmentation output (B, classes, H, W)
            Z: Projected AE features (B, D, H, W)
        """
        # Project AE features to lower dimension
        Z_raw = self.proj(ae)
        Z = self.proj_activation(Z_raw)  # BxDxhxw, positive values only
        
        # Concatenate satellite images with projected AE features
        x = torch.cat([img, Z], dim=1)  # (B, 6+D, H, W)
        
        # Pass through segmentation model
        logits = self.seg(x)
        
        return logits, Z


def create_ae_enhanced_model(base_model_name, **kwargs):
    """
    Create an AE-enhanced version of any SMP model.
    
    Args:
        base_model_name: Name of the base SMP model
        **kwargs: Model arguments
    
    Returns:
        AEEnhancedSMP instance
    """
    return AEEnhancedSMP(base_model_name=base_model_name, **kwargs)
