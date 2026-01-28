"""
Segmentation Models PyTorch (SMP) model definitions for mangrove segmentation.

This module provides factory functions to create various segmentation models
using the SMP library, supporting both regular and AE-enhanced versions.

Author: Assistant
"""

import segmentation_models_pytorch as smp
import torch.nn as nn


def create_unet_plusplus(encoder_name="resnet34", in_channels=6, classes=1, encoder_weights=None):
    """Create UNet++ model."""
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )


def create_manet(encoder_name="resnet50", in_channels=6, classes=1, encoder_weights=None):
    """Create MAnet model."""
    return smp.MAnet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )


def create_pan(encoder_name="resnet50", in_channels=6, classes=1, encoder_weights=None):
    """Create PAN model."""
    return smp.PAN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )


def create_deeplabv3plus(encoder_name="tu-swinv2_small_window8_256", in_channels=6, classes=1, encoder_weights=None):
    """Create DeepLabV3Plus model (Swin Transformer v2-based)."""
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )


def create_segformer(encoder_name="mit_b2", in_channels=6, classes=1, encoder_weights=None):
    """Create Segformer model."""
    return smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )


def create_fpn(encoder_name="tu-pvt_v2_b2", in_channels=6, classes=1, encoder_weights=None):
    """Create FPN model (PVT v2-based)."""
    return smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )


def create_dpt(encoder_name="tu-vit_base_patch16_224.augreg_in21k", in_channels=6, classes=1, encoder_weights=None):
    """Create DPT model (Vision Transformer-based)."""
    return smp.DPT(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )


def create_upernet(encoder_name="tu-swin_tiny_patch4_window7_224", in_channels=6, classes=1, encoder_weights=None):
    """Create UperNet model (Swin Transformer-based)."""
    return smp.UPerNet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )


# Model factory mapping
MODEL_FACTORY = {
    'UnetPlusPlus': create_unet_plusplus,
    'MAnet': create_manet,
    'PAN': create_pan,
    'DeepLabV3Plus': create_deeplabv3plus,
    'Segformer': create_segformer,
    'FPN': create_fpn,
    'DPT': create_dpt,
    'UperNet': create_upernet,
}


def create_model(model_name, **kwargs):
    """
    Create a model by name using the factory pattern.
    
    Args:
        model_name: Name of the model to create
        **kwargs: Arguments to pass to the model creation function
    
    Returns:
        The created model
    """
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_FACTORY.keys())}")
    
    return MODEL_FACTORY[model_name](**kwargs)
