import torch.nn as nn
import segmentation_models_pytorch as smp

def UnetPlusPlus(
    encoder_name: str = "resnet34", 
    in_channels: int = 6, 
    classes: int = 1,
    bce_weight: float = 1.0
) -> nn.Module:
    """
    Initializes a U-Net++ model using the segmentation-models-pytorch library.

    Args:
        encoder_name (str): Name of the classification model that will be used as an encoder
                            (e.g., 'resnet34', 'efficientnet-b0').
        encoder_weights (str): One of 'imagenet', 'ssl', 'swsl' or None.
        in_channels (int): Number of input channels for the model (e.g., 6 for our satellite data).
        classes (int): Number of output classes (e.g., 1 for binary segmentation).

    Returns:
        nn.Module: The U-Net++ model.
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        in_channels=in_channels,
        classes=classes,
    )
    return model 