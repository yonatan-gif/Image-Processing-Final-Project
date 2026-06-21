"""Task 2 (high-level, DL, dense): pet-vs-background segmentation with DeepLabV3-ResNet50.

Why DeepLabV3: atrous convolutions + ASPP capture multi-scale context; pretrained head is
easy to adapt to a 2-class (pet/background) problem. Metric: mIoU.
"""
from __future__ import annotations

import torch
import torchvision
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights


def build_model(num_classes: int = 2, pretrained: bool = True) -> torch.nn.Module:
    """DeepLabV3-ResNet50 with classifier head resized to `num_classes`."""
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
    model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def preprocess():
    """Standard DeepLabV3 inference transforms."""
    return DeepLabV3_ResNet50_Weights.DEFAULT.transforms()


# TODO (incremental): trimap -> binary mask, train loop, evaluate() -> mIoU.
