"""Task 1 (high-level, DL): breed classification with ResNet-50.

Why ResNet-50: deep residual CNN, strong ImageNet-pretrained features, cheap to fine-tune
to the 37 Oxford-IIIT Pet breeds. Metric: Top-1 accuracy.
"""
from __future__ import annotations

import torch
import torchvision
from torchvision.models import ResNet50_Weights


def build_model(num_classes: int = 37, pretrained: bool = True) -> torch.nn.Module:
    """ResNet-50 with the classifier head resized to `num_classes`."""
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def preprocess():
    """Standard ResNet-50 inference transforms (resize/crop/normalize)."""
    return ResNet50_Weights.IMAGENET1K_V2.transforms()


# TODO (incremental): train_one_epoch(), evaluate() -> top-1, fine-tune loop (Colab).
