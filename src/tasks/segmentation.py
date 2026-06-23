"""Pet-vs-background segmentation with a fine-tuned DeepLabV3-ResNet50 (Task 2). Metric: mIoU."""
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


def fit(model, loader, device, epochs: int = 5, lr: float = 1e-4):
    """Fine-tune DeepLabV3 with Adam + cross-entropy on the 'out' logits."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x)["out"], y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        print(f"  epoch {ep + 1}/{epochs}  loss={total / len(loader.dataset):.4f}")
    return model


@torch.no_grad()
def evaluate_miou(model, loader, device, num_classes: int = 2) -> float:
    """Mean IoU over `loader`, accumulated via a global confusion matrix."""
    model.eval()
    inter = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    for x, y in loader:
        pred = model(x.to(device))["out"].argmax(1).cpu()
        for c in range(num_classes):
            p, g = pred == c, y == c
            inter[c] += (p & g).sum()
            union[c] += (p | g).sum()
    ious = inter[union > 0] / union[union > 0]
    return float(ious.mean()) if len(ious) else 0.0
