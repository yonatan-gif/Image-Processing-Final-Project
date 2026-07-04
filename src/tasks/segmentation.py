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
def evaluate_miou(model, loader, device, num_classes: int = 2) -> tuple[float, list[float]]:
    """Mean IoU over `loader`, accumulated via a global confusion matrix.

    Returns (mIoU, per-class IoU list indexed by class id: 0=background, 1=pet), so
    results can be reported per class as well as aggregated.
    """
    model.eval()
    inter = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    for x, y in loader:
        pred = model(x.to(device))["out"].argmax(1).cpu()
        for c in range(num_classes):
            p, g = pred == c, y == c
            inter[c] += (p & g).sum()
            union[c] += (p | g).sum()
    valid = union > 0
    ious = torch.full((num_classes,), float("nan"))
    ious[valid] = inter[valid] / union[valid]
    miou = float(ious[valid].mean()) if valid.any() else 0.0
    return miou, [float(v) for v in ious]
