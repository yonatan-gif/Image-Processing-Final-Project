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


def get_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, device, optimizer, criterion) -> float:
    """One pass over `loader`; returns mean training loss."""
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    """Return (preds, labels) numpy arrays over `loader`."""
    import numpy as np
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def fit(model, train_loader, device, epochs: int = 5, lr: float = 1e-4):
    """Fine-tune the whole network with Adam + cross-entropy."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        print(f"  epoch {ep + 1}/{epochs}  loss={loss:.4f}")
    return model
