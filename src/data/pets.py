"""Oxford-IIIT Pet loaders (torchvision built-in).

Provides GT for two tasks:
  - classification: 37 breed labels (target_types="category")
  - segmentation:   trimap mask    (target_types="segmentation")
"""
from __future__ import annotations

from torchvision.datasets import OxfordIIITPet


def load_pets_classification(root: str = "data", split: str = "trainval", download: bool = True):
    """Return OxfordIIITPet with breed-category targets (no transforms applied)."""
    return OxfordIIITPet(
        root=root, split=split, target_types="category", download=download
    )


def load_pets_segmentation(root: str = "data", split: str = "trainval", download: bool = True):
    """Return OxfordIIITPet with trimap segmentation targets (no transforms applied)."""
    return OxfordIIITPet(
        root=root, split=split, target_types="segmentation", download=download
    )
