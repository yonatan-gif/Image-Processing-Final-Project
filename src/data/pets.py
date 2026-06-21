"""Oxford-IIIT Pet loaders (torchvision built-in).

Provides GT for two tasks:
  - classification: 37 breed labels (target_types="category")
  - segmentation:   trimap mask    (target_types="segmentation")
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet


def load_pets_classification(root: str = "data", split: str = "trainval", download: bool = True):
    """Return OxfordIIITPet with breed-category targets (no transforms applied)."""
    return OxfordIIITPet(
        root=root, split=split, target_types="category", download=download
    )


def subset_split(n: int, subset_size: int, val_frac: float = 0.2, seed: int = 42):
    """Pick `subset_size` random indices from [0, n) and split into (train, val)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[:subset_size]
    cut = int(len(idx) * (1 - val_frac))
    return idx[:cut].tolist(), idx[cut:].tolist()


class PetClsDataset(Dataset):
    """Wrap OxfordIIITPet(category) with an optional numpy image-op before `preprocess`.

    img_op: maps an RGB uint8 array -> RGB uint8 array (e.g. distortion + enhancement).
    Used to evaluate the same trained model on clean / distorted / restored inputs.
    """

    def __init__(self, base, indices: Sequence[int], preprocess: Callable,
                 img_op: Callable | None = None):
        self.base, self.indices = base, list(indices)
        self.preprocess, self.img_op = preprocess, img_op

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        pil, label = self.base[self.indices[i]]
        img = np.asarray(pil.convert("RGB"))
        if self.img_op is not None:
            img = self.img_op(img)
        return self.preprocess(Image.fromarray(img)), int(label)


def load_pets_segmentation(root: str = "data", split: str = "trainval", download: bool = True):
    """Return OxfordIIITPet with trimap segmentation targets (no transforms applied)."""
    return OxfordIIITPet(
        root=root, split=split, target_types="segmentation", download=download
    )
