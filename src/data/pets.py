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


def sample_pet_images(root: str = "data", n: int = 40, seed: int = 0, download: bool = True):
    """Return `n` seeded RGB uint8 arrays from trainval.

    Shared by the SIFT sweep (run_keypoints.py), its visualization, and the PSNR/SNR
    calibration (snr_table.py), so keypoint metrics and the SNR table describe the exact
    same image sample.
    """
    ds = load_pets_classification(root=root, download=download)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), n, replace=False)
    return [np.asarray(ds[int(i)][0].convert("RGB")) for i in idxs]


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


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def trimap_to_binary(trimap: np.ndarray) -> np.ndarray:
    """Oxford trimap (1=pet, 2=background, 3=border) -> binary pet/background.

    Border (3) is merged into the pet foreground, so background is exactly value 2.
    """
    return (np.asarray(trimap) != 2).astype(np.int64)


class PetSegDataset(Dataset):
    """OxfordIIITPet(segmentation) -> (normalized image tensor, binary mask) at fixed size.

    img_op (optional) maps an RGB uint8 array -> RGB uint8 array (distortion + enhancement),
    applied to the image only; the mask is never distorted.
    """

    def __init__(self, base, indices: Sequence[int], size: int = 256,
                 img_op: Callable | None = None):
        self.base, self.indices, self.size, self.img_op = base, list(indices), size, img_op

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        pil, trimap = self.base[self.indices[i]]
        img = np.asarray(pil.convert("RGB"))
        if self.img_op is not None:
            img = self.img_op(img)
        img_r = Image.fromarray(img).resize((self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.array(img_r)).permute(2, 0, 1).float() / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD

        mask = trimap_to_binary(trimap).astype(np.uint8)
        mask_r = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)
        y = torch.from_numpy(np.array(mask_r)).long()
        return x, y
