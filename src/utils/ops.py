"""Picklable image-op used to feed distorted / restored inputs through a model.

A plain closure can't be pickled, so DataLoader(num_workers>0) crashes with it on
macOS/Windows (spawn start method). This small callable class fixes that: it holds
module-level functions (which pickle fine) and applies distortion then optional cleanup.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class ImageOp:
    """Apply a distortion at a fixed level, then an optional matched enhancement."""

    def __init__(self, dist_fn: Callable, level, enhance_fn: Optional[Callable] = None):
        self.dist_fn = dist_fn
        self.level = level
        self.enhance_fn = enhance_fn

    def __call__(self, img: np.ndarray) -> np.ndarray:
        out = self.dist_fn(img, self.level)
        return self.enhance_fn(out) if self.enhance_fn is not None else out


class RandomDistortOp:
    """Training augmentation: apply a random distortion at a random level per call.

    Used for improvement #2 (fine-tune on distorted data) so the model sees the full
    distortion mix during training. `specs` is a list of (dist_fn, [levels]); all
    entries are module-level functions/values, so instances pickle fine.
    """

    def __init__(self, specs):
        self.specs = list(specs)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        fn, levels = self.specs[np.random.randint(len(self.specs))]
        level = levels[np.random.randint(len(levels))]
        return fn(img, level)
