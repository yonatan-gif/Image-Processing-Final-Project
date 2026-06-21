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
