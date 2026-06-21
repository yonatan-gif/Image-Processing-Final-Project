"""Matched cleaners, one per distortion (classical OpenCV defaults).

Matching rule: noise->denoise, blur->deblur, jpeg->dejpeg.
All functions take/return RGB uint8 (H, W, 3). DL upgrades can be added later.
"""
from __future__ import annotations

import cv2
import numpy as np


def denoise(img: np.ndarray, method: str = "nlm") -> np.ndarray:
    """Non-Local Means denoising: average pixels with similar neighborhoods."""
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    raise NotImplementedError(f"denoise method '{method}' (DL upgrade TODO)")


def deblur(img: np.ndarray, method: str = "unsharp", amount: float = 1.5) -> np.ndarray:
    """Unsharp masking: boost high frequencies = sharpen."""
    if method == "unsharp":
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)
    raise NotImplementedError(f"deblur method '{method}' (Wiener / DL upgrade TODO)")


def dejpeg(img: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Edge-aware smoothing to suppress JPEG block boundaries."""
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    raise NotImplementedError(f"dejpeg method '{method}' (DL upgrade TODO)")


# Map each distortion to its matched cleaner.
ENHANCEMENTS = {
    "gaussian_noise": denoise,
    "blur": deblur,
    "jpeg": dejpeg,
}
