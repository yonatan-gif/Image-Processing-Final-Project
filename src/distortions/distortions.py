"""Image distortions with a single intensity ("SNR") knob each.

All functions take and return an RGB uint8 numpy array (H, W, 3) in [0, 255].
"""
from __future__ import annotations

import cv2
import numpy as np


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add zero-mean Gaussian noise. Intensity = sigma (in 0-255 pixel units)."""
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur. Intensity = kernel std (sigma); kernel size derived from sigma."""
    if sigma <= 0:
        return img.copy()
    ksize = int(2 * round(3 * sigma) + 1)  # cover +-3 sigma, force odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def apply_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """JPEG compression artifacts. Intensity = quality factor (lower = worse)."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


# Registry: name -> (function, intensity-param-name)
DISTORTIONS = {
    "gaussian_noise": (add_gaussian_noise, "sigma"),
    "blur": (apply_blur, "sigma"),
    "jpeg": (apply_jpeg, "quality"),
}
