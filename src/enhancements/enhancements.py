"""Matched cleaners, one per distortion (classical OpenCV defaults).

Matching rule: noise->denoise, blur->deblur, jpeg->dejpeg.
All functions take/return RGB uint8 (H, W, 3). DL upgrades can be added later.

Two denoising protocols are provided (see README, noise-fairness experiment):
  - denoise:       fixed dose (h=10 default) — the level-blind baseline arm.
  - denoise_blind: estimates the noise level from the image itself (Immerkaer 1996)
    and sets h proportionally — pixels-only interface, no side information.
"""
from __future__ import annotations

import cv2
import numpy as np

# Immerkaer's Laplacian-difference mask: cancels locally-linear image structure
# (flat areas, gradients, straight edges -> ~0) while passing pixel-to-pixel noise.
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def estimate_noise_sigma(img: np.ndarray) -> float:
    """Estimate Gaussian noise sigma from a single image (no side information).

    J. Immerkaer, "Fast Noise Variance Estimation", Computer Vision and Image
    Understanding 64(2), 1996 (doi:10.1006/cviu.1996.0060): one 3x3 convolution;
    the sqrt(pi/2)/6 factor converts the mean |response| to the Gaussian sigma.
    Validated on this project's data by scripts/validate_noise_estimator.py.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    resp = cv2.filter2D(gray, -1, _NOISE_KERNEL, borderType=cv2.BORDER_REFLECT)
    h, w = gray.shape
    return float(np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * np.abs(resp).sum())


def denoise(img: np.ndarray, method: str = "nlm", h: float = 10.0) -> np.ndarray:
    """Non-Local Means denoising: average pixels with similar neighborhoods.

    `h` is the patch-similarity tolerance (the dose). OpenCV's colored NLM assumes
    BGR input for its internal luminance/chroma split, so convert around the call.
    """
    if method == "nlm":
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    raise NotImplementedError(f"denoise method '{method}' (DL upgrade TODO)")


def denoise_blind(img: np.ndarray) -> np.ndarray:
    """Blind (self-calibrating) NLM: estimate sigma from the image, dose accordingly.

    h = 0.8 * sigma_hat follows the sigma-proportional rule from the NLM literature
    (Buades et al. / IPOL recommend h in the 0.55-1.0*sigma range). Same pixels-only
    interface as the fine-tuned model at evaluation time.
    """
    sigma_hat = estimate_noise_sigma(img)
    h = float(np.clip(0.8 * sigma_hat, 3.0, 60.0))
    return denoise(img, h=h)


def deblur(img: np.ndarray, method: str = "unsharp", amount: float = 1.5) -> np.ndarray:
    """Unsharp masking: boost high frequencies = sharpen."""
    if method == "unsharp":
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        # addWeighted already saturates to uint8, so no extra clip/cast is needed.
        return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
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
