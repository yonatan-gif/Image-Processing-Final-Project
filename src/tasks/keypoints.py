"""SIFT interest-point detection (Task 3, classical, CPU-only).

Metrics: repeatability rate and matching score (see src/metrics). Requires opencv-contrib-python.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_and_describe(img: np.ndarray):
    """Return (keypoints, descriptors) for an RGB uint8 image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)


def match(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75):
    """Lowe-ratio-tested matches between two descriptor sets.

    The ratio test needs two neighbours per query, so fewer than 2 descriptors on
    either side (e.g. strong blur leaves 0-1 keypoints) means no testable matches.
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc1, desc2, k=2)
    return [m for m, n in knn if m.distance < ratio * n.distance]
