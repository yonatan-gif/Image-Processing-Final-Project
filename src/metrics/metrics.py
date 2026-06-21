"""Task metrics. Reported on two axes: per class AND per distortion intensity (SNR)."""
from __future__ import annotations

import numpy as np


def top1_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Classification: fraction of correct Top-1 predictions."""
    preds, labels = np.asarray(preds), np.asarray(labels)
    return float((preds == labels).mean()) if len(labels) else 0.0


def mean_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 2) -> float:
    """Segmentation: mean Intersection-over-Union across classes."""
    ious = []
    for c in range(num_classes):
        p, g = pred_mask == c, gt_mask == c
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def matching_score(num_good_matches: int, num_keypoints: int) -> float:
    """Keypoints: good matches / detected keypoints (proxy for SIFT robustness)."""
    return float(num_good_matches / num_keypoints) if num_keypoints else 0.0


def repeatability_rate(kps_clean, kps_distorted, tol: float = 3.0) -> float:
    """Keypoints: fraction of clean keypoints re-detected within `tol` pixels.

    Our distortions (noise/blur/JPEG) do not move pixels, so the clean->distorted
    correspondence is the identity: a clean keypoint is "repeated" if any distorted
    keypoint lies within `tol` pixels of it.
    """
    if not kps_clean or not kps_distorted:
        return 0.0
    pc = np.array([kp.pt for kp in kps_clean])          # (Nc, 2)
    pd = np.array([kp.pt for kp in kps_distorted])      # (Nd, 2)
    d = np.linalg.norm(pc[:, None, :] - pd[None, :, :], axis=2)  # clean rows x distorted cols
    repeated = int((d.min(axis=1) <= tol).sum())
    return float(repeated / len(kps_clean))
