"""Task metrics. Reported on two axes: per class AND per distortion intensity (SNR)."""
from __future__ import annotations

import numpy as np


def top1_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Classification: fraction of correct Top-1 predictions."""
    preds, labels = np.asarray(preds), np.asarray(labels)
    return float((preds == labels).mean()) if len(labels) else 0.0


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Classification: Top-1 accuracy for each class (NaN where the class is absent)."""
    preds, labels = np.asarray(preds), np.asarray(labels)
    out = np.full(num_classes, np.nan)
    for c in range(num_classes):
        m = labels == c
        if m.any():
            out[c] = (preds[m] == c).mean()
    return out


def psnr(clean: np.ndarray, distorted: np.ndarray) -> float:
    """Peak signal-to-noise ratio (dB) between two uint8 images; higher = more similar."""
    mse = np.mean((clean.astype(np.float64) - distorted.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def mean_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 2) -> float:
    """Segmentation: mean IoU across classes (per-image helper).

    A class absent from both prediction and ground truth scores IoU = 1.0 (standard
    convention), so the mean is always over `num_classes` and is stable across images.
    NOTE: the pipeline reports mIoU via segmentation.evaluate_miou, which accumulates a
    global confusion matrix over the whole eval set; this helper is for per-image use.
    """
    ious = []
    for c in range(num_classes):
        p, g = pred_mask == c, gt_mask == c
        union = np.logical_or(p, g).sum()
        ious.append(1.0 if union == 0 else np.logical_and(p, g).sum() / union)
    return float(np.mean(ious)) if ious else 0.0


def matching_score(num_good_matches: int, num_keypoints: int) -> float:
    """Keypoints: good matches / detected keypoints (proxy for SIFT robustness)."""
    return float(num_good_matches / num_keypoints) if num_keypoints else 0.0


def repeatability_rate(kps_clean, kps_distorted, tol: float = 3.0) -> float:
    """Keypoints: fraction of clean keypoints re-detected within `tol` pixels, one-to-one.

    Our distortions (noise/blur/JPEG) do not move pixels, so the clean->distorted
    correspondence is the identity. Each distorted keypoint may match at most ONE clean
    keypoint (greedy nearest assignment) — without this, a dense cluster of distorted
    keypoints could mark many clean ones as repeated and inflate the rate.
    """
    if not kps_clean or not kps_distorted:
        return 0.0
    pc = np.array([kp.pt for kp in kps_clean])          # (Nc, 2)
    pd = np.array([kp.pt for kp in kps_distorted])      # (Nd, 2)
    d = np.linalg.norm(pc[:, None, :] - pd[None, :, :], axis=2)  # clean rows x distorted cols
    used = np.zeros(len(pd), dtype=bool)
    repeated = 0
    for i in range(len(pc)):
        row = np.where(used, np.inf, d[i])
        j = int(np.argmin(row))
        if row[j] <= tol:
            used[j] = True
            repeated += 1
    return float(repeated / len(pc))
