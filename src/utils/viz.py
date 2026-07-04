"""Plotting helpers: before/after grids, degradation/recovery curves, per-class heatmaps."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def before_after_grid(images: Sequence[np.ndarray], titles: Sequence[str], save_path: str | None = None):
    """Show a single row of images (e.g. clean | distorted | restored)."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def heatmap(matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str],
            xlabel: str, title: str, cbar_label: str, save_path: str | None = None):
    """Class x intensity heatmap (rows = classes, columns = distortion levels)."""
    fig, ax = plt.subplots(figsize=(7, 10.5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=cbar_label)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def curve(x: Sequence[float], series: dict[str, Sequence[float]], xlabel: str, ylabel: str,
          title: str, save_path: str | None = None,
          std: dict[str, Sequence[float]] | None = None):
    """Plot metric-vs-intensity curves (one line per series: baseline/distorted/restored/finetuned).

    `std` (optional) maps a series name to per-point standard deviations, drawn as a
    shaded mean +- std band around that line.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ys in series.items():
        (line,) = ax.plot(x, ys, marker="o", label=name)
        if std and name in std:
            ys_a, sd_a = np.asarray(ys), np.asarray(std[name])
            ax.fill_between(x, ys_a - sd_a, ys_a + sd_a, alpha=0.15, color=line.get_color())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
