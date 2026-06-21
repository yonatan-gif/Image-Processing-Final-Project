"""Plotting helpers: before/after image grids and degradation/recovery curves."""
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


def curve(x: Sequence[float], series: dict[str, Sequence[float]], xlabel: str, ylabel: str,
          title: str, save_path: str | None = None):
    """Plot metric-vs-intensity curves (one line per series: baseline/distorted/restored/finetuned)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ys in series.items():
        ax.plot(x, ys, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
