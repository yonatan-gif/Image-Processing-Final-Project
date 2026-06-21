"""Qualitative Task-3 figure: SIFT keypoints on clean vs. each distortion (strongest level).

Shows, per distortion, how many keypoints survive and where — the visual counterpart to the
repeatability curves.

Run:  python scripts/keypoints_viz.py
Outputs: assets/keypoints_visual.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import yaml
from skimage.data import chelsea

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distortions import DISTORTIONS  # noqa: E402
from src.tasks.keypoints import detect_and_describe  # noqa: E402

ASSETS = ROOT / "assets"


def draw_kps(img):
    kps, _ = detect_and_describe(img)
    vis = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return vis, len(kps)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    img = chelsea()

    panels = []
    vis, n = draw_kps(img)
    panels.append((vis, f"clean — {n} keypoints"))
    for dname, (dist_fn, param) in DISTORTIONS.items():
        strong = cfg["distortions"][dname][param][-1]
        vis, n = draw_kps(dist_fn(img, strong))
        panels.append((vis, f"{dname} {param}={strong} — {n} kps"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (vis, title) in zip(axes, panels):
        ax.imshow(vis); ax.set_title(title, fontsize=10); ax.axis("off")
    fig.suptitle("SIFT keypoints: clean vs. strongest distortion", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSETS / "keypoints_visual.png", dpi=110, bbox_inches="tight")
    print(f"saved {ASSETS / 'keypoints_visual.png'}")


if __name__ == "__main__":
    main()
