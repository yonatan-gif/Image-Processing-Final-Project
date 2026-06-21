"""Week-7 deliverable: before/after distortion grids on a real pet image.

For each distortion: clean -> increasing intensity -> matched restoration (at strongest).
Saved to assets/ (tracked) so the grid renders in the README report.

Run:  python scripts/make_distortion_grids.py
Outputs: assets/distortion_grid.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pets import load_pets_classification  # noqa: E402
from src.distortions import DISTORTIONS  # noqa: E402
from src.enhancements import ENHANCEMENTS  # noqa: E402

ASSETS = ROOT / "assets"


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    ds = load_pets_classification(root=str(ROOT / "data"), download=False)

    pil, label = ds[0]
    img = np.asarray(pil.convert("RGB"))
    breed = ds.classes[label]

    # columns: clean | weak | mid | strong | restored(strongest)
    rows = list(DISTORTIONS.items())
    ncols = 5
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.2 * ncols, 3.2 * len(rows)))

    for r, (dname, (dist_fn, param)) in enumerate(rows):
        levels = cfg["distortions"][dname][param]
        weak, mid, strong = levels[0], levels[len(levels) // 2], levels[-1]
        restored = ENHANCEMENTS[dname](dist_fn(img, strong))
        panels = [
            (img, "clean"),
            (dist_fn(img, weak), f"{param}={weak}"),
            (dist_fn(img, mid), f"{param}={mid}"),
            (dist_fn(img, strong), f"{param}={strong} (strong)"),
            (restored, "restored"),
        ]
        for c, (im, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(dname, fontsize=12)
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
        axes[r, 1].set_title(panels[1][1], fontsize=11)
        axes[r, 2].set_title(panels[2][1], fontsize=11)
        axes[r, 3].set_title(panels[3][1], fontsize=11)

    fig.suptitle(f"Distortions + matched restoration  ({breed})", fontsize=14)
    fig.tight_layout()
    out = ASSETS / "distortion_grid.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
