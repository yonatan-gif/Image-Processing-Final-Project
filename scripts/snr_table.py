"""Quantify each distortion level as a measured SNR (PSNR in dB).

The brief asks for results "per SNR". Our distortions are parameterised by a knob
(sigma / quality); this maps every knob value to the mean PSNR(clean, distorted) over a
sample of images, so the intensity sweep has a physical SNR interpretation.

Run:  python scripts/snr_table.py
Outputs: results/snr_levels.csv, assets/snr_levels.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pets import sample_pet_images  # noqa: E402
from src.distortions import DISTORTIONS  # noqa: E402
from src.metrics import psnr  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

ASSETS = ROOT / "assets"


def main(n_images: int = 40) -> None:
    ASSETS.mkdir(exist_ok=True)
    results_dir = ROOT / "results"; results_dir.mkdir(exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    seed_everything(cfg["data"]["seed"])
    # Same seeded sample as run_keypoints.py, so SNR values describe the SIFT eval images.
    imgs = sample_pet_images(root=str(ROOT / "data"), n=n_images)

    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for dname, (dist_fn, param) in DISTORTIONS.items():
        levels = cfg["distortions"][dname][param]
        psnrs = []
        for level in levels:
            vals = [psnr(im, dist_fn(im, level)) for im in imgs]
            vals = [v for v in vals if np.isfinite(v)]
            mean_psnr = float(np.mean(vals)) if vals else float("inf")
            psnrs.append(mean_psnr)
            rows.append(dict(distortion=dname, param=param, level=level,
                             mean_psnr_db=round(mean_psnr, 2)))
        ax.plot(range(len(levels)), psnrs, marker="o", label=dname)

    ax.set_xlabel("intensity step (weak → strong)")
    ax.set_ylabel("mean PSNR (dB)")
    ax.set_title(f"Distortion level → measured SNR  (mean over {n_images} images)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "snr_levels.png", dpi=110, bbox_inches="tight")

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "snr_levels.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print(df.to_string(index=False))
    print(f"\nSaved {results_dir/'snr_levels.csv'} and {ASSETS/'snr_levels.png'}")


if __name__ == "__main__":
    main()
