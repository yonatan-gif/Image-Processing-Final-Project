"""Quantify each distortion level as a measured SNR (and PSNR) in dB.

The brief asks for results "per SNR". Our distortions are parameterised by a knob
(sigma / quality); this maps every knob value to the mean SNR and PSNR of (clean, distorted)
over a seeded image sample, so the intensity sweep has a physical dB interpretation.

SNR uses the image's own signal power (10 log10 mean(I^2)/MSE); PSNR fixes the numerator at
the 255^2 peak. SNR is therefore a per-image offset below PSNR — a monotone re-scaling of the
same fidelity axis, kept alongside for reference.

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
from src.metrics import psnr, snr  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

ASSETS = ROOT / "assets"


def main(n_images: int = 40) -> None:
    ASSETS.mkdir(exist_ok=True)
    results_dir = ROOT / "results"; results_dir.mkdir(exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    seed_everything(cfg["data"]["seed"])
    # Same seeded sample as run_keypoints.py, so the dB values describe the SIFT eval images.
    imgs = sample_pet_images(root=str(ROOT / "data"), n=n_images)

    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for dname, (dist_fn, param) in DISTORTIONS.items():
        levels = cfg["distortions"][dname][param]
        snrs = []
        for level in levels:
            pairs = [(im, dist_fn(im, level)) for im in imgs]
            snr_vals = [snr(c, d) for c, d in pairs if np.isfinite(snr(c, d))]
            psnr_vals = [psnr(c, d) for c, d in pairs if np.isfinite(psnr(c, d))]
            mean_snr = float(np.mean(snr_vals)) if snr_vals else float("inf")
            mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else float("inf")
            snrs.append(mean_snr)
            rows.append(dict(distortion=dname, param=param, level=level,
                             mean_snr_db=round(mean_snr, 2), mean_psnr_db=round(mean_psnr, 2)))
        ax.plot(range(len(levels)), snrs, marker="o", label=dname)

    ax.set_xlabel("intensity step (weak → strong)")
    ax.set_ylabel("mean SNR (dB)")
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
