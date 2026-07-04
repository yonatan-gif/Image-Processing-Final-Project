"""Validate the Immerkaer noise-level estimator before it is used for blind denoising.

Protocol: take the same seeded 40-image Pet sample used by the SIFT/SNR experiments,
inject Gaussian noise at known sigma (including sigma=0 and an off-sweep sigma=60),
estimate sigma_hat from the noisy pixels alone, and compare. The estimator earns its
place in the blind-restoration arm only if sigma_hat tracks the true sigma monotonically
with a spread small enough to dose NLM correctly.

Reference: J. Immerkaer, "Fast Noise Variance Estimation", CVIU 64(2), 1996.

Run:  python scripts/validate_noise_estimator.py
Outputs: results/noise_estimator_validation.csv, assets/noise_estimator_validation.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pets import sample_pet_images  # noqa: E402
from src.distortions import add_gaussian_noise  # noqa: E402
from src.enhancements import estimate_noise_sigma  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

SIGMAS = [0, 5, 10, 20, 40, 60, 80]


def main() -> None:
    seed_everything(42)
    results_dir = ROOT / "results"; results_dir.mkdir(exist_ok=True)
    assets = ROOT / "assets"; assets.mkdir(exist_ok=True)
    imgs = sample_pet_images(root=str(ROOT / "data"), n=40)

    rows = []
    for sigma in SIGMAS:
        ests = [estimate_noise_sigma(add_gaussian_noise(im, sigma) if sigma else im)
                for im in imgs]
        rows.append(dict(true_sigma=sigma, est_mean=float(np.mean(ests)),
                         est_std=float(np.std(ests)),
                         est_min=float(np.min(ests)), est_max=float(np.max(ests))))
        print(f"  true sigma={sigma:<3} -> estimated {rows[-1]['est_mean']:6.1f} "
              f"± {rows[-1]['est_std']:.1f}  (min {rows[-1]['est_min']:.1f}, "
              f"max {rows[-1]['est_max']:.1f})")

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "noise_estimator_validation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(df.true_sigma, df.est_mean, yerr=df.est_std, marker="o",
                capsize=3, label="estimated $\\hat{\\sigma}$ (mean ± std, 40 images)")
    ax.plot(SIGMAS, SIGMAS, linestyle="--", color="gray", label="identity ($\\hat{\\sigma}=\\sigma$)")
    ax.set_xlabel("true noise sigma (injected)")
    ax.set_ylabel("estimated sigma (from pixels only)")
    ax.set_title("Immerkaer noise-level estimation on 40 Pet images")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(assets / "noise_estimator_validation.png", dpi=120, bbox_inches="tight")
    print(f"\nSaved CSV to {results_dir}/ and figure to assets/")


if __name__ == "__main__":
    main()
