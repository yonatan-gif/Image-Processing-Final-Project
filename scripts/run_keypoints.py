"""Task 3 runner: SIFT keypoint robustness to distortions + restoration recovery.

Baseline -> Distortion sweep -> Restoration (matched cleaner), for noise/blur/JPEG.
Metrics: repeatability rate and matching score vs. the clean image.
CPU-only, no dataset download (uses skimage's built-in 'chelsea' cat photo by default).

Run:  python scripts/run_keypoints.py
Outputs: results/keypoints_<distortion>.png  (curves)
         results/keypoints_grid_<distortion>.png  (clean | distorted | restored)
         results/keypoints_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from skimage.data import chelsea

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distortions import DISTORTIONS  # noqa: E402
from src.enhancements import ENHANCEMENTS  # noqa: E402
from src.metrics import repeatability_rate, matching_score  # noqa: E402
from src.tasks.keypoints import detect_and_describe, match  # noqa: E402
from src.utils.viz import curve, before_after_grid  # noqa: E402


def load_image() -> np.ndarray:
    """Built-in 451x300 cat photo (RGB uint8) — texture-rich, ideal for SIFT."""
    return np.ascontiguousarray(chelsea())


def evaluate(img: np.ndarray, cfg: dict, results_dir: Path) -> pd.DataFrame:
    kp_clean, desc_clean = detect_and_describe(img)
    n_clean = len(kp_clean)
    rows = []

    for dname, (dist_fn, param) in DISTORTIONS.items():
        levels = cfg["distortions"][dname][param]
        rep_dist, rep_rest, ms_dist, ms_rest = [], [], [], []
        strongest = None

        for level in levels:
            distorted = dist_fn(img, level)
            restored = ENHANCEMENTS[dname](distorted)

            kp_d, desc_d = detect_and_describe(distorted)
            kp_r, desc_r = detect_and_describe(restored)

            rep_dist.append(repeatability_rate(kp_clean, kp_d))
            rep_rest.append(repeatability_rate(kp_clean, kp_r))
            ms_dist.append(matching_score(len(match(desc_clean, desc_d)), n_clean))
            ms_rest.append(matching_score(len(match(desc_clean, desc_r)), n_clean))

            rows.append(dict(distortion=dname, param=param, level=level,
                             repeatability_distorted=rep_dist[-1],
                             repeatability_restored=rep_rest[-1],
                             matching_distorted=ms_dist[-1],
                             matching_restored=ms_rest[-1]))
            strongest = (distorted, restored)  # last level = strongest

        # Per-distortion degradation/recovery curve (repeatability).
        curve(levels,
              {"distorted": rep_dist, "restored": rep_rest},
              xlabel=f"{dname} ({param})", ylabel="repeatability rate",
              title=f"SIFT repeatability vs {dname}",
              save_path=str(results_dir / f"keypoints_{dname}.png"))

        # Before/after grid at strongest level.
        before_after_grid([img, strongest[0], strongest[1]],
                          ["clean", f"{dname} (strongest)", "restored"],
                          save_path=str(results_dir / f"keypoints_grid_{dname}.png"))

    return pd.DataFrame(rows)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    results_dir = ROOT / cfg.get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    img = load_image()
    kp, _ = detect_and_describe(img)
    print(f"Clean image: {img.shape}, {len(kp)} SIFT keypoints (baseline repeatability = 1.00)\n")

    df = evaluate(img, cfg, results_dir)
    df.to_csv(results_dir / "keypoints_results.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(df.to_string(index=False))
    print(f"\nSaved curves, grids, and CSV to {results_dir}/")


if __name__ == "__main__":
    main()
