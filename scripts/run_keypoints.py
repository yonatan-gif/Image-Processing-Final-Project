"""Task 3 runner: SIFT keypoint robustness to distortions + restoration recovery.

Baseline -> Distortion sweep -> Restoration (matched cleaner), for noise/blur/JPEG.
Metrics: repeatability rate and matching score of each image against its own clean
version, averaged over a seeded sample of Oxford-IIIT Pet images (mean +- std across
images). The sample is the same one the PSNR/SNR calibration uses (scripts/snr_table.py),
so the per-SNR axis and the keypoint metrics describe identical data. CPU-only.

Run:  python scripts/run_keypoints.py
Outputs: results/keypoints_<distortion>.png           (repeatability curves, mean +- std)
         results/keypoints_matching_<distortion>.png  (matching-score curves, mean +- std)
         results/keypoints_grid_<distortion>.png      (clean | distorted | restored, sample image)
         results/keypoints_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pets import sample_pet_images  # noqa: E402
from src.distortions import DISTORTIONS  # noqa: E402
from src.enhancements import ENHANCEMENTS  # noqa: E402
from src.metrics import repeatability_rate, matching_score  # noqa: E402
from src.tasks.keypoints import detect_and_describe, match  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from src.utils.viz import curve, before_after_grid  # noqa: E402


def evaluate(imgs: list[np.ndarray], cfg: dict, results_dir: Path) -> pd.DataFrame:
    clean = [detect_and_describe(im) for im in imgs]  # (keypoints, descriptors) per image
    n_clean = [len(kp) for kp, _ in clean]
    print(f"clean keypoints per image: mean {np.mean(n_clean):.0f} "
          f"(min {min(n_clean)}, max {max(n_clean)})\n")
    rows = []

    for dname, (dist_fn, param) in DISTORTIONS.items():
        levels = cfg["distortions"][dname][param]
        rep_d, rep_r, ms_d, ms_r = ([] for _ in range(4))              # per-level means
        rep_d_sd, rep_r_sd, ms_d_sd, ms_r_sd = ([] for _ in range(4))  # per-level stds
        strongest = None

        for level in levels:
            per_img = {k: [] for k in ("rd", "rr", "md", "mr")}
            for im, (kp_c, desc_c), nc in zip(imgs, clean, n_clean):
                distorted = dist_fn(im, level)
                restored = ENHANCEMENTS[dname](distorted)
                kp_d, desc_d = detect_and_describe(distorted)
                kp_r, desc_r = detect_and_describe(restored)
                per_img["rd"].append(repeatability_rate(kp_c, kp_d))
                per_img["rr"].append(repeatability_rate(kp_c, kp_r))
                per_img["md"].append(matching_score(len(match(desc_c, desc_d)), nc))
                per_img["mr"].append(matching_score(len(match(desc_c, desc_r)), nc))
            strongest = (imgs[0], dist_fn(imgs[0], level),
                         ENHANCEMENTS[dname](dist_fn(imgs[0], level)))

            for means, sds, key in ((rep_d, rep_d_sd, "rd"), (rep_r, rep_r_sd, "rr"),
                                    (ms_d, ms_d_sd, "md"), (ms_r, ms_r_sd, "mr")):
                means.append(float(np.mean(per_img[key])))
                sds.append(float(np.std(per_img[key])))
            rows.append(dict(distortion=dname, param=param, level=level,
                             repeatability_distorted=rep_d[-1], repeatability_distorted_std=rep_d_sd[-1],
                             repeatability_restored=rep_r[-1], repeatability_restored_std=rep_r_sd[-1],
                             matching_distorted=ms_d[-1], matching_distorted_std=ms_d_sd[-1],
                             matching_restored=ms_r[-1], matching_restored_std=ms_r_sd[-1]))
            print(f"  {dname:14s} {param}={level:<6} "
                  f"rep: {rep_d[-1]:.2f}±{rep_d_sd[-1]:.2f} -> {rep_r[-1]:.2f}±{rep_r_sd[-1]:.2f}   "
                  f"match: {ms_d[-1]:.2f}±{ms_d_sd[-1]:.2f} -> {ms_r[-1]:.2f}±{ms_r_sd[-1]:.2f}")

        curve(levels, {"distorted": rep_d, "restored": rep_r},
              xlabel=f"{dname} ({param})", ylabel="repeatability rate",
              title=f"SIFT repeatability vs {dname} ({len(imgs)} images, mean ± std)",
              save_path=str(results_dir / f"keypoints_{dname}.png"),
              std={"distorted": rep_d_sd, "restored": rep_r_sd})

        curve(levels, {"distorted": ms_d, "restored": ms_r},
              xlabel=f"{dname} ({param})", ylabel="matching score",
              title=f"SIFT matching score vs {dname} ({len(imgs)} images, mean ± std)",
              save_path=str(results_dir / f"keypoints_matching_{dname}.png"),
              std={"distorted": ms_d_sd, "restored": ms_r_sd})

        before_after_grid(list(strongest),
                          ["clean", f"{dname} (strongest)", "restored"],
                          save_path=str(results_dir / f"keypoints_grid_{dname}.png"))

    return pd.DataFrame(rows)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    seed_everything(cfg["data"]["seed"])
    results_dir = ROOT / cfg.get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    n_images = cfg["tasks"]["keypoints"].get("n_images", 40)
    imgs = sample_pet_images(root=str(ROOT / "data"), n=n_images)
    print(f"{n_images} seeded Oxford-IIIT Pet images (baseline repeatability = 1.00)")

    df = evaluate(imgs, cfg, results_dir)
    df.to_csv(results_dir / "keypoints_results.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(df.to_string(index=False))
    print(f"\nSaved curves, grids, and CSV to {results_dir}/")


if __name__ == "__main__":
    main()
