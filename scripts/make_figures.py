"""Regenerate every CPU report figure in one command (reproducibility helper).

Runs the dataset/distortion/keypoint figure generators that need no GPU and writes their
PNGs into assets/ (tracked) and results/. The DL task curves come from the run_* scripts.

Run:  python scripts/make_figures.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CPU_FIGURE_SCRIPTS = [
    "scripts/eda.py",                   # eda_samples, eda_class_distribution
    "scripts/make_distortion_grids.py", # distortion_grid
    "scripts/snr_table.py",             # snr_levels
    "scripts/keypoints_viz.py",         # keypoints_visual
    "scripts/run_keypoints.py",         # keypoints_* curves + grids
]


def main() -> None:
    for script in CPU_FIGURE_SCRIPTS:
        print(f"\n=== {script} ===")
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    print("\nAll CPU report figures regenerated. DL curves: run scripts/run_{classification,segmentation}.py")


if __name__ == "__main__":
    main()
