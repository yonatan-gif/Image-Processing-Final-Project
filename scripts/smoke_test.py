"""Quick sanity check: distortions + matched enhancements run end-to-end on a dummy image.

Run:  python scripts/smoke_test.py
(No dataset download or GPU needed.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distortions import DISTORTIONS  # noqa: E402
from src.enhancements import ENHANCEMENTS  # noqa: E402

INTENSITY = {"gaussian_noise": 30, "blur": 3.0, "jpeg": 20}


def main() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)

    for name, (fn, param) in DISTORTIONS.items():
        distorted = fn(img, INTENSITY[name])
        restored = ENHANCEMENTS[name](distorted)
        assert distorted.shape == img.shape == restored.shape
        assert distorted.dtype == np.uint8 and restored.dtype == np.uint8
        print(f"[OK] {name:14s} ({param}={INTENSITY[name]}) -> distorted -> restored")

    print("\nAll distortion/enhancement pipelines work.")


if __name__ == "__main__":
    main()
