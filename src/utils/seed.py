"""Single entry point to seed all RNGs for reproducible runs.

Covers Python `random`, NumPy's global RNG (used by add_gaussian_noise and
RandomDistortOp), and torch. Note: full bit-exact determinism is not guaranteed on
MPS/CUDA, but seeding removes run-to-run variation in distortion/augmentation sampling.
"""
from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
