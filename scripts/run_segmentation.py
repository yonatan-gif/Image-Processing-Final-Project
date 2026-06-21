"""Task 2 runner: DeepLabV3-ResNet50 pet/background segmentation robustness.

Pipeline:
  1. Fine-tune DeepLabV3 on CLEAN images (baseline), cached to checkpoints/.
  2. Baseline mIoU on a clean val subset.
  3. Distortion sweep (noise/blur/jpeg): mIoU on distorted vs. restored inputs.

Run:  python scripts/run_segmentation.py [--quick] [--retrain]
Outputs: results/segmentation_<distortion>.png, results/segmentation_results.csv,
         checkpoints/deeplabv3_pets_clean.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pets import load_pets_segmentation, subset_split, PetSegDataset  # noqa: E402
from src.distortions import DISTORTIONS  # noqa: E402
from src.enhancements import ENHANCEMENTS  # noqa: E402
from src.tasks.classification import get_device  # noqa: E402
from src.tasks.segmentation import build_model, fit, evaluate_miou  # noqa: E402
from src.utils.ops import ImageOp  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from src.utils.viz import curve  # noqa: E402


def loader_for(base, idx, size, img_op, batch, workers):
    return DataLoader(PetSegDataset(base, idx, size, img_op),
                      batch_size=batch, num_workers=workers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    seed_everything(cfg["data"]["seed"])
    results_dir = ROOT / cfg.get("results_dir", "results"); results_dir.mkdir(exist_ok=True)
    ckpt_dir = ROOT / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / "deeplabv3_pets_clean.pth"

    subset = 300 if args.quick else cfg["data"]["subset_size"]
    epochs = 2 if args.quick else cfg["tasks"]["segmentation"]["finetune_epochs"]
    size = cfg["data"]["image_size"]
    batch = 8
    workers = 0 if args.quick else 4
    device = get_device()
    print(f"device={device} | subset={subset} | epochs={epochs} | size={size}\n")

    base = load_pets_segmentation(root=str(ROOT / "data"), download=True)
    train_idx, val_idx = subset_split(len(base), subset, seed=cfg["data"]["seed"])

    model = build_model(num_classes=cfg["tasks"]["segmentation"]["num_classes"])
    if ckpt.exists() and not args.retrain:
        model.load_state_dict(torch.load(ckpt, map_location="cpu")); model.to(device)
        print(f"Loaded cached baseline checkpoint: {ckpt.name}")
    else:
        print("Fine-tuning DeepLabV3 on CLEAN images (baseline)...")
        fit(model, loader_for(base, train_idx, size, None, batch, workers), device, epochs=epochs)
        torch.save(model.state_dict(), ckpt)
        print(f"Saved {ckpt.name}")

    clean_miou = evaluate_miou(model, loader_for(base, val_idx, size, None, batch, workers), device)
    print(f"\nBaseline (clean) mIoU = {clean_miou:.3f}  on {len(val_idx)} val images\n")

    rows = []
    for dname, (dist_fn, param) in DISTORTIONS.items():
        levels = cfg["distortions"][dname][param]
        miou_dist, miou_rest = [], []
        for level in levels:
            md = evaluate_miou(model, loader_for(base, val_idx, size,
                               ImageOp(dist_fn, level), batch, workers), device)
            mr = evaluate_miou(model, loader_for(base, val_idx, size,
                               ImageOp(dist_fn, level, ENHANCEMENTS[dname]), batch, workers), device)
            miou_dist.append(md); miou_rest.append(mr)
            rows.append(dict(distortion=dname, param=param, level=level,
                             miou_distorted=md, miou_restored=mr))
            print(f"  {dname:14s} {param}={level:<6} distorted={md:.3f}  restored={mr:.3f}")

        curve(levels,
              {"clean baseline": [clean_miou] * len(levels), "distorted": miou_dist, "restored": miou_rest},
              xlabel=f"{dname} ({param})", ylabel="mIoU",
              title=f"DeepLabV3 mIoU vs {dname}",
              save_path=str(results_dir / f"segmentation_{dname}.png"))

    pd.DataFrame(rows).to_csv(results_dir / "segmentation_results.csv", index=False)
    print(f"\nSaved curves + CSV to {results_dir}/")


if __name__ == "__main__":
    main()
