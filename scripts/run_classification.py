"""Task 1 runner: ResNet-50 breed classification robustness.

Pipeline:
  1. Fine-tune ResNet-50 on CLEAN pet images  -> this is the baseline (cached to checkpoints/).
  2. Baseline Top-1 on a clean val subset.
  3. Distortion sweep (noise/blur/jpeg): Top-1 on distorted vs. restored inputs.
Improvement #2 (fine-tune on distorted) lives in run_classification_finetune.py / Colab.

Run:  python scripts/run_classification.py            # uses config defaults
      python scripts/run_classification.py --quick    # tiny subset/epochs for a fast check
Outputs: results/classification_<distortion>.png, results/classification_results.csv,
         checkpoints/resnet50_pets_clean.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pets import load_pets_classification, subset_split, PetClsDataset  # noqa: E402
from src.distortions import DISTORTIONS  # noqa: E402
from src.enhancements import ENHANCEMENTS  # noqa: E402
from src.metrics import top1_accuracy  # noqa: E402
from src.tasks.classification import build_model, preprocess, get_device, fit, predict  # noqa: E402
from src.utils.viz import curve  # noqa: E402


def make_img_op(dist_fn, level, enhance_fn=None):
    def op(img):
        out = dist_fn(img, level)
        return enhance_fn(out) if enhance_fn is not None else out
    return op


def loader_for(base, idx, pre, img_op, batch_size, workers):
    ds = PetClsDataset(base, idx, pre, img_op)
    return DataLoader(ds, batch_size=batch_size, num_workers=workers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny subset + 2 epochs for a fast check")
    ap.add_argument("--retrain", action="store_true", help="ignore cached checkpoint")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    results_dir = ROOT / cfg.get("results_dir", "results"); results_dir.mkdir(exist_ok=True)
    ckpt_dir = ROOT / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / "resnet50_pets_clean.pth"

    subset = 400 if args.quick else cfg["data"]["subset_size"]
    epochs = 2 if args.quick else cfg["tasks"]["classification"]["finetune_epochs"]
    batch = 32
    workers = 0 if args.quick else 4
    device = get_device()
    print(f"device={device} | subset={subset} | epochs={epochs}\n")

    base = load_pets_classification(root=str(ROOT / "data"), download=False)
    train_idx, val_idx = subset_split(len(base), subset, seed=cfg["data"]["seed"])
    pre = preprocess()

    model = build_model(num_classes=cfg["tasks"]["classification"]["num_classes"])
    if ckpt.exists() and not args.retrain:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device)
        print(f"Loaded cached baseline checkpoint: {ckpt.name}")
    else:
        print("Fine-tuning ResNet-50 on CLEAN pet images (baseline)...")
        fit(model, loader_for(base, train_idx, pre, None, batch, workers), device, epochs=epochs)
        torch.save(model.state_dict(), ckpt)
        print(f"Saved {ckpt.name}")

    # Baseline (clean) Top-1.
    preds, labels = predict(model, loader_for(base, val_idx, pre, None, batch, workers), device)
    clean_acc = top1_accuracy(preds, labels)
    print(f"\nBaseline (clean) Top-1 = {clean_acc:.3f}  on {len(val_idx)} val images\n")

    rows = []
    for dname, (dist_fn, param) in DISTORTIONS.items():
        levels = cfg["distortions"][dname][param]
        acc_dist, acc_rest = [], []
        for level in levels:
            pd_ = predict(model, loader_for(base, val_idx, pre,
                          make_img_op(dist_fn, level), batch, workers), device)
            pr_ = predict(model, loader_for(base, val_idx, pre,
                          make_img_op(dist_fn, level, ENHANCEMENTS[dname]), batch, workers), device)
            acc_dist.append(top1_accuracy(*pd_))
            acc_rest.append(top1_accuracy(*pr_))
            rows.append(dict(distortion=dname, param=param, level=level,
                             acc_distorted=acc_dist[-1], acc_restored=acc_rest[-1]))
            print(f"  {dname:14s} {param}={level:<6} distorted={acc_dist[-1]:.3f}  restored={acc_rest[-1]:.3f}")

        curve(levels,
              {"clean baseline": [clean_acc] * len(levels), "distorted": acc_dist, "restored": acc_rest},
              xlabel=f"{dname} ({param})", ylabel="Top-1 accuracy",
              title=f"ResNet-50 accuracy vs {dname}",
              save_path=str(results_dir / f"classification_{dname}.png"))

    pd.DataFrame(rows).to_csv(results_dir / "classification_results.csv", index=False)
    print(f"\nSaved curves + CSV to {results_dir}/")


if __name__ == "__main__":
    main()
