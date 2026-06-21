"""Task 1, improvement #2: fine-tune ResNet-50 on DISTORTED data, then re-measure.

Starts from the clean baseline checkpoint, continues training with random distortion
augmentation (noise/blur/jpeg at random levels), then sweeps the same distortions to see
whether model adaptation beats blind restoration.

Run:  python scripts/run_classification_finetune.py [--quick] [--retrain]
Outputs: results/classification_finetune_<distortion>.png  (4 lines: clean / distorted /
         restored / finetuned), results/classification_finetune_results.csv,
         checkpoints/resnet50_pets_distorted.pth
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
from src.utils.ops import ImageOp, RandomDistortOp  # noqa: E402
from src.utils.viz import curve  # noqa: E402


def loader_for(base, idx, pre, img_op, batch, workers, shuffle=False):
    ds = PetClsDataset(base, idx, pre, img_op)
    return DataLoader(ds, batch_size=batch, num_workers=workers, shuffle=shuffle)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    results_dir = ROOT / cfg.get("results_dir", "results"); results_dir.mkdir(exist_ok=True)
    ckpt_dir = ROOT / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    base_ckpt = ckpt_dir / "resnet50_pets_clean.pth"
    ft_ckpt = ckpt_dir / "resnet50_pets_distorted.pth"

    subset = 400 if args.quick else cfg["data"]["subset_size"]
    epochs = 2 if args.quick else cfg["tasks"]["classification"]["finetune_epochs"]
    batch, workers = 32, (0 if args.quick else 4)
    device = get_device()
    print(f"device={device} | subset={subset} | epochs={epochs}\n")

    base = load_pets_classification(root=str(ROOT / "data"), download=False)
    train_idx, val_idx = subset_split(len(base), subset, seed=cfg["data"]["seed"])
    pre = preprocess()

    # Random-distortion augmentation built from the config sweeps.
    specs = [(fn, cfg["distortions"][name][param]) for name, (fn, param) in DISTORTIONS.items()]
    aug = RandomDistortOp(specs)

    model = build_model(num_classes=cfg["tasks"]["classification"]["num_classes"])
    if ft_ckpt.exists() and not args.retrain:
        model.load_state_dict(torch.load(ft_ckpt, map_location="cpu")); model.to(device)
        print(f"Loaded cached fine-tuned checkpoint: {ft_ckpt.name}")
    else:
        if base_ckpt.exists():
            model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
            print("Starting from clean baseline checkpoint.")
        print("Fine-tuning on DISTORTED (augmented) images...")
        fit(model, loader_for(base, train_idx, pre, aug, batch, workers, shuffle=True), device, epochs=epochs)
        torch.save(model.state_dict(), ft_ckpt)
        print(f"Saved {ft_ckpt.name}")

    # Read the baseline sweep (distorted/restored) for the comparison lines, if present.
    base_csv = results_dir / "classification_results.csv"
    base_df = pd.read_csv(base_csv) if base_csv.exists() else None

    rows = []
    for dname, (dist_fn, param) in DISTORTIONS.items():
        levels = cfg["distortions"][dname][param]
        acc_ft = []
        for level in levels:
            p = predict(model, loader_for(base, val_idx, pre, ImageOp(dist_fn, level), batch, workers), device)
            acc_ft.append(top1_accuracy(*p))
            rows.append(dict(distortion=dname, param=param, level=level, acc_finetuned=acc_ft[-1]))
            print(f"  {dname:14s} {param}={level:<6} finetuned={acc_ft[-1]:.3f}")

        series = {"finetuned": acc_ft}
        if base_df is not None:
            sub = base_df[base_df.distortion == dname]
            series["distorted"] = sub.acc_distorted.tolist()
            series["restored"] = sub.acc_restored.tolist()
        curve(levels, series, xlabel=f"{dname} ({param})", ylabel="Top-1 accuracy",
              title=f"ResNet-50: fine-tune vs restore vs {dname}",
              save_path=str(results_dir / f"classification_finetune_{dname}.png"))

    pd.DataFrame(rows).to_csv(results_dir / "classification_finetune_results.csv", index=False)
    print(f"\nSaved fine-tune curves + CSV to {results_dir}/")


if __name__ == "__main__":
    main()
