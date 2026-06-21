"""Task 2, improvement #2: fine-tune DeepLabV3 on DISTORTED data, then re-measure mIoU.

Mirrors run_classification_finetune.py: start from the clean baseline, continue training with
random distortion augmentation (image only; masks unchanged), then sweep and compare the
fine-tuned model against the distorted and restored baselines.

Run:  python scripts/run_segmentation_finetune.py [--quick] [--retrain]
Outputs: results/segmentation_finetune_<distortion>.png, results/segmentation_finetune_results.csv,
         checkpoints/deeplabv3_pets_distorted.pth
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

from src.data.pets import load_pets_segmentation, subset_split, PetSegDataset  # noqa: E402
from src.distortions import DISTORTIONS  # noqa: E402
from src.tasks.classification import get_device  # noqa: E402
from src.tasks.segmentation import build_model, fit, evaluate_miou  # noqa: E402
from src.utils.ops import ImageOp, RandomDistortOp  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from src.utils.viz import curve  # noqa: E402


def loader_for(base, idx, size, img_op, batch, workers, shuffle=False):
    return DataLoader(PetSegDataset(base, idx, size, img_op),
                      batch_size=batch, num_workers=workers, shuffle=shuffle)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    seed_everything(cfg["data"]["seed"])
    results_dir = ROOT / cfg.get("results_dir", "results"); results_dir.mkdir(exist_ok=True)
    ckpt_dir = ROOT / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    base_ckpt = ckpt_dir / "deeplabv3_pets_clean.pth"
    ft_ckpt = ckpt_dir / "deeplabv3_pets_distorted.pth"

    subset = 300 if args.quick else cfg["data"]["subset_size"]
    epochs = 2 if args.quick else cfg["tasks"]["segmentation"]["finetune_epochs"]
    size = cfg["data"]["image_size"]
    batch, workers = 8, (0 if args.quick else 4)
    device = get_device()
    print(f"device={device} | subset={subset} | epochs={epochs} | size={size}\n")

    base = load_pets_segmentation(root=str(ROOT / "data"), download=True)
    train_idx, val_idx = subset_split(len(base), subset, seed=cfg["data"]["seed"])

    specs = [(fn, cfg["distortions"][name][param]) for name, (fn, param) in DISTORTIONS.items()]
    aug = RandomDistortOp(specs)

    model = build_model(num_classes=cfg["tasks"]["segmentation"]["num_classes"])
    if ft_ckpt.exists() and not args.retrain:
        model.load_state_dict(torch.load(ft_ckpt, map_location="cpu")); model.to(device)
        print(f"Loaded cached fine-tuned checkpoint: {ft_ckpt.name}")
    else:
        if base_ckpt.exists():
            model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
            print("Starting from clean baseline checkpoint.")
        else:
            print("WARNING: no baseline checkpoint found — training distorted model from pretrained ImageNet/COCO weights.")
        print("Fine-tuning on DISTORTED (augmented) images...")
        fit(model, loader_for(base, train_idx, size, aug, batch, workers, shuffle=True), device, epochs=epochs)
        torch.save(model.state_dict(), ft_ckpt)
        print(f"Saved {ft_ckpt.name}")

    base_csv = results_dir / "segmentation_results.csv"
    base_df = pd.read_csv(base_csv) if base_csv.exists() else None
    if base_df is None:
        print("WARNING: baseline results CSV missing — distorted/restored comparison lines will be omitted.")

    rows = []
    for dname, (dist_fn, param) in DISTORTIONS.items():
        levels = cfg["distortions"][dname][param]
        miou_ft = []
        for level in levels:
            m = evaluate_miou(model, loader_for(base, val_idx, size, ImageOp(dist_fn, level), batch, workers), device)
            miou_ft.append(m)
            rows.append(dict(distortion=dname, param=param, level=level, miou_finetuned=m))
            print(f"  {dname:14s} {param}={level:<6} finetuned={m:.3f}")

        series = {"finetuned": miou_ft}
        if base_df is not None:
            sub = base_df[base_df.distortion == dname]
            series["distorted"] = sub.miou_distorted.tolist()
            series["restored"] = sub.miou_restored.tolist()
        curve(levels, series, xlabel=f"{dname} ({param})", ylabel="mIoU",
              title=f"DeepLabV3: fine-tune vs restore vs {dname}",
              save_path=str(results_dir / f"segmentation_finetune_{dname}.png"))

    pd.DataFrame(rows).to_csv(results_dir / "segmentation_finetune_results.csv", index=False)
    print(f"\nSaved fine-tune curves + CSV to {results_dir}/")


if __name__ == "__main__":
    main()
