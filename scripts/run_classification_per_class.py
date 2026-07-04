"""Task 1 per-class x per-intensity sweep: which breeds degrade first?

Evaluates the frozen clean-baseline ResNet-50 on every (distortion, level) and reports
Top-1 accuracy PER BREED, crossing the two required reporting axes. Uses every trainval
image outside the 1,200-image training subset (2,480 images, ~67 per breed) so the
per-breed numbers carry real sample size — the 300-image val subset used for the
aggregate sweep gives only ~8 images per breed, too few to rank breeds.

Requires the cached baseline checkpoint (run scripts/run_classification.py first).

Run:  python scripts/run_classification_per_class.py
Outputs: results/classification_per_class_sweep.csv
         assets/classification_heatmap_<distortion>.png  (37 breeds x levels)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pets import load_pets_classification, subset_split, PetClsDataset  # noqa: E402
from src.distortions import DISTORTIONS  # noqa: E402
from src.metrics import top1_accuracy, per_class_accuracy  # noqa: E402
from src.tasks.classification import build_model, preprocess, get_device, predict  # noqa: E402
from src.utils.ops import ImageOp  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from src.utils.viz import heatmap  # noqa: E402


def loader_for(base, idx, pre, img_op, batch, workers):
    return DataLoader(PetClsDataset(base, idx, pre, img_op),
                      batch_size=batch, num_workers=workers)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    seed_everything(cfg["data"]["seed"])
    results_dir = ROOT / cfg.get("results_dir", "results"); results_dir.mkdir(exist_ok=True)
    assets = ROOT / "assets"; assets.mkdir(exist_ok=True)
    ckpt = ROOT / "checkpoints" / "resnet50_pets_clean.pth"
    if not ckpt.exists():
        sys.exit("Baseline checkpoint missing — run scripts/run_classification.py first.")

    num_classes = cfg["tasks"]["classification"]["num_classes"]
    batch, workers = 64, 4
    device = get_device()

    base = load_pets_classification(root=str(ROOT / "data"), download=True)
    train_idx, _ = subset_split(len(base), cfg["data"]["subset_size"], seed=cfg["data"]["seed"])
    # Every trainval image NOT used for training -> large held-out set for per-breed stats.
    eval_idx = sorted(set(range(len(base))) - set(train_idx))
    pre = preprocess()
    print(f"device={device} | eval images={len(eval_idx)} (~{len(eval_idx) // num_classes}/breed)\n")

    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)

    preds, labels = predict(model, loader_for(base, eval_idx, pre, None, batch, workers), device)
    pca_clean = per_class_accuracy(preds, labels, num_classes)
    print(f"clean: overall Top-1 = {top1_accuracy(preds, labels):.3f}")

    # Rows sorted by clean accuracy (best first) so the degradation gradient reads top-down.
    order = np.argsort(-np.nan_to_num(pca_clean, nan=-1.0))
    breeds = [base.classes[i] for i in order]

    rows = [dict(distortion="none", param="-", level=0, breed=base.classes[c], accuracy=pca_clean[c])
            for c in range(num_classes)]
    for dname, (dist_fn, param) in DISTORTIONS.items():
        levels = cfg["distortions"][dname][param]
        matrix = [pca_clean[order]]
        for level in levels:
            p, l = predict(model, loader_for(base, eval_idx, pre,
                           ImageOp(dist_fn, level), batch, workers), device)
            pca = per_class_accuracy(p, l, num_classes)
            matrix.append(pca[order])
            rows += [dict(distortion=dname, param=param, level=level,
                          breed=base.classes[c], accuracy=pca[c]) for c in range(num_classes)]
            print(f"  {dname:14s} {param}={level:<6} overall={top1_accuracy(p, l):.3f}")

        heatmap(np.stack(matrix, axis=1), breeds, ["clean"] + [str(v) for v in levels],
                xlabel=f"{dname} ({param})",
                title=f"Per-breed Top-1 vs {dname} ({len(eval_idx)} held-out images)",
                cbar_label="Top-1 accuracy",
                save_path=str(assets / f"classification_heatmap_{dname}.png"))

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "classification_per_class_sweep.csv", index=False)

    # Quick summary for the report: do weak breeds collapse first?
    strong = {"gaussian_noise": 80, "blur": 4.0, "jpeg": 10}
    top5, bot5 = [base.classes[i] for i in order[:5]], [base.classes[i] for i in order[-5:]]
    for dname, level in strong.items():
        sub = df[(df.distortion == dname) & (df.level == level)].set_index("breed").accuracy
        print(f"\n{dname} @ {level}: best-on-clean 5 breeds -> {sub[top5].mean():.3f}, "
              f"worst-on-clean 5 breeds -> {sub[bot5].mean():.3f}")

    print(f"\nSaved heatmaps to assets/ and CSV to {results_dir}/")


if __name__ == "__main__":
    main()
