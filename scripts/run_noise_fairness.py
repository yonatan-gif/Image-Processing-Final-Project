"""Noise-fairness experiment: is "restoration hurts / adds nothing" a dose artifact?

The main sweep (Section 9 tables) runs NLM at one fixed dose (h=10) across noise
sigma 5-80. This script re-runs the NOISE restoration arm under two protocols and
compares them against the frozen-model distorted baseline and the fine-tuned model:

  1. restored (fixed h=10)  - the level-blind baseline arm (as in the main tables,
                              but with the RGB->BGR ordering corrected).
  2. restored (blind sigma) - self-calibrating: estimate sigma from the pixels
                              (Immerkaer 1996, validated by
                              scripts/validate_noise_estimator.py), dose h = 0.8*sigma_hat.
                              Pixels-only interface, same as the fine-tuned model.

Fine-tune probes (classification): the fine-tuned model is also evaluated on CLEAN
images (the price of fine-tuning) and at sigma=60 - a level BETWEEN its trained levels
(40, 80) - to test whether it learned the noise axis or memorized the training menu.

Runs from the cached checkpoints; inference only.

Run:  python scripts/run_noise_fairness.py
Outputs: results/noise_fairness_{classification,segmentation,keypoints}.csv
         results/noise_fairness_probes.csv
         assets/{classification,segmentation,keypoints}_noise_fairness.png
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

from src.data.pets import (load_pets_classification, load_pets_segmentation,  # noqa: E402
                           sample_pet_images, subset_split, PetClsDataset, PetSegDataset)
from src.distortions import add_gaussian_noise  # noqa: E402
from src.enhancements import denoise, denoise_blind  # noqa: E402
from src.metrics import top1_accuracy, repeatability_rate, matching_score  # noqa: E402
from src.tasks.classification import (build_model as build_cls, preprocess,  # noqa: E402
                                      get_device, predict)
from src.tasks.keypoints import detect_and_describe, match  # noqa: E402
from src.tasks.segmentation import build_model as build_seg, evaluate_miou  # noqa: E402
from src.utils.ops import ImageOp  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from src.utils.viz import curve  # noqa: E402

PROBE_SIGMA = 60  # between the trained levels 40 and 80 -> interpolation probe


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    seed_everything(cfg["data"]["seed"])
    results_dir = ROOT / cfg.get("results_dir", "results"); results_dir.mkdir(exist_ok=True)
    assets = ROOT / "assets"; assets.mkdir(exist_ok=True)
    levels = cfg["distortions"]["gaussian_noise"]["sigma"]
    device = get_device()
    batch_cls, batch_seg, workers = 32, 8, 4
    print(f"device={device} | noise levels={levels} | probe sigma={PROBE_SIGMA}\n")

    # ---------------- Classification (frozen baseline + fine-tuned probes) ----------------
    print("=== classification ===")
    base_ckpt = ROOT / "checkpoints/resnet50_pets_clean.pth"
    ft_ckpt = ROOT / "checkpoints/resnet50_pets_distorted.pth"
    for p in (base_ckpt, ft_ckpt):
        if not p.exists():
            sys.exit(f"missing checkpoint {p.name} — run the classification scripts first.")

    base = load_pets_classification(root=str(ROOT / "data"), download=True)
    train_idx, val_idx = subset_split(len(base), cfg["data"]["subset_size"], seed=cfg["data"]["seed"])
    pre = preprocess()

    def cls_loader(img_op):
        return DataLoader(PetClsDataset(base, val_idx, pre, img_op),
                          batch_size=batch_cls, num_workers=workers)

    frozen = build_cls(num_classes=cfg["tasks"]["classification"]["num_classes"])
    frozen.load_state_dict(torch.load(base_ckpt, map_location="cpu")); frozen.to(device)

    clean_acc = top1_accuracy(*predict(frozen, cls_loader(None), device))
    print(f"  clean (frozen) = {clean_acc:.3f}")

    rows = []
    acc = {"distorted": [], "restored (fixed h=10)": [], "restored (blind sigma)": []}
    for s in levels:
        a_d = top1_accuracy(*predict(frozen, cls_loader(ImageOp(add_gaussian_noise, s)), device))
        a_f = top1_accuracy(*predict(frozen, cls_loader(ImageOp(add_gaussian_noise, s, denoise)), device))
        a_b = top1_accuracy(*predict(frozen, cls_loader(ImageOp(add_gaussian_noise, s, denoise_blind)), device))
        acc["distorted"].append(a_d)
        acc["restored (fixed h=10)"].append(a_f)
        acc["restored (blind sigma)"].append(a_b)
        rows.append(dict(sigma=s, acc_distorted=a_d, acc_restored_fixed=a_f, acc_restored_blind=a_b))
        print(f"  sigma={s:<4} distorted={a_d:.3f}  fixed={a_f:.3f}  blind={a_b:.3f}")

    series = {"clean baseline": [clean_acc] * len(levels), **acc}
    ft_csv = results_dir / "classification_finetune_results.csv"
    if ft_csv.exists():
        sub = pd.read_csv(ft_csv).query("distortion == 'gaussian_noise'")
        series["fine-tuned"] = sub.acc_finetuned.tolist()
    curve(levels, series, xlabel="gaussian noise (sigma)", ylabel="Top-1 accuracy",
          title="ResNet-50 under noise: fixed vs blind restoration vs fine-tune",
          save_path=str(assets / "classification_noise_fairness.png"))
    pd.DataFrame(rows).to_csv(results_dir / "noise_fairness_classification.csv", index=False)

    # Fine-tune probes: clean cost + held-out level (interpolation).
    print("  fine-tune probes:")
    ft = build_cls(num_classes=cfg["tasks"]["classification"]["num_classes"])
    ft.load_state_dict(torch.load(ft_ckpt, map_location="cpu")); ft.to(device)
    probes = dict(
        clean_frozen=clean_acc,
        clean_finetuned=top1_accuracy(*predict(ft, cls_loader(None), device)),
        sigma60_frozen=top1_accuracy(*predict(frozen, cls_loader(ImageOp(add_gaussian_noise, PROBE_SIGMA)), device)),
        sigma60_finetuned=top1_accuracy(*predict(ft, cls_loader(ImageOp(add_gaussian_noise, PROBE_SIGMA)), device)),
        sigma60_blind_frozen=top1_accuracy(
            *predict(frozen, cls_loader(ImageOp(add_gaussian_noise, PROBE_SIGMA, denoise_blind)), device)),
    )
    for k, v in probes.items():
        print(f"    {k:22s} = {v:.3f}")
    pd.DataFrame([probes]).to_csv(results_dir / "noise_fairness_probes.csv", index=False)
    del frozen, ft

    # ---------------- Segmentation (frozen baseline; fine-tuned from CSV) ----------------
    print("\n=== segmentation ===")
    seg_ckpt = ROOT / "checkpoints/deeplabv3_pets_clean.pth"
    if not seg_ckpt.exists():
        sys.exit("missing deeplabv3_pets_clean.pth — run scripts/run_segmentation.py first.")
    seg_base = load_pets_segmentation(root=str(ROOT / "data"), download=True)
    seg_train, seg_val = subset_split(len(seg_base), cfg["data"]["subset_size"], seed=cfg["data"]["seed"])
    size = cfg["data"]["image_size"]

    def seg_loader(img_op):
        return DataLoader(PetSegDataset(seg_base, seg_val, size, img_op),
                          batch_size=batch_seg, num_workers=workers)

    seg = build_seg(num_classes=cfg["tasks"]["segmentation"]["num_classes"])
    seg.load_state_dict(torch.load(seg_ckpt, map_location="cpu")); seg.to(device)

    seg_clean, _ = evaluate_miou(seg, seg_loader(None), device)
    print(f"  clean (frozen) mIoU = {seg_clean:.3f}")
    rows = []
    miou = {"distorted": [], "restored (fixed h=10)": [], "restored (blind sigma)": []}
    for s in levels:
        m_d, _ = evaluate_miou(seg, seg_loader(ImageOp(add_gaussian_noise, s)), device)
        m_f, _ = evaluate_miou(seg, seg_loader(ImageOp(add_gaussian_noise, s, denoise)), device)
        m_b, _ = evaluate_miou(seg, seg_loader(ImageOp(add_gaussian_noise, s, denoise_blind)), device)
        miou["distorted"].append(m_d)
        miou["restored (fixed h=10)"].append(m_f)
        miou["restored (blind sigma)"].append(m_b)
        rows.append(dict(sigma=s, miou_distorted=m_d, miou_restored_fixed=m_f, miou_restored_blind=m_b))
        print(f"  sigma={s:<4} distorted={m_d:.3f}  fixed={m_f:.3f}  blind={m_b:.3f}")

    series = {"clean baseline": [seg_clean] * len(levels), **miou}
    ft_csv = results_dir / "segmentation_finetune_results.csv"
    if ft_csv.exists():
        sub = pd.read_csv(ft_csv).query("distortion == 'gaussian_noise'")
        series["fine-tuned"] = sub.miou_finetuned.tolist()
    curve(levels, series, xlabel="gaussian noise (sigma)", ylabel="mIoU",
          title="DeepLabV3 under noise: fixed vs blind restoration vs fine-tune",
          save_path=str(assets / "segmentation_noise_fairness.png"))
    pd.DataFrame(rows).to_csv(results_dir / "noise_fairness_segmentation.csv", index=False)
    del seg

    # ---------------- SIFT keypoints (40-image sample, mean +- std) ----------------
    print("\n=== keypoints ===")
    imgs = sample_pet_images(root=str(ROOT / "data"), n=cfg["tasks"]["keypoints"].get("n_images", 40))
    clean_kd = [detect_and_describe(im) for im in imgs]
    n_clean = [len(kp) for kp, _ in clean_kd]

    rows = []
    rep = {k: [] for k in ("distorted", "restored (fixed h=10)", "restored (blind sigma)")}
    rep_sd = {k: [] for k in rep}
    for s in levels:
        per = {k: [] for k in rep}
        mat = {k: [] for k in rep}
        for im, (kp_c, desc_c), nc in zip(imgs, clean_kd, n_clean):
            noisy = add_gaussian_noise(im, s)
            variants = {"distorted": noisy,
                        "restored (fixed h=10)": denoise(noisy),
                        "restored (blind sigma)": denoise_blind(noisy)}
            for k, v in variants.items():
                kp_v, desc_v = detect_and_describe(v)
                per[k].append(repeatability_rate(kp_c, kp_v))
                mat[k].append(matching_score(len(match(desc_c, desc_v)), nc))
        row = dict(sigma=s)
        for k in rep:
            rep[k].append(float(np.mean(per[k]))); rep_sd[k].append(float(np.std(per[k])))
            tag = {"distorted": "distorted", "restored (fixed h=10)": "fixed",
                   "restored (blind sigma)": "blind"}[k]
            row[f"rep_{tag}"] = rep[k][-1]; row[f"rep_{tag}_std"] = rep_sd[k][-1]
            row[f"match_{tag}"] = float(np.mean(mat[k]))
        rows.append(row)
        print(f"  sigma={s:<4} rep: distorted={row['rep_distorted']:.2f}  "
              f"fixed={row['rep_fixed']:.2f}  blind={row['rep_blind']:.2f}")

    curve(levels, rep, xlabel="gaussian noise (sigma)", ylabel="repeatability rate",
          title=f"SIFT under noise: fixed vs blind restoration ({len(imgs)} images, mean ± std)",
          save_path=str(assets / "keypoints_noise_fairness.png"), std=rep_sd)
    pd.DataFrame(rows).to_csv(results_dir / "noise_fairness_keypoints.csv", index=False)

    print(f"\nSaved CSVs to {results_dir}/ and figures to assets/")


if __name__ == "__main__":
    main()
