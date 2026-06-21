# Presentation content (source for the final PPT/PDF)

Slide-by-slide verbatim content. Figures referenced live in `assets/`. Numbers marked
_(pending)_ are filled from the GPU fine-tune batch.

---

## Slide 1 — Title
**Robustness of Vision Models to Image Distortions**
Image-Processing & Computer-Vision — Final Project
Yonatan Haba · [github.com/yonatan-gif/Image-Processing-Final-Project](https://github.com/yonatan-gif/Image-Processing-Final-Project)

## Slide 2 — Objective
- Question: how much do common image distortions hurt vision algorithms, and can we recover?
- 3 tasks mixing **low- and high-level**, ≥3 distortions, metrics **per class AND per SNR**.
- Two recovery strategies compared: **clean the image** (restoration) vs **adapt the model** (fine-tune).

## Slide 3 — The bundle (one decision table)
| Axis | Choice |
|---|---|
| Dataset | Oxford-IIIT Pet (GT for classification + segmentation) |
| Task 1 (high, DL) | Classification — ResNet-50 — Top-1 |
| Task 2 (high, DL) | Segmentation — DeepLabV3 — mIoU |
| Task 3 (low, classical) | Interest points — SIFT — repeatability + matching |
| Distortions | Gaussian noise · blur · JPEG (intensity sweeps) |
| Enhancements | NLM denoise · unsharp deblur · bilateral de-JPEG |

## Slide 4 — Dataset & EDA
- 3,680 images · 37 breeds · ~100/breed (balanced) · median 500×375.
- GT shown: breed label + head bbox + pet mask.
- Figure: `assets/eda_samples.png`

## Slide 5 — Distortions and measured SNR
- Each knob → measured PSNR (noise σ 5→80 = 34→12 dB; blur 0.5→8 = 39→22 dB; JPEG 90→10 = 39→27 dB).
- Figures: `assets/distortion_grid.png`, `assets/snr_levels.png`

## Slide 6 — Method: experiment design
For each (task × distortion): **Baseline (clean) → Distorted (sweep) → Restored → Fine-tuned**.
- Restoration = matched classical cleaner, model frozen.
- Fine-tune = continue training on distortion-augmented data (DL tasks only).

## Slide 7 — Task 3: SIFT keypoints (low-level)
- Blur erases keypoints (559→21); noise/JPEG spawn spurious ones (→633/678).
- Restoration helps **only blur** (repeatability 0.44→0.86 at σ=1); hurts noise/JPEG.
- Figures: `assets/keypoints_visual.png`, `assets/keypoints_*.png`

## Slide 8 — Task 1: ResNet-50 classification (high-level)
- Baseline Top-1 = 0.933. Robust to mild distortion; collapses only at strong levels.
- Restoration mostly hurts (denoise 0.93→0.70 at σ=5); deblur helps at strong blur.
- Fine-tune on distorted: _(pending — improvement #2)_.
- Figures: `assets/classification_*.png`

## Slide 9 — Task 2: DeepLabV3 segmentation (high-level)
- Baseline mIoU ≈ 0.92. **More robust than classification** — shape survives better than texture.
- Restoration roughly neutral; small help at strong blur/JPEG.
- Fine-tune on distorted: _(pending — improvement #2)_.

## Slide 10 — Cross-cutting finding
- **Blind classical enhancement is not free.** Tuned for human-visible quality, it removes
  the fine detail models and detectors rely on → often *lowers* task performance.
- **Fine-tuning vs restoration:** _(fill: which recovers more, per task/distortion)_.

## Slide 11 — Conclusions
- Robustness is task-dependent: dense/high-level (segmentation) > recognition > low-level features.
- Match the recovery method to the *consumer* (model), not the human eye.
- Reproducible: one dataset, pretrained models, runs on Apple MPS / free Colab T4.

## Slide 12 — Repo & reproducibility
- `scripts/`: eda · run_keypoints · run_classification(+finetune) · run_segmentation(+finetune)
- `notebooks/colab_finetune.ipynb` — full T4 pipeline
- README = full report with all tables, grids, curves.
