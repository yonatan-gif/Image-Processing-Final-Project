# Image-Processing Final Project — Robustness of Vision Models to Distortions

**Goal:** evaluate how robust image-processing / computer-vision algorithms and models are to
controlled image distortions, and how much we can recover with (1) image restoration and
(2) model fine-tuning.

This README doubles as the project **report**: it holds the design decisions, the experiment
matrix, and (once experiments run) the result tables, before/after grids, and degradation/recovery
curves.

---

## 1. Decision table (the chosen end-to-end bundle)

| Axis | Choice | One-line justification |
|---|---|---|
| **Dataset** | Oxford-IIIT Pet (37 breeds) | One small dataset with GT for **both** classification (breed label) and segmentation (trimap mask); built into `torchvision`; Colab-friendly. |
| **Task 1 — high-level, DL** | Image **classification** → **ResNet-50** | Strong pretrained backbone; fine-tune to 37 breeds; metric = **Top-1 accuracy**. |
| **Task 2 — high-level, DL, dense** | Semantic **segmentation** → **DeepLabV3-ResNet50** | Pretrained dense model; fine-tune pet-vs-background; metric = **mIoU**. |
| **Task 3 — low-level, classical** | **Interest-point detection** → **SIFT** | No GT needed, CPU-only; metrics = **repeatability rate** + **matching score**. |
| **Distortion 1** | **Gaussian noise** | Well-defined intensity axis (σ); attacks fine texture → SIFT + classifier. |
| **Distortion 2** | **Blur (Gaussian/defocus)** | Intensity = kernel σ; removes high-frequency detail → keypoints + edges. |
| **Distortion 3** | **JPEG compression** | Intensity = quality factor; blocky artifacts → all tasks. |
| **Enhancement 1 (noise)** | **Denoising** (OpenCV NLM default, optional DL) | Matched cleaner for Gaussian noise. |
| **Enhancement 2 (blur)** | **Deblurring** (unsharp/Wiener default, optional DL) | Matched cleaner for blur. |
| **Enhancement 3 (JPEG)** | **JPEG artifact removal** (bilateral/guided default, optional DL) | Matched cleaner for compression artifacts. |
| **Improvement per DL task** | (1) restoration pre-processing, (2) fine-tuning | SIFT gets restoration only (no weights to train). |
| **Metrics axes** | per **class** AND per **intensity (SNR)** | Required two-axis reporting; plus before/after grids and curves. |

**Task mix check:** 2 high-level DL tasks + 1 low-level classical task ✅ · ≥3 distortions ✅ ·
dataset with GT ✅ · ≥1 DL model ✅.

---

## 2. The math/why in one sentence each

- **Gaussian noise:** add `N(0, σ²)` per pixel; raising σ lowers SNR and buries fine texture.
- **Blur:** convolve with a Gaussian kernel of width σ; larger σ suppresses high frequencies.
- **JPEG:** quantize 8×8 DCT blocks; lower quality factor = coarser quantization = blocky artifacts.
- **Denoising (NLM):** average pixels with similar neighborhoods to cancel zero-mean noise.
- **Deblurring (unsharp/Wiener):** boost high frequencies / invert the blur kernel in frequency domain.
- **JPEG removal:** edge-aware smoothing to suppress block boundaries while keeping structure.
- **ResNet-50:** deep residual CNN; residual connections let very deep nets train stably.
- **DeepLabV3:** atrous (dilated) convolutions + ASPP to segment at multiple scales.
- **SIFT:** scale-space DoG keypoints with gradient-orientation descriptors invariant to scale/rotation.

---

## 3. Experiment matrix

For every (task × distortion) we measure:

1. **Baseline** — clean images.
2. **Distorted** — degradation across the full intensity sweep.
3. **Improvement (1) Restoration** — distorted → matched cleaner → model.
4. **Improvement (2) Fine-tune** — re-train the DL model (classification, segmentation) on distorted/mixed data.

Outputs: result tables (per class + per intensity), degradation/recovery **curves**, and
**before/after** image grids.

---

## 4. Compute & GPU flags

- **CPU (local):** SIFT, all distortions/enhancements, baseline inference on small subsets.
- **GPU needed (free Colab T4):** the two fine-tunes (ResNet-50, DeepLabV3) on a ~1–2k image subset.
  - Colab path: open `notebooks/colab_finetune.ipynb`, set runtime → T4 GPU, run top-to-bottom.

---

## 5. Repository structure

```
.
├── configs/            # experiment config (paths, intensity sweeps, subset sizes)
├── src/
│   ├── data/           # Oxford-IIIT Pet loaders (classification + segmentation)
│   ├── distortions/    # gaussian noise / blur / jpeg + intensity sweeps
│   ├── enhancements/   # denoise / deblur / de-jpeg (classical + optional DL)
│   ├── tasks/          # classification (ResNet-50), segmentation (DeepLabV3), keypoints (SIFT)
│   ├── metrics/        # top-1, mIoU, repeatability, matching score
│   └── utils/          # viz: before/after grids, curves
├── scripts/            # entry points to run baselines / sweeps
├── notebooks/          # Colab fine-tuning
└── results/            # generated tables / figures (git-ignored)
```

---

## 6. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 7. Results

_To be filled as experiments run: tables, before/after grids, degradation & recovery curves._
