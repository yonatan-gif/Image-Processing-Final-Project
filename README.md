# Image-Processing Final Project — Robustness of Vision Models to Distortions

**Goal:** evaluate how robust image-processing / computer-vision algorithms and models are to
controlled image distortions, and how much we can recover with (1) image restoration and
(2) model fine-tuning.

This README is the report — design decisions and references, the experiment setup, and all the
result tables, before/after grids, and degradation/recovery curves are here. Everything was run on
a MacBook (Apple MPS); a free Colab T4 reproduces it.

---

## Abstract

We measure how three common image degradations — **Gaussian noise, blur, and JPEG compression** —
affect vision algorithms at two levels of abstraction: **low-level feature detection** (SIFT
keypoints) and **high-level scene understanding** (breed classification with ResNet-50, semantic
segmentation with DeepLabV3). Using the Oxford-IIIT Pet dataset, we sweep each distortion over
multiple intensities (quantified as PSNR) and, for every condition, compare two recovery
strategies: **classical pre-processing** (NLM denoising, unsharp deblurring, bilateral
de-blocking) and **fine-tuning** the deep models on distorted data. The central question is
whether cleaning the *image* or adapting the *model* recovers more task performance — and our
results show the answer depends strongly on the task's level of abstraction.

## Introduction

Real imaging pipelines rarely receive clean input: sensor noise, defocus/motion blur, and lossy
compression all corrupt an image before any algorithm sees it. Knowing **which methods are robust**
— and whether a cheap restoration step actually helps — matters for building reliable systems.

We follow a controlled design: start from clean ground-truth imagery, apply parametric
distortions, then evaluate three recovery strategies against the clean baseline:

1. **None** — run the model / detector directly on the degraded image.
2. **Restoration** — apply a distortion-specific classical cleaner, then run the frozen model.
3. **Fine-tuning** — adapt the deep model to the distortion domain (DL tasks only).

Mixing a low-level classical task (SIFT, no training, no labels) with two high-level deep tasks
lets us contrast how degradation and recovery behave across the abstraction spectrum. We went in
expecting the matched cleaners to help; the most interesting part of the project turned out to be
the cases where they don't.

---

## 1. Decision table (the chosen end-to-end bundle)

| Axis | Choice | One-line justification |
|---|---|---|
| **Dataset** | [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) ([torchvision loader](https://pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html)) | One small dataset with GT for **both** classification (breed label) and segmentation (trimap mask); also ships head bounding boxes; Colab-friendly. |
| **Task 1 — high-level, DL** | classification → [ResNet-50](https://arxiv.org/abs/1512.03385) ([torchvision](https://pytorch.org/vision/stable/models/resnet.html)) | Strong pretrained backbone; fine-tune to 37 breeds; metric = **Top-1 accuracy**. |
| **Task 2 — high-level, DL, dense** | segmentation → [DeepLabV3-ResNet50](https://arxiv.org/abs/1706.05587) ([torchvision](https://pytorch.org/vision/stable/models/deeplabv3.html)) | Pretrained dense model; fine-tune pet-vs-background; metric = **mIoU**. |
| **Task 3 — low-level, classical** | interest points → [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) ([OpenCV](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)) | No GT needed, CPU-only; metrics = **repeatability rate** + **matching score**. |
| **Distortion 1** | [Gaussian noise](https://en.wikipedia.org/wiki/Gaussian_noise) | Intensity axis = σ; attacks fine texture → SIFT + classifier. |
| **Distortion 2** | [Gaussian blur](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html) | Intensity = kernel σ; removes high-frequency detail. |
| **Distortion 3** | [JPEG compression](https://ieeexplore.ieee.org/document/125072) | Intensity = quality factor; blocky 8×8 DCT artifacts. |
| **Metrics axes** | per **class** AND per **intensity (SNR)** | Required two-axis reporting; plus before/after grids and curves. |

**Requirement check:** 2 high-level DL tasks + 1 low-level classical task ✔ · ≥3 distortions ✔ ·
dataset with GT ✔ · ≥1 DL model ✔ · low + high level mix ✔.

---

## 2. Methods & matched enhancements (with references)

Matching rule: each distortion is paired with the cleaner designed to invert it.

| Distortion | Matched enhancement (default = classical) | Reference |
|---|---|---|
| Gaussian noise | **Denoising** — Non-Local Means (`cv2.fastNlMeansDenoisingColored`) | [Buades et al. 2005 / IPOL](https://www.ipol.im/pub/art/2011/bcm_nlm/) |
| Blur | **Deblurring** — unsharp masking (Wiener / DL optional) | [Unsharp masking](https://en.wikipedia.org/wiki/Unsharp_masking) |
| JPEG | **Artifact removal** — bilateral filter (DL optional) | [Tomasi & Manduchi 1998](https://users.cs.duke.edu/~tomasi/papers/tomasi/tomasiIccv98.pdf) |

**Improvement per DL task:** (1) restoration pre-processing, (2) fine-tuning on distorted data.
SIFT gets restoration only — it is a fixed algorithm with no weights to train.

### 2.1 Degradation model

**Gaussian noise.** Independent zero-mean noise is added to every pixel:

$$I'(x) = I(x) + n,\qquad n \sim \mathcal{N}(0,\sigma^2)$$

Levels: $\sigma \in \{5,10,20,40,80\}$ (0–255 scale) → PSNR 34→12 dB. *Visual effect:* fine grain
that buries texture and edges. *Matched restoration:* NLM denoising (§2.2).

**Blur.** Convolution with a Gaussian kernel of width $\sigma$:

$$I' = I * G_\sigma,\qquad G_\sigma(u,v)=\tfrac{1}{2\pi\sigma^2}e^{-(u^2+v^2)/2\sigma^2}$$

Levels: $\sigma \in \{0.5,1,2,4,8\}$ → PSNR 39→22 dB. *Visual effect:* loss of high-frequency
detail; defocus-like softening. *Matched restoration:* unsharp deblurring (§2.2).

**JPEG.** Each 8×8 block is DCT-transformed and quantized at quality factor $Q$, then decoded.
Levels: $Q \in \{90,70,50,30,10\}$ → PSNR 39→27 dB. *Visual effect:* blocking on 8×8 seams and
ringing near edges. *Matched restoration:* bilateral de-blocking (§2.2).

### 2.2 Restoration methods (why it helps · trade-off)

- **NLM denoising** (noise) — replaces each pixel by a weighted average of pixels with *similar
  patches*. *Why it helps:* exploits natural self-similarity to cancel zero-mean noise.
  *Trade-off:* smooths fine texture, which is exactly what SIFT and the classifier rely on.
- **Unsharp deblurring** (blur) — adds back a scaled high-pass: $I_\text{sharp}=I+a\,(I-G*I)$.
  *Why it helps:* re-boosts the high frequencies blur attenuated. *Trade-off:* amplifies noise and
  cannot recover frequencies fully removed.
- **Bilateral de-blocking** (JPEG) — edge-aware smoothing that averages only spatially *and*
  photometrically close pixels. *Why it helps:* softens block seams in flat regions while keeping
  true edges. *Trade-off:* cannot restore discarded DCT coefficients.

### 2.3 Models
- **ResNet-50:** deep residual CNN; skip connections let very deep nets train stably.
- **DeepLabV3:** atrous (dilated) convolutions + ASPP to segment at multiple scales.
- **SIFT:** scale-space DoG keypoints with gradient-orientation descriptors invariant to scale/rotation.

---

## 3. Dataset & EDA

Oxford-IIIT Pet, `trainval` split, downloaded via torchvision (`python -c "from src.data.pets import load_pets_classification; load_pets_classification(download=True)"`).

| Property | Value |
|---|---|
| Images (trainval) | 3,680 |
| Breeds (classes) | 37 (cats + dogs) |
| Images per class | min 93 · max 100 · mean 99.5 (well balanced) |
| Image size (sampled) | width 140–650 (median 500) · height 134–566 (median 375) |
| Ground truth | breed label · head bounding box · trimap segmentation mask |

**Annotated samples** (breed title · green head bbox · red pet mask overlay) — generated by
`scripts/eda.py`:

![Annotated samples](assets/eda_samples.png)

**Class distribution** (37 breeds, near-uniform ~100 images each):

![Class distribution](assets/eda_class_distribution.png)

> Note: Oxford-IIIT Pet bounding boxes annotate the **head** only, so the green box is small
> relative to the full animal that the segmentation mask covers.

---

## 4. Experiment design & metrics

For every (task × distortion) we evaluate four conditions against the clean baseline:

1. **Baseline** — clean images (upper bound).
2. **Distorted** — full intensity sweep.
3. **Restoration** — distorted → matched cleaner → frozen model.
4. **Fine-tune** — DL model re-trained on distortion-augmented data (DL tasks only).

We define **recovery** as the gain a strategy buys back: 

$$
\Delta = \text{metric}_{\text{recovered}} - \text{metric}_{\text{distorted}}
$$

(positive = helps, negative = hurts). Results are reported **per class** and **per intensity (SNR)**.

### Experimental sample

| Task | Model | Eval set | Metric | GT requirement |
|---|---|---|---|---|
| Feature detection | SIFT | 1 reference image, full sweep | repeatability, matching ratio | none |
| Classification | ResNet-50 | 300 val images | Top-1 (overall + per breed) | breed label |
| Segmentation | DeepLabV3 | 300 val images | mIoU (per class) | trimap mask |

Subsets use a fixed seed (42) for reproducibility.

### Metric definitions

- **Top-1 accuracy:** $\frac{1}{N}\sum_i \mathbb{1}[\hat{y}_i = y_i]$.
- **mIoU:** mean over classes of $\mathrm{IoU}=\dfrac{|P\cap G|}{|P\cup G|}$ (prediction $P$, ground truth $G$).
- **Repeatability:** fraction of clean keypoints with a distorted keypoint within $\tau=3$ px
  (distortions are non-geometric, so the correspondence is the identity).
- **Matching ratio:** Lowe-ratio-passing matches / clean keypoints.
- **PSNR** (quantifies each distortion level as SNR): $10\log_{10}\!\dfrac{255^2}{\mathrm{MSE}}$.

Outputs: result tables (per class + per intensity), degradation/recovery **curves**, and
**before/after** image grids.

---

## 5. Compute & GPU flags

- **CPU (local):** SIFT, all distortions/enhancements, EDA, baseline inference.
- **GPU:** the two fine-tunes (ResNet-50, DeepLabV3). Confirmed to run locally on **Apple MPS**;
  free **Colab T4** is the documented fallback (`notebooks/colab_finetune.ipynb`, runtime → T4).

---

## 6. Repository structure

```
.
├── configs/            # experiment config (paths, intensity sweeps, subset sizes)
├── src/
│   ├── data/           # Oxford-IIIT Pet loaders + distortion-injecting datasets
│   ├── distortions/    # gaussian noise / blur / jpeg + intensity sweeps
│   ├── enhancements/   # denoise / deblur / de-jpeg (classical + optional DL)
│   ├── tasks/          # classification (ResNet-50), segmentation (DeepLabV3), keypoints (SIFT)
│   ├── metrics/        # top-1, mIoU, repeatability, matching score
│   └── utils/          # viz (grids/curves), picklable image-op
├── scripts/            # eda.py + run_{keypoints,classification,segmentation}.py + smoke_test.py
├── notebooks/          # Colab fine-tuning
├── assets/             # README figures (tracked)
└── results/            # generated tables / figures (git-ignored)
```

---

## 7. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py          # sanity check (no GPU/data needed)
python scripts/eda.py                 # dataset stats + annotated grid
python scripts/run_keypoints.py       # Task 3 (CPU)
python scripts/run_classification.py  # Task 1 (MPS/GPU)
python scripts/run_segmentation.py    # Task 2 (MPS/GPU)
```

---

## 8. Progress vs. course weekly plan

| Week | Deliverable | Status |
|---|---|---|
| 1 | Open Git repo + register | ✔ repo · *register in course table (manual)* |
| 2 | Dataset/distortions/tasks table with links | ✔ §1 |
| 3 | Methods/enhancements table with links | ✔ §2 |
| 4 | Download + EDA + annotated grid | ✔ §3, `scripts/eda.py` |
| 5–6 | Run on clean data + measure (per class) | ✔ all 3 tasks (cls 0.93, seg 0.92, SIFT) |
| 7 | Apply distortions + before/after | ✔ `src/distortions`, grids |
| 8 | Measure degradation | ✔ all 3 tasks |
| 9 | Apply enhancements + measure | ✔ all 3 tasks |
| 10–11 | Fine-tune + measure | ✔ classification + segmentation (improvement #2) |
| 12 | Polish README (de-AI pass) | ◻ scheduled last |
| 13 | PPT/PDF + final repo review | ◻ (slide draft in `docs/SLIDES.md`) |

---

## 9. Results

### Distortions + matched restoration (qualitative)

Generated by `scripts/make_distortion_grids.py`:

![Distortion grid](assets/distortion_grid.png)

### Distortion intensity → measured SNR

Each intensity knob maps to a physical SNR (mean PSNR over 40 images, `scripts/snr_table.py`),
so the "per-SNR" sweep has a dB interpretation:

| Distortion | weak → strong (knob) | PSNR (dB), weak → strong |
|---|---|---|
| noise (σ) | 5 → 80 | 34.2 → 11.7 |
| blur (σ) | 0.5 → 8 | 38.6 → 21.7 |
| jpeg (q) | 90 → 10 | 39.2 → 26.9 |

![SNR per level](assets/snr_levels.png)

### Task 1 — ResNet-50 classification (high-level, DL)

Baseline (clean) Top-1 = **0.933** on 300 held-out val images (fine-tuned 5 epochs on 1,200
clean images, Apple MPS).

Top-1 accuracy under each strategy (distorted vs. restored vs. fine-tuned on distorted data):

| Distortion | level | distorted | restored | fine-tuned |
|---|---|---|---|---|
| noise (σ) | 5 | 0.93 | 0.70 | 0.90 |
| noise (σ) | 20 | 0.86 | 0.72 | 0.90 |
| noise (σ) | 80 | 0.22 | 0.24 | **0.75** |
| blur (σ) | 1.0 | 0.90 | 0.87 | 0.90 |
| blur (σ) | 4.0 | 0.44 | 0.50 | **0.79** |
| jpeg (q) | 50 | 0.90 | 0.87 | 0.91 |
| jpeg (q) | 10 | 0.64 | 0.64 | **0.82** |

<p>
<img src="assets/classification_finetune_gaussian_noise.png" width="32%">
<img src="assets/classification_finetune_blur.png" width="32%">
<img src="assets/classification_finetune_jpeg.png" width="32%">
</p>

*Top-1 vs. intensity for noise / blur / JPEG: distorted, restored, and fine-tuned.*

**Findings.** (1) ResNet-50 is **robust to mild distortion** — accuracy barely moves for noise σ≤20,
blur σ≤1, or JPEG q≥50, and collapses only at strong levels. (2) Blind classical restoration
**mostly hurts**: NLM denoising drops σ=5 accuracy 0.93→0.70, and de-JPEG gives nothing back;
only deblurring helps, and only at strong blur. (3) **Fine-tuning is decisively the best recovery**:
at the worst levels it recovers what restoration cannot — noise σ=80 **0.22→0.75**, blur σ=8
0.13→0.59, JPEG q=10 0.64→0.82.

Per-breed accuracy (worst: Birman 0.62, best: Abyssinian 1.00):

![Per-breed accuracy](assets/classification_per_class.png)

### Task 2 — DeepLabV3 segmentation (high-level, DL)

Baseline (clean) mIoU = **0.923** on 300 val images (fine-tuned 5 epochs on 1,200 clean images).

| Distortion | level | distorted | restored | fine-tuned |
|---|---|---|---|---|
| noise (σ) | 5 | 0.923 | 0.901 | 0.912 |
| noise (σ) | 20 | 0.906 | 0.896 | 0.907 |
| noise (σ) | 80 | 0.627 | 0.621 | **0.865** |
| blur (σ) | 1.0 | 0.916 | 0.921 | 0.910 |
| blur (σ) | 8.0 | 0.791 | 0.799 | **0.833** |
| jpeg (q) | 50 | 0.922 | 0.918 | 0.912 |
| jpeg (q) | 10 | 0.847 | 0.873 | **0.891** |

<p>
<img src="assets/segmentation_gaussian_noise.png" width="32%">
<img src="assets/segmentation_blur.png" width="32%">
<img src="assets/segmentation_jpeg.png" width="32%">
</p>

*mIoU vs. intensity for noise / blur / JPEG; clean baseline, distorted, and restored.*

**Findings.** Segmentation is the **most robust** task: mIoU holds ≥0.85 until the strongest
levels and only noise σ=80 causes a large drop (0.923→0.627). Restoration is **roughly neutral**
(small help at strong blur/JPEG, small harm at mild noise). **Fine-tuning again wins where it
matters most** — noise σ=80 0.627→**0.865**, blur σ=8 0.791→0.833, JPEG q=10 0.847→0.891.

### Task 3 — SIFT keypoints (low-level, classical)

Repeatability and matching score vs. distortion intensity, on the clean image and after the
matched cleaner. Baseline (clean vs. clean) repeatability = 1.00.

| Distortion | level | repeatability (distorted → restored) | Δ recovery | matching (distorted → restored) |
|---|---|---|---|---|
| noise (σ) | 5 | 0.74 → 0.26 | **−0.48** | 0.74 → 0.25 |
| noise (σ) | 20 | 0.46 → 0.37 | −0.09 | 0.35 → 0.30 |
| noise (σ) | 80 | 0.21 → 0.21 | 0.00 | 0.10 → 0.11 |
| blur (σ) | 0.5 | 0.74 → **0.83** | +0.08 | 0.74 → 0.72 |
| blur (σ) | 1.0 | 0.38 → **0.80** | **+0.42** | 0.35 → **0.72** |
| blur (σ) | 8.0 | 0.01 → 0.02 | +0.01 | 0.03 → 0.03 |
| jpeg (q) | 90 | 0.83 → 0.45 | −0.38 | 0.84 → 0.42 |
| jpeg (q) | 10 | 0.54 → 0.39 | −0.15 | 0.32 → 0.27 |

*Repeatability uses one-to-one keypoint matching (τ = 3 px); seeded run.*

<p>
<img src="assets/keypoints_gaussian_noise.png" width="32%">
<img src="assets/keypoints_blur.png" width="32%">
<img src="assets/keypoints_jpeg.png" width="32%">
</p>

*Repeatability vs. intensity for noise / blur / JPEG; each plot shows distorted vs. restored.*

Keypoints drawn on the image (`scripts/keypoints_viz.py`): **blur** erases them (559 → 21),
while **noise** and **JPEG** spawn *spurious* unstable ones (559 → 661 / 678) — which is why
both repeatability and matching collapse.

![Keypoint visualization](assets/keypoints_visual.png)

**Key finding:** the matched cleaner helps only for **blur** (deblur restores repeatability from
0.38 → 0.80 at σ=1). For **noise** and **JPEG**, classical cleaners (NLM, bilateral) *hurt*
SIFT — they smooth away the fine texture keypoints sit on. Blind enhancement is not free for
low-level feature tasks; it trades pixel-level fidelity for lost structure.

---

## 10. Discussion

**Robustness diverges with abstraction level.** Across the same distortions, fragility ranks
SIFT keypoints (most fragile) → classification → segmentation (most robust). Fine local structure
is destroyed first; region/shape labels survive longest. At noise σ=80, SIFT repeatability falls to
0.21 and ResNet-50 to 0.22 Top-1, while segmentation mIoU — though it too drops — holds at 0.63.

**Blind classical restoration is not free.** The Δ-recovery columns make this concrete: the matched
cleaner helps only for **blur** (deblur re-adds attenuated high frequencies — repeatability +0.42 at
σ=1). For **noise** and **JPEG**, NLM and bilateral filtering *smooth away* the exact detail the
downstream task relies on, so recovery is often **negative** (SIFT −0.48 at σ=5; ResNet-50 −0.23 at
σ=5). Enhancement tuned for human-visible quality is not tuned for the algorithm consuming it.

**Restoration vs. fine-tuning — the headline result.** Fine-tuning the model on distorted data
**beats restoration decisively**, and the gap is largest exactly where restoration fails — the
strongest distortions:

| Strongest level | distorted | restored | fine-tuned |
|---|---|---|---|
| Classification, noise σ=80 | 0.22 | 0.24 | **0.75** |
| Classification, blur σ=8 | 0.13 | 0.13 | **0.59** |
| Classification, JPEG q=10 | 0.64 | 0.64 | **0.82** |
| Segmentation, noise σ=80 | 0.63 | 0.62 | **0.87** |

At mild distortion the model is already robust so all three agree; at severe distortion restoration
adds nothing while fine-tuning recovers most of the loss.

**Practical takeaway.** Match the recovery method to the *consumer*. Blind classical enhancement
tuned for the human eye can *hurt* the algorithm; **adapting the model** is the more reliable lever,
especially under severe degradation.

---

## 11. Notes from building this

A few things that came up along the way, in case they're useful to anyone repeating the experiment:

- The result that surprised us most is that denoising the input *lowers* SIFT repeatability and
  classifier accuracy at low noise. NLM produces a cleaner-looking picture but removes the micro-texture
  the algorithm keys on. We left it in rather than tuning the cleaner to look good — the negative
  number is the point.
- We expected to need Colab, but a MacBook's MPS backend trained both models fine on a ~1.5k subset,
  so everything runs locally; Colab is just the fallback.
- First version of the repeatability metric counted a clean keypoint as "repeated" if *any* distorted
  keypoint was nearby. That over-counts when noise spawns clusters of keypoints, so it now uses
  one-to-one matching. The fix lowered the numbers a few points but didn't change any conclusion.
- Limitations: small per-task eval sets (300 val images, one reference image for SIFT), binary
  (pet/background) segmentation, and only classical restorers by default — DL restorers (DnCNN,
  Restormer, FBCNN) are an obvious next step.
