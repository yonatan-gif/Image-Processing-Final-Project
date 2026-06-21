"""Week-4 EDA: dataset stats + an annotated sample grid for the README.

Oxford-IIIT Pet ships three kinds of ground truth, and we visualize all of them:
  - breed label   (classification target)  -> panel title
  - bounding box  (annotations/xmls/*.xml)  -> green rectangle
  - trimap mask   (segmentation target)     -> red pet overlay

Run:  python scripts/eda.py
Outputs: assets/eda_samples.png, assets/eda_class_distribution.png  (tracked, for README)
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pets import load_pets_classification, load_pets_segmentation, trimap_to_binary  # noqa: E402

ASSETS = ROOT / "assets"
DATA = ROOT / "data"
XML_DIR = DATA / "oxford-iiit-pet" / "annotations" / "xmls"


def load_bbox(stem: str):
    """Return [xmin, ymin, xmax, ymax] from the VOC xml, or None if absent."""
    xml = XML_DIR / f"{stem}.xml"
    if not xml.exists():
        return None
    obj = ET.parse(xml).getroot().find("object")
    if obj is None:
        return None
    b = obj.find("bndbox")
    return [int(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]


def overlay_mask(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Tint the pet pixels (mask==1) red over the original image."""
    out = img.astype(np.float32).copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    out[mask == 1] = (1 - alpha) * out[mask == 1] + alpha * red
    return out.astype(np.uint8)


def sample_grid(cls_ds, seg_ds, n: int = 8, seed: int = 42):
    """Grid of `n` images, each with breed title + bbox + mask overlay."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(cls_ds), size=n, replace=False)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax, i in zip(axes.ravel(), idxs):
        pil, label = cls_ds[int(i)]
        _, trimap = seg_ds[int(i)]
        img = np.asarray(pil.convert("RGB"))
        mask = trimap_to_binary(trimap)
        ax.imshow(overlay_mask(img, mask))
        bbox = load_bbox(cls_ds._images[int(i)].stem)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor="lime", linewidth=2))
        ax.set_title(cls_ds.classes[label], fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Oxford-IIIT Pet — breed label + bbox (green) + pet mask (red)", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSETS / "eda_samples.png", dpi=110, bbox_inches="tight")
    print(f"  saved {ASSETS / 'eda_samples.png'}")


def class_distribution(cls_ds):
    counts = Counter(cls_ds._labels)
    names = cls_ds.classes
    order = sorted(range(len(names)), key=lambda c: names[c])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(names)), [counts[c] for c in order], color="#2b6cb0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([names[c] for c in order], rotation=90, fontsize=7)
    ax.set_ylabel("# images (trainval)")
    ax.set_title(f"Class distribution — {len(names)} breeds, {len(cls_ds)} images")
    fig.tight_layout()
    fig.savefig(ASSETS / "eda_class_distribution.png", dpi=110, bbox_inches="tight")
    print(f"  saved {ASSETS / 'eda_class_distribution.png'}")
    return counts


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    cls_ds = load_pets_classification(root=str(DATA), download=False)
    seg_ds = load_pets_segmentation(root=str(DATA), download=False)

    # Image-size stats over a sample (full scan is slow and unnecessary).
    rng = np.random.default_rng(0)
    sizes = np.array([Image.open(cls_ds._images[int(i)]).size
                      for i in rng.choice(len(cls_ds), 300, replace=False)])  # (w, h)

    print("=== Oxford-IIIT Pet (trainval) ===")
    print(f"images:        {len(cls_ds)}")
    print(f"breeds:        {len(cls_ds.classes)}")
    counts = class_distribution(cls_ds)
    print(f"per-class:     min={min(counts.values())}  max={max(counts.values())}  "
          f"mean={np.mean(list(counts.values())):.1f}")
    print(f"image width:   min={sizes[:,0].min()}  max={sizes[:,0].max()}  median={int(np.median(sizes[:,0]))}")
    print(f"image height:  min={sizes[:,1].min()}  max={sizes[:,1].max()}  median={int(np.median(sizes[:,1]))}")
    sample_grid(cls_ds, seg_ds)
    print("\nEDA done.")


if __name__ == "__main__":
    main()
