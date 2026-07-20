#!/usr/bin/env python3
"""Quantitative Grad-CAM / Body-Mask Overlap Metric
=====================================================
Addresses Reviewer 2, Minor Comment #11: "The Grad-CAM analysis (Sec 5.4)
is purely qualitative/visual. A quantitative attention-overlap measure
(e.g. against a coarse body-segmentation mask) would strengthen the 'keys
on the animal, not background' claim."

This reuses the existing Grad-CAM implementation in scripts/gradcam_reid.py
(same model loading, same compute_gradcam()) and adds:

  1. A coarse foreground (bird) mask per crop via GrabCut, initialized with
     a fixed interior rectangle (turkey crops are tight bounding boxes, so
     the bird occupies most of the frame; the rectangle assumption is the
     "coarse" part the reviewer explicitly said was acceptable).
  2. Overlap metrics between the Grad-CAM heatmap and that mask:
       - IoU between the top-quartile CAM region and the foreground mask
       - mean CAM activation inside the foreground vs. background
       - a "background leakage" ratio (lower is better; supports the
         "background is strongly suppressed" claim quantitatively)

Runs on CPU. Only needs TensorFlow for the forward+backward pass through
the trained model (Grad-CAM proper), same requirement as the existing
gradcam_reid.py script — no additional GPU dependency beyond what you
already need for that figure.

Usage
-----
python scripts/gradcam_body_overlap.py \\
    --weights  runs/turkey_reid_v5/best_model_keras.h5 \\
    --manifest runs/turkey_reid_v5/split_manifest.json \\
    --arch v5 --n_ids 16 --k_per_id 2 \\
    --output_csv runs/turkey_reid_v5/gradcam_overlap.csv
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.dataset import load_split_from_manifest
from scripts.gradcam_reid import (
    _build_full_model,
    _resolve_target_layer,
    compute_gradcam,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantify Grad-CAM overlap with a coarse body mask",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", required=True, help="Keras .h5 weights.")
    p.add_argument("--manifest", required=True, help="split_manifest.json.")
    p.add_argument("--arch", default="v5", choices=["v1", "v2", "v3", "v4", "v5", "v6"])
    p.add_argument("--target_layer", default=None)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--num_filters", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num_strips", type=int, default=4)
    p.add_argument("--se_reduction", type=int, default=4)

    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n_ids", type=int, default=16,
                   help="Number of identities to sample (default: all 16 test IDs).")
    p.add_argument("--k_per_id", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_h", type=int, default=128)
    p.add_argument("--image_w", type=int, default=64)

    p.add_argument("--cam_top_quantile", type=float, default=0.75,
                   help="CAM pixels above this quantile are treated as 'attended' "
                        "for the IoU calculation (0.75 = top 25%% of activation).")
    p.add_argument("--border_margin", type=float, default=0.12,
                   help="Fraction of width/height trimmed from each edge for the "
                        "GrabCut interior-rectangle seed (assumes a roughly "
                        "centered, tight bounding-box crop).")
    p.add_argument("--output_csv", default="gradcam_overlap.csv")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Coarse foreground mask via GrabCut
# ---------------------------------------------------------------------------

def grabcut_foreground_mask(image_bgr: np.ndarray, border_margin: float) -> np.ndarray:
    """Coarse bird/background segmentation.

    Initializes GrabCut with an interior rectangle (the crop is a tight
    bounding box around one turkey, so a centered rectangle covering
    (1 - 2*border_margin) of the frame is a reasonable foreground seed).
    Returns a bool mask, True = foreground (bird).
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    mx = int(w * border_margin)
    my = int(h * border_margin)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    try:
        cv2.grabCut(image_bgr.astype(np.uint8), mask, rect,
                    bgd_model, fgd_model, iterCount=5,
                    mode=cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # Degenerate crop (e.g. near-uniform) — fall back to the rectangle itself.
        fg = np.zeros((h, w), dtype=bool)
        fg[my:h - my, mx:w - mx] = True
        return fg

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), True, False)
    if fg.sum() == 0:
        # GrabCut collapsed to nothing — fall back to the seed rectangle.
        fg = np.zeros((h, w), dtype=bool)
        fg[my:h - my, mx:w - mx] = True
    return fg


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------

def overlap_metrics(cam: np.ndarray, fg_mask: np.ndarray, top_quantile: float) -> Dict[str, float]:
    thr = np.quantile(cam, top_quantile)
    attended = cam >= thr

    inter = np.logical_and(attended, fg_mask).sum()
    union = np.logical_or(attended, fg_mask).sum()
    iou = float(inter) / float(union) if union > 0 else 0.0

    fg_mean = float(cam[fg_mask].mean()) if fg_mask.any() else 0.0
    bg_mask = ~fg_mask
    bg_mean = float(cam[bg_mask].mean()) if bg_mask.any() else 0.0

    # Fraction of total CAM "mass" that falls on background (lower = better
    # suppression, supports the qualitative claim in Sec 5.4 quantitatively).
    total_mass = cam.sum()
    bg_mass_frac = float(cam[bg_mask].sum() / total_mass) if total_mass > 0 else 0.0

    return dict(
        iou_top_quartile=iou,
        cam_mean_foreground=fg_mean,
        cam_mean_background=bg_mean,
        fg_bg_ratio=(fg_mean / bg_mean) if bg_mean > 1e-8 else float("inf"),
        background_mass_fraction=bg_mass_frac,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    train_data, val_data, test_data = load_split_from_manifest(args.manifest)
    split_data = {"train": train_data, "val": val_data, "test": test_data}[args.split]

    num_classes = len(train_data)
    import tensorflow as tf  # deferred import, keeps CLI --help fast
    model = _build_full_model(args, num_classes)
    model.load_weights(args.weights)
    target_layer = _resolve_target_layer(model, args.arch, args.target_layer)
    gradcam_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output[1]],
    )

    split_ids = sorted(split_data.keys())
    chosen_ids = sorted(rng.sample(split_ids, min(args.n_ids, len(split_ids))))

    rows: List[Dict[str, object]] = []
    for id_name in chosen_ids:
        paths = split_data[id_name]
        selected = rng.sample(paths, min(args.k_per_id, len(paths)))
        for path in selected:
            img = cv2.imread(path)
            if img is None:
                print("[WARN] could not read {}".format(path))
                continue
            img = cv2.resize(img, (args.image_w, args.image_h)).astype(np.float32)

            cam, _ = compute_gradcam(gradcam_model, img)
            fg_mask = grabcut_foreground_mask(img, args.border_margin)
            metrics = overlap_metrics(cam, fg_mask, args.cam_top_quantile)
            metrics["identity"] = id_name
            metrics["path"] = path
            rows.append(metrics)
            print("[{}] {}: IoU={:.3f}  fg/bg CAM ratio={:.2f}  bg_mass={:.3f}".format(
                id_name, Path(path).name, metrics["iou_top_quartile"],
                metrics["fg_bg_ratio"], metrics["background_mass_fraction"],
            ))

    if not rows:
        print("[ERROR] No samples processed.")
        return

    # ---- Aggregate ----
    def agg(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def agg_std(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.std(vals)) if vals else float("nan")

    print("\n" + "=" * 60)
    print("  Grad-CAM / body-mask overlap summary (n={})".format(len(rows)))
    print("=" * 60)
    for key in ["iou_top_quartile", "cam_mean_foreground", "cam_mean_background",
                "fg_bg_ratio", "background_mass_fraction"]:
        print("  {:<28}: {:.4f} +/- {:.4f}".format(key, agg(key), agg_std(key)))
    print("=" * 60)

    import csv
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("[save] Per-image CSV -> {}".format(out_path))


if __name__ == "__main__":
    main()
