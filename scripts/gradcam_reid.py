#!/usr/bin/env python3
"""Feature-map Visualization for Turkey ReID
=============================================
Two visualization modes depending on the model source:

  Keras .h5  →  Grad-CAM (Selvaraju et al., 2017)
               Gradient of the ID classification head score w.r.t. the last
               stage-4 conv feature map.  Shows which spatial regions most
               activate the predicted turkey identity.

  Frozen .pb →  Occlusion Sensitivity
               A gray patch is slid across the image; each position is scored
               by the drop in cosine similarity between the occluded and
               original embedding.  Works without a classification head and is
               the natural visualization for a pure metric-learning model.

Both modes produce the same 3-panel figure layout:
  [Original]  |  [Overlay]  |  [Sensitivity / CAM heatmap]

Output: a grid PDF/PNG suitable for direct inclusion in a paper.

Usage
-----
# Grad-CAM — v6 (default)
conda run -n mot python scripts/gradcam_reid.py \\
    --weights  runs/turkey_reid_v6/best_model_keras.h5 \\
    --manifest runs/turkey_reid_v6/split_manifest.json

# Grad-CAM — v5
conda run -n mot python scripts/gradcam_reid.py \\
    --arch v5 --dropout 0.3 \\
    --weights  runs/turkey_reid_v5/best_model_keras.h5 \\
    --manifest runs/turkey_reid_v6/split_manifest.json

# Occlusion sensitivity — mars-small128 baseline
conda run -n mot python scripts/gradcam_reid.py \\
    --pb_model model_feature_extractor/mars-small128.pb \\
    --manifest runs/turkey_reid_v6/split_manifest.json \\
    --label mars --output_dir runs/baseline_mars
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dataset import load_split_from_manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grad-CAM / occlusion sensitivity for turkey ReID models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights",   help="Path to Keras .h5 weights (Grad-CAM mode).")
    g.add_argument("--pb_model",  help="Path to frozen .pb graph (occlusion mode).")

    p.add_argument("--manifest",     required=True, help="Path to split_manifest.json.")

    # Keras-specific
    p.add_argument("--arch",         default="v6",
                   choices=["v1", "v2", "v3", "v4", "v5", "v6"])
    p.add_argument("--target_layer", default=None,
                   help="Conv layer name for CAM (auto-detected if omitted).")
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--num_filters",  type=int, default=32)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--num_strips",   type=int, default=4)
    p.add_argument("--se_reduction", type=int, default=4)

    # .pb-specific
    p.add_argument("--input_name",   default="images",
                   help="Input tensor name in the .pb graph.")
    p.add_argument("--output_name",  default="features",
                   help="Output tensor name in the .pb graph.")
    p.add_argument("--patch_size",   type=int, default=16,
                   help="Occlusion patch size in pixels.")
    p.add_argument("--patch_stride", type=int, default=8,
                   help="Occlusion patch stride in pixels.")

    # Common
    p.add_argument("--label",        default=None,
                   help="Short model label for the figure title / filename.")
    p.add_argument("--split",        default="test",
                   choices=["train", "val", "test"],
                   help="Split to draw sample images from.")
    p.add_argument("--n_ids",        type=int, default=5,
                   help="Number of identities to visualise (randomly sampled).")
    p.add_argument("--k_per_id",     type=int, default=2,
                   help="Number of images per identity.")
    p.add_argument("--output_dir",   default=None,
                   help="Where to save figures.")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--image_h",      type=int, default=128)
    p.add_argument("--image_w",      type=int, default=64)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_DEFAULT_TARGET_LAYER = {
    "v1": "res4_out",      # last spatial ReLU in v1 (post-activation ResBlock output)
    "v2": "stage4_final_relu",
    "v3": "stage4_final_relu",
    "v4": "stage4_final_relu",
    "v5": "stage4_final_relu",
    "v6": "stage4_final_relu",
}


def _build_full_model(args: argparse.Namespace, num_classes: int) -> tf.keras.Model:
    kwargs = dict(
        input_shape=(args.image_h, args.image_w, 3),
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        dropout_rate=args.dropout,
        num_classes=num_classes,
    )
    if args.arch == "v1":
        from scripts.model import build_embedding_model
    elif args.arch == "v2":
        from scripts.model_v2 import build_embedding_model
    elif args.arch == "v3":
        from scripts.model_v3 import build_embedding_model
        kwargs["se_reduction"] = args.se_reduction
    elif args.arch == "v4":
        from scripts.model_v4 import build_embedding_model
        kwargs["num_strips"] = args.num_strips
    elif args.arch == "v5":
        from scripts.model_v5 import build_embedding_model
        kwargs["se_reduction"] = args.se_reduction
    elif args.arch == "v6":
        from scripts.model_v6 import build_embedding_model
        kwargs["num_strips"] = args.num_strips
        kwargs["se_reduction"] = args.se_reduction
    else:
        raise ValueError("Unknown arch: {}".format(args.arch))
    return build_embedding_model(**kwargs)


def _resolve_target_layer(model: tf.keras.Model, arch: str,
                           override: Optional[str]) -> tf.keras.layers.Layer:
    name = override or _DEFAULT_TARGET_LAYER.get(arch)
    if name is None:
        raise ValueError("Cannot auto-detect target layer for arch '{}'.".format(arch))
    try:
        layer = model.get_layer(name)
        print("[gradcam] Target layer: '{}' (output shape: {})".format(
            layer.name, layer.output_shape))
        return layer
    except ValueError:
        # Fall back: find the last Conv2D before the pooling head
        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError(
                "Layer '{}' not found and no Conv2D fallback available.".format(name)
            )
        layer = conv_layers[-1]
        print("[gradcam] Layer '{}' not found — using fallback: '{}' ({})".format(
            name, layer.name, layer.output_shape))
        return layer


# ---------------------------------------------------------------------------
# Frozen .pb loading
# ---------------------------------------------------------------------------

def load_frozen_pb(pb_path: str, input_name: str = "images",
                   output_name: str = "features"):
    graph = tf.Graph()
    with graph.as_default():
        with tf.compat.v1.gfile.GFile(pb_path, "rb") as f:
            gd = tf.compat.v1.GraphDef()
            gd.ParseFromString(f.read())
        tf.import_graph_def(gd, name="")
        inp = graph.get_tensor_by_name("{}:0".format(input_name))
        out = graph.get_tensor_by_name("{}:0".format(output_name))
    sess = tf.compat.v1.Session(graph=graph)
    return sess, inp, out


def _infer_pb(sess, inp_t, out_t, image: np.ndarray) -> np.ndarray:
    return sess.run(out_t, feed_dict={inp_t: image[np.newaxis].astype(np.float32)})[0]


# ---------------------------------------------------------------------------
# Occlusion sensitivity (for .pb models without a classification head)
# ---------------------------------------------------------------------------

def compute_occlusion_sensitivity(
    sess,
    inp_t,
    out_t,
    image: np.ndarray,
    patch_size: int = 16,
    patch_stride: int = 8,
    fill_value: float = 128.0,
) -> np.ndarray:
    """Occlusion sensitivity map via cosine-similarity drop.

    For every patch position, replace that region with a flat fill, re-run
    inference, and record 1 - cosine_similarity(occluded, original).
    High values → the region is important for the embedding.

    Returns
    -------
    cam : (H, W) float32 sensitivity map, values in [0, 1]
    """
    H, W = image.shape[:2]
    original_emb = _infer_pb(sess, inp_t, out_t, image)

    # Grid of patch top-left corners
    ys = list(range(0, H - patch_size + 1, patch_stride))
    xs = list(range(0, W - patch_size + 1, patch_stride))

    score_map = np.zeros((len(ys), len(xs)), dtype=np.float32)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            occluded = image.copy()
            occluded[y: y + patch_size, x: x + patch_size, :] = fill_value
            occ_emb = _infer_pb(sess, inp_t, out_t, occluded)
            # Cosine similarity (both are L2-normalised already)
            sim = float(np.dot(original_emb, occ_emb))
            score_map[i, j] = max(0.0, 1.0 - sim)  # drop in similarity

    # Upsample to original image size
    cam = cv2.resize(score_map, (W, H), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0, None)
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max
    return cam.astype(np.float32)


# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------

def compute_gradcam(
    gradcam_model: tf.keras.Model,
    image: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Compute Grad-CAM heatmap for a single image.

    Parameters
    ----------
    gradcam_model : model with two outputs — [feature_map, logits]
    image         : (H, W, 3) float32 raw BGR [0, 255]

    Returns
    -------
    cam     : (H, W) float32 heatmap, values in [0, 1]
    pred_id : predicted class index
    """
    img_tensor = tf.constant(image[np.newaxis], dtype=tf.float32)

    with tf.GradientTape() as tape:
        feature_map, logits = gradcam_model(img_tensor, training=False)
        pred_id = int(tf.argmax(logits[0]).numpy())
        score = logits[0, pred_id]

    # Gradient of predicted class score w.r.t. last-stage feature map
    grads = tape.gradient(score, feature_map)  # (1, fH, fW, C)

    # Global-average the gradients over spatial dims → per-channel weights
    weights = tf.reduce_mean(grads, axis=[0, 1, 2])  # (C,)

    # Weighted sum of feature map channels
    cam = tf.reduce_sum(feature_map[0] * weights, axis=-1).numpy()  # (fH, fW)

    # ReLU: we only care about positive influence
    cam = np.maximum(cam, 0)

    # Normalize to [0, 1]
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max

    # Upsample to original image size (H x W)
    h, w = image.shape[:2]
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0, 1)

    return cam, pred_id


def cam_to_heatmap(cam: np.ndarray) -> np.ndarray:
    """Convert [0,1] CAM to a JET heatmap (H, W, 3) uint8 RGB."""
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_cam(image_bgr: np.ndarray, cam: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
    """Blend Grad-CAM heatmap onto the original image (returns RGB uint8)."""
    img_rgb = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    heatmap = cam_to_heatmap(cam)
    blended = (img_rgb.astype(np.float32) * (1 - alpha) +
               heatmap.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def render_gradcam_grid(
    samples: List[Tuple[str, np.ndarray, np.ndarray, int, str]],
    out_path: str,
    arch: str,
    method: str = "Grad-CAM",
) -> None:
    """Render a paper-ready grid of [original | overlay | CAM] triplets.

    Parameters
    ----------
    samples  : list of (identity_name, img_bgr, cam, pred_id, pred_name)
    out_path : output file path
    arch     : architecture name string for title
    method   : visualization method label ("Grad-CAM" or "Occlusion Sensitivity")
    """
    n = len(samples)
    fig = plt.figure(figsize=(3 * 3, n * 2.4))
    fig.suptitle(
        "{}  —  Turkey ReID ({})".format(method, arch.upper()),
        fontsize=13, fontweight="bold", y=1.01,
    )

    col_titles = ["Original", "{} Overlay".format(method), "Activation Map"]

    gs = gridspec.GridSpec(n, 3, figure=fig,
                           hspace=0.08, wspace=0.04,
                           left=0.02, right=0.98,
                           top=0.97, bottom=0.02)

    for row, (id_name, img_bgr, cam, pred_id, pred_name) in enumerate(samples):
        img_rgb = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        overlay = overlay_cam(img_bgr, cam)
        heatmap = cam_to_heatmap(cam)

        panels = [img_rgb, overlay, heatmap]

        for col, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(panel, aspect="auto")
            ax.axis("off")

            # Column headers on first row only
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, pad=3)

            # Row label: identity name (left column)
            if col == 0:
                correct = id_name == pred_name
                color = "#2ca02c" if correct else "#d62728"
                ax.set_ylabel(
                    "{}".format(id_name),
                    fontsize=7, rotation=0, labelpad=40,
                    ha="right", va="center", color=color,
                )
            # Predicted class annotation (overlay column)
            if col == 1:
                label = "pred: {}".format(pred_name)
                correct = id_name == pred_name
                ax.text(
                    0.5, 0.02, label,
                    transform=ax.transAxes,
                    fontsize=6, ha="center", va="bottom",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="#2ca02c" if correct else "#d62728",
                              alpha=0.80, linewidth=0),
                )

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print("[gradcam] Figure saved -> {}".format(out_path))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Resolve label, output dir, and mode
    if args.pb_model:
        model_label = args.label or Path(args.pb_model).stem
        default_out  = Path(args.pb_model).parent / "gradcam_{}".format(model_label)
        method       = "Occlusion Sensitivity"
    else:
        model_label = args.label or args.arch
        default_out  = Path(args.weights).parent
        method       = "Grad-CAM"

    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, test_data = load_split_from_manifest(args.manifest)
    split_data = {"train": train_data, "val": val_data, "test": test_data}[args.split]

    # ---- Load model ----
    if args.pb_model:
        print("[gradcam] Loading frozen .pb: {}".format(args.pb_model))
        sess, inp_t, out_t = load_frozen_pb(
            args.pb_model, args.input_name, args.output_name
        )
        gradcam_model = None
        train_classes = None
    else:
        num_classes = len(train_data)
        model = _build_full_model(args, num_classes)
        model.load_weights(args.weights)
        print("[gradcam] Weights loaded from: {}".format(args.weights))
        target_layer = _resolve_target_layer(model, args.arch, args.target_layer)
        gradcam_model = tf.keras.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output[1]],
        )
        train_classes = sorted(train_data.keys())
        sess = inp_t = out_t = None

    # ---- Sample images ----
    split_ids  = sorted(split_data.keys())
    chosen_ids = sorted(rng.sample(split_ids, min(args.n_ids, len(split_ids))))

    samples: List[Tuple[str, np.ndarray, np.ndarray, int, str]] = []

    for id_name in chosen_ids:
        paths    = split_data[id_name]
        selected = rng.sample(paths, min(args.k_per_id, len(paths)))
        for path in selected:
            img = cv2.imread(path)
            if img is None:
                print("[gradcam] WARNING: could not read {}".format(path))
                continue
            img = cv2.resize(img, (args.image_w, args.image_h))

            if sess is not None:
                cam = compute_occlusion_sensitivity(
                    sess, inp_t, out_t,
                    img.astype(np.float32),
                    patch_size=args.patch_size,
                    patch_stride=args.patch_stride,
                )
                pred_idx, pred_name = -1, "—"
            else:
                cam, pred_idx = compute_gradcam(gradcam_model, img.astype(np.float32))
                pred_name = (train_classes[pred_idx]
                             if pred_idx < len(train_classes) else "?")

            samples.append((id_name, img, cam, pred_idx, pred_name))
            print("[gradcam] {} — {} (sensitivity range [{:.3f}, {:.3f}])".format(
                Path(path).name, method, cam.min(), cam.max()
            ))

    if sess is not None:
        sess.close()

    if not samples:
        print("[gradcam] ERROR: no samples could be processed.")
        return

    tag = "{}_{}ids_{}k".format(model_label, args.n_ids, args.k_per_id)
    for ext in ("pdf", "png"):
        render_gradcam_grid(
            samples,
            out_path=str(out_dir / "gradcam_{}.{}".format(tag, ext)),
            arch=model_label,
            method=method,
        )


if __name__ == "__main__":
    main()
