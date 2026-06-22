#!/usr/bin/env python3
"""t-SNE Visualization of Turkey ReID Embedding Space
======================================================
Loads a trained model, extracts embeddings from the test split (or any
combination of splits), and produces a publication-quality t-SNE scatter
plot saved as both PDF and PNG.

Supports both Keras .h5 models (--weights / --arch) and frozen TensorFlow
.pb graphs (--pb_model), so the same script can compare fine-tuned models
against the mars-small128 DeepSORT baseline.

Usage
-----
# Default: v6 model, test-split only
conda run -n mot python scripts/visualize_tsne.py \\
    --weights  runs/turkey_reid_v6/best_model_keras.h5 \\
    --manifest runs/turkey_reid_v6/split_manifest.json

# Include all splits (train + val + test)
conda run -n mot python scripts/visualize_tsne.py \\
    --weights  runs/turkey_reid_v6/best_model_keras.h5 \\
    --manifest runs/turkey_reid_v6/split_manifest.json \\
    --splits train val test --output_dir runs/turkey_reid_v6

# v1 baseline (Keras)
conda run -n mot python scripts/visualize_tsne.py \\
    --arch v1 \\
    --weights  runs/turkey_reid/best_model_keras.h5 \\
    --manifest runs/turkey_reid/split_manifest.json

# mars-small128 DeepSORT baseline (.pb frozen graph)
conda run -n mot python scripts/visualize_tsne.py \\
    --pb_model model_feature_extractor/mars-small128.pb \\
    --manifest runs/turkey_reid_v6/split_manifest.json \\
    --label mars --output_dir runs/baseline_mars
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dataset import load_split_from_manifest, load_split_as_arrays


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="t-SNE visualisation of turkey ReID embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- model source (one of the two groups is required) ----
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights",   help="Path to Keras .h5 weights file.")
    g.add_argument("--pb_model",  help="Path to frozen TF .pb graph (e.g. mars-small128.pb).")

    p.add_argument("--manifest",  required=True, help="Path to split_manifest.json.")

    # Keras-specific
    p.add_argument("--arch",       default="v6",
                   choices=["v1", "v2", "v3", "v4", "v5", "v6"],
                   help="Architecture variant (only used with --weights).")
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--num_filters",   type=int, default=32)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--num_strips", type=int,   default=4)
    p.add_argument("--se_reduction", type=int, default=4)

    # .pb-specific
    p.add_argument("--input_name",  default="images",
                   help="Input tensor name in the .pb graph (without :0).")
    p.add_argument("--output_name", default="features",
                   help="Output tensor name in the .pb graph (without :0).")
    p.add_argument("--label",       default=None,
                   help="Short model label for plot title / filename "
                        "(defaults to arch name or pb filename stem).")

    # Common
    p.add_argument("--splits",     nargs="+", default=["test"],
                   choices=["train", "val", "test"],
                   help="Which splits to include in the plot.")
    p.add_argument("--output_dir", default=None,
                   help="Where to save the figure.")
    p.add_argument("--perplexity", type=float, default=15.0,
                   help="t-SNE perplexity (try 10–30 for small datasets).")
    p.add_argument("--n_iter",     type=int,   default=2000,
                   help="t-SNE optimisation iterations.")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--image_h",    type=int,   default=128)
    p.add_argument("--image_w",    type=int,   default=64)
    p.add_argument("--batch_size", type=int,   default=64)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _build_model(args: argparse.Namespace, num_classes: int) -> tf.keras.Model:
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


def _build_feature_model(args: argparse.Namespace, num_classes: int) -> tf.keras.Model:
    """Load full model, then return a features-only model."""
    full = _build_model(args, num_classes)
    full.load_weights(args.weights)
    print("[tsne] Weights loaded from: {}".format(args.weights))

    feat = _build_model(args, None)
    for layer in feat.layers:
        try:
            feat.get_layer(layer.name).set_weights(
                full.get_layer(layer.name).get_weights()
            )
        except Exception:
            pass
    return feat


# ---------------------------------------------------------------------------
# Embedding extraction — Keras
# ---------------------------------------------------------------------------

def extract_embeddings(
    model: tf.keras.Model,
    images: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch_size):
        batch = tf.constant(images[i: i + batch_size], dtype=tf.float32)
        result = model(batch, training=False)
        if isinstance(result, (list, tuple)):
            result = result[0]
        out.append(result.numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Embedding extraction — frozen .pb (TF1 session)
# ---------------------------------------------------------------------------

def load_frozen_pb(
    pb_path: str,
    input_name: str = "images",
    output_name: str = "features",
):
    """Load a frozen TF graph and return (session, input_tensor, output_tensor)."""
    graph = tf.Graph()
    with graph.as_default():
        with tf.compat.v1.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        inp = graph.get_tensor_by_name("{}:0".format(input_name))
        out = graph.get_tensor_by_name("{}:0".format(output_name))
    sess = tf.compat.v1.Session(graph=graph)
    return sess, inp, out


def extract_embeddings_pb(
    sess,
    inp_tensor,
    out_tensor,
    images: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i: i + batch_size]
        results.append(sess.run(out_tensor, feed_dict={inp_tensor: batch}))
    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _split_marker(split: str) -> str:
    return {"train": "o", "val": "s", "test": "^"}.get(split, "o")


def _make_color_map(n: int):
    if n <= 10:
        cmap = plt.cm.get_cmap("tab10", n)
    elif n <= 20:
        cmap = plt.cm.get_cmap("tab20", n)
    else:
        cmap = plt.cm.get_cmap("hsv", n)
    return [cmap(i) for i in range(n)]


def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    split_tags: np.ndarray,
    title: str = "t-SNE of Turkey ReID Embeddings",
    out_path: str = "tsne.pdf",
) -> None:
    unique_ids = sorted(set(labels))
    colors = _make_color_map(len(unique_ids))
    id_to_color = {uid: colors[i] for i, uid in enumerate(unique_ids)}

    fig, ax = plt.subplots(figsize=(9, 7))

    for uid in unique_ids:
        mask = labels == uid
        for split in ["train", "val", "test"]:
            smask = mask & (split_tags == split)
            if not smask.any():
                continue
            ax.scatter(
                coords[smask, 0], coords[smask, 1],
                c=[id_to_color[uid]],
                marker=_split_marker(split),
                s=55, alpha=0.82, linewidths=0.3, edgecolors="white",
                zorder=2,
            )

    # Per-identity legend (colour patches)
    legend_patches = [
        mpatches.Patch(color=id_to_color[uid], label=class_names[uid])
        for uid in unique_ids
    ]
    legend1 = ax.legend(
        handles=legend_patches,
        title="Identity", title_fontsize=8,
        fontsize=7, ncol=2,
        loc="upper right", framealpha=0.85,
        bbox_to_anchor=(1.0, 1.0),
    )
    ax.add_artist(legend1)

    # Split marker legend (only show splits that are present)
    present_splits = [s for s in ["train", "val", "test"]
                      if np.any(split_tags == s)]
    if len(present_splits) > 1:
        from matplotlib.lines import Line2D
        marker_handles = [
            Line2D([0], [0], marker=_split_marker(s), color="gray",
                   linestyle="None", markersize=7, label=s.capitalize())
            for s in present_splits
        ]
        ax.legend(handles=marker_handles, title="Split", title_fontsize=8,
                  fontsize=7, loc="lower right", framealpha=0.85)
        ax.add_artist(legend1)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("[tsne] Figure saved -> {}".format(out_path))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve model label and output directory
    if args.pb_model:
        model_label = args.label or Path(args.pb_model).stem
        default_out  = Path(args.pb_model).parent / "tsne_{}".format(model_label)
    else:
        model_label = args.label or args.arch
        default_out  = Path(args.weights).parent

    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, test_data = load_split_from_manifest(args.manifest)
    all_splits = {"train": train_data, "val": val_data, "test": test_data}

    # Load model once
    if args.pb_model:
        print("[tsne] Loading frozen .pb: {}".format(args.pb_model))
        sess, inp_tensor, out_tensor = load_frozen_pb(
            args.pb_model, args.input_name, args.output_name
        )
        print("[tsne] .pb loaded — input: {}:0  output: {}:0".format(
            args.input_name, args.output_name
        ))
    else:
        num_classes = len(train_data)
        model = _build_feature_model(args, num_classes)
        sess = None

    # Build a consistent global label mapping across all requested splits
    global_classes: List[str] = sorted({
        cls
        for split in args.splits
        for cls in all_splits[split].keys()
    })
    c2i = {c: i for i, c in enumerate(global_classes)}

    all_embs: List[np.ndarray]  = []
    all_labels: List[np.ndarray] = []
    all_split_tags: List[np.ndarray] = []

    for split in args.splits:
        imgs, lbls, cls_names = load_split_as_arrays(
            all_splits[split], (args.image_h, args.image_w)
        )
        local_to_global = np.array([c2i[cn] for cn in cls_names], dtype=np.int32)
        global_lbls = local_to_global[lbls]

        print("[tsne] Extracting embeddings for '{}' split ({} images) ...".format(
            split, len(imgs)
        ))
        if sess is not None:
            embs = extract_embeddings_pb(sess, inp_tensor, out_tensor,
                                         imgs, args.batch_size)
        else:
            embs = extract_embeddings(model, imgs, args.batch_size)

        all_embs.append(embs)
        all_labels.append(global_lbls)
        all_split_tags.append(np.array([split] * len(embs), dtype=object))

    if sess is not None:
        sess.close()

    embs_all   = np.concatenate(all_embs, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    splits_all = np.concatenate(all_split_tags, axis=0)

    norms = np.linalg.norm(embs_all, axis=1)
    print("[tsne] Embedding norms: mean={:.4f} std={:.4f}".format(
        norms.mean(), norms.std()
    ))

    print("[tsne] Running t-SNE on {} embeddings (perplexity={}, n_iter={}) ...".format(
        len(embs_all), args.perplexity, args.n_iter
    ))
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=args.seed,
        learning_rate=200.0,
        init="pca",
        metric="cosine",
    )
    coords = tsne.fit_transform(embs_all)
    print("[tsne] KL divergence: {:.4f}".format(tsne.kl_divergence_))

    split_label = "_".join(args.splits)
    title = "t-SNE  —  Turkey ReID ({}, {} split{})".format(
        model_label.upper(),
        split_label,
        "s" if len(args.splits) > 1 else "",
    )

    for ext in ("pdf", "png"):
        fname = "tsne_{}_{}.{}".format(model_label, split_label, ext)
        plot_tsne(
            coords, labels_all, global_classes, splits_all,
            title=title,
            out_path=str(out_dir / fname),
        )


if __name__ == "__main__":
    main()
