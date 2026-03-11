#!/usr/bin/env python3
"""Turkey ReID Evaluation Script for Frozen .pb Models
========================================================
Evaluates a frozen TensorFlow graph (.pb model) on CMC and mAP metrics.
Works with DeepSORT-style frozen graphs with "images" input and "features" output.

Usage
-----
# Evaluate best_model.pb
python scripts/eval_reid_pb.py \\
    --model_pb output/best_model.pb \\
    --manifest runs/turkey_reid/split_manifest.json

# Evaluate mars-small128.pb
python scripts/eval_reid_pb.py \\
    --model_pb model_feature_extractor/mars-small128.pb \\
    --manifest runs/turkey_reid/split_manifest.json \\
    --show_distances
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dataset import load_split_from_manifest, load_split_as_arrays
from scripts.metrics import compute_cmc_map, mean_intra_inter_distance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate frozen .pb ReID model (CMC + mAP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_pb", required=True,
                   help="Path to frozen TensorFlow .pb model.")
    p.add_argument("--manifest", required=True,
                   help="Path to split manifest JSON (from train_reid.py).")

    p.add_argument("--query_split",   default="test",
                   choices=["train", "val", "test"],
                   help="Split to use as query set.")
    p.add_argument("--gallery_splits", nargs="+",
                   choices=["train", "val", "test"],
                   default=None,
                   help="Splits to build the gallery from.  If omitted, "
                        "uses the same split as --query_split (leave-one-out).")

    p.add_argument("--image_h",       type=int,   default=128)
    p.add_argument("--image_w",       type=int,   default=64)
    p.add_argument("--input_name",    default="images",
                   help="Input tensor name in the frozen graph.")
    p.add_argument("--output_name",   default="features",
                   help="Output tensor name in the frozen graph.")
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--ranks",         type=int,   nargs="+", default=[1, 5, 10])
    p.add_argument("--show_distances", action="store_true",
                   help="Print mean intra/inter cosine distance (slower).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Frozen .pb Model Loading
# ---------------------------------------------------------------------------

def load_frozen_pb_model(
    model_path: str,
    input_name: str = "images",
    output_name: str = "features",
) -> Tuple[tf.compat.v1.Session, tf.Tensor, tf.Tensor]:
    """Load a frozen TensorFlow graph and return session + input/output tensors.

    Parameters
    ----------
    model_path : str
        Path to the .pb file
    input_name : str
        Name of the input tensor (without :0)
    output_name : str
        Name of the output tensor (without :0)

    Returns
    -------
    session : tf.compat.v1.Session
        TensorFlow session for inference
    input_tensor : tf.Tensor
        Input placeholder tensor
    output_tensor : tf.Tensor
        Output tensor (embeddings)
    """
    graph = tf.Graph()
    with graph.as_default():
        with tf.compat.v1.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        input_tensor = graph.get_tensor_by_name(f"{input_name}:0")
        output_tensor = graph.get_tensor_by_name(f"{output_name}:0")

    session = tf.compat.v1.Session(graph=graph)
    return session, input_tensor, output_tensor


def extract_embeddings_pb(
    session: tf.compat.v1.Session,
    input_tensor: tf.Tensor,
    output_tensor: tf.Tensor,
    images: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract embeddings using a frozen .pb model."""
    embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i: i + batch_size]
        feed_dict = {input_tensor: batch}
        batch_embs = session.run(output_tensor, feed_dict=feed_dict)
        embs.append(batch_embs)
    return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def merge_splits(
    splits: List[Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    """Merge multiple split dicts into one (union of images per ID)."""
    merged: Dict[str, List[str]] = {}
    for s in splits:
        for cls, paths in s.items():
            merged.setdefault(cls, [])
            for p in paths:
                if p not in merged[cls]:
                    merged[cls].append(p)
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Load split manifest ----
    train_data, val_data, test_data = load_split_from_manifest(args.manifest)
    all_splits = {"train": train_data, "val": val_data, "test": test_data}

    image_size = (args.image_h, args.image_w)

    # Query split
    q_data = all_splits[args.query_split]

    # Gallery splits
    leave_one_out = False
    if args.gallery_splits is None:
        # Same split → leave-one-out evaluation
        g_data = q_data
        leave_one_out = True
    else:
        g_data = merge_splits([all_splits[s] for s in args.gallery_splits])

    # ---- Load images ----
    q_imgs, q_lbls, q_names = load_split_as_arrays(q_data, image_size)
    g_imgs, g_lbls, g_names = load_split_as_arrays(g_data, image_size)

    print("\n[eval] Query   split '{}': {:d} images, {:d} IDs".format(
        args.query_split, len(q_imgs), len(set(q_lbls))
    ))
    gal_tag = args.gallery_splits or [args.query_split]
    print("[eval] Gallery split(s) {}: {:d} images, {:d} IDs".format(
        gal_tag, len(g_imgs), len(set(g_lbls))
    ))

    # ---- Load frozen .pb model ----
    print("[eval] Loading model from: {}".format(args.model_pb))
    session, input_tensor, output_tensor = load_frozen_pb_model(
        args.model_pb,
        input_name=args.input_name,
        output_name=args.output_name,
    )
    print("[eval] Model loaded successfully")

    # ---- Extract embeddings ----
    print("[eval] Extracting query embeddings ...")
    q_embs = extract_embeddings_pb(session, input_tensor, output_tensor, q_imgs, args.batch_size)

    print("[eval] Extracting gallery embeddings ...")
    if leave_one_out:
        g_embs = q_embs
    else:
        g_embs = extract_embeddings_pb(session, input_tensor, output_tensor, g_imgs, args.batch_size)

    # Verify norms
    norms = np.linalg.norm(q_embs, axis=1)
    print("[eval] Embedding norms: mean={:.4f} std={:.4f} (expect ~1.0)".format(
        norms.mean(), norms.std()
    ))

    # ---- Compute CMC + mAP ----
    ranks = tuple(args.ranks)
    cmc, mAP = compute_cmc_map(
        q_embs, q_lbls,
        g_embs, g_lbls,
        ranks=ranks,
        remove_self=leave_one_out,
    )

    # ---- Print results ----
    print("\n" + "=" * 52)
    print("  ReID Evaluation  (query: {})".format(args.query_split))
    print("=" * 52)
    for k in sorted(ranks):
        key = "rank{}".format(k)
        print("  Rank-{:2d} :  {:.4f}  ({:.2f}%)".format(
            k, cmc.get(key, 0.0), cmc.get(key, 0.0) * 100.0
        ))
    print("  mAP    :  {:.4f}  ({:.2f}%)".format(mAP, mAP * 100.0))
    print("=" * 52)

    # ---- Optional: distance separation ----
    if args.show_distances:
        print("\n[eval] Computing intra/inter distance separation ...")
        # Use all available embeddings for a richer estimate
        all_embs = g_embs
        all_lbls = g_lbls
        if not leave_one_out:
            all_embs = np.concatenate([q_embs, g_embs], axis=0)
            all_lbls = np.concatenate([q_lbls, g_lbls], axis=0)
        d_intra, d_inter = mean_intra_inter_distance(all_embs, all_lbls)
        print("  Mean intra-class cosine dist : {:.4f}".format(d_intra))
        print("  Mean inter-class cosine dist : {:.4f}".format(d_inter))
        print("  Separability (inter - intra) : {:.4f}".format(d_inter - d_intra))

    session.close()


if __name__ == "__main__":
    main()
