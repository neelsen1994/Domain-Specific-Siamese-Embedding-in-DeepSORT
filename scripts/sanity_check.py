#!/usr/bin/env python3
"""Sanity-check: load an exported .pb and run inference on 1–3 images.

Verifies:
  * The .pb loads without errors.
  * Input/output tensor names are correct.
  * Embeddings are L2-normalised (norm ≈ 1.0).
  * Same-ID cosine similarity > different-ID cosine similarity (if images 2+3 provided).

Usage
-----
# Minimal (1 image)
python scripts/sanity_check.py \\
    --pb_path runs/turkey_reid/best_model.pb \\
    --image1  dataset_siam_21/object_1/frame_100.jpg

# Full triplet check
python scripts/sanity_check.py \\
    --pb_path runs/turkey_reid/best_model.pb \\
    --image1  dataset_siam_21/object_1/frame_100.jpg \\
    --image2  dataset_siam_21/object_1/frame_200.jpg \\
    --image3  dataset_siam_21/object_2/frame_100.jpg

# If your graph uses different node names
python scripts/sanity_check.py \\
    --pb_path runs/turkey_reid/best_model.pb \\
    --image1  ... \\
    --input_name images --output_name features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sanity-check the exported .pb model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pb_path",     required=True, help="Frozen .pb path")
    p.add_argument("--image1",      required=True, help="Query image")
    p.add_argument("--image2",      default=None,
                   help="Second image — SAME identity as image1 (expect HIGH sim)")
    p.add_argument("--image3",      default=None,
                   help="Third image — DIFFERENT identity (expect LOW sim)")
    p.add_argument("--image_h",     type=int, default=128)
    p.add_argument("--image_w",     type=int, default=64)
    p.add_argument("--input_name",  default="images",
                   help="Name of the input tensor node in the graph.")
    p.add_argument("--output_name", default="features",
                   help="Name of the output tensor node in the graph.")
    p.add_argument("--list_nodes",  action="store_true",
                   help="Print all graph node names (useful for debugging).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pb(pb_path: str):
    """Load a frozen .pb into a TF1-compat session (no node prefix)."""
    with tf.io.gfile.GFile(pb_path, "rb") as fh:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(fh.read())

    graph = tf.Graph()
    with graph.as_default():
        # name="" preserves original node names (images:0, features:0)
        tf.import_graph_def(graph_def, name="")

    session = tf.compat.v1.Session(graph=graph)
    return session, graph


def preprocess(path: str, H: int, W: int) -> np.ndarray:
    """Load BGR image, resize, return float32 [0, 255] with batch dim."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("Cannot read image: {}".format(path))
    img = cv2.resize(img, (W, H)).astype(np.float32)
    return img[np.newaxis]   # (1, H, W, 3)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("[sanity] Loading .pb: {}".format(args.pb_path))
    session, graph = load_pb(args.pb_path)

    # ---- List nodes if requested ----
    if args.list_nodes:
        print("\n[sanity] All graph nodes:")
        for op in graph.get_operations():
            print("  {} ({})".format(op.name, op.type))
        print()

    # ---- Get input / output tensors ----
    try:
        in_t  = graph.get_tensor_by_name("{}:0".format(args.input_name))
    except KeyError:
        print("[sanity] ERROR: input tensor '{}:0' not found in graph.".format(
            args.input_name))
        print("[sanity] Run with --list_nodes to see available nodes.")
        session.close()
        sys.exit(1)

    try:
        out_t = graph.get_tensor_by_name("{}:0".format(args.output_name))
    except KeyError:
        print("[sanity] ERROR: output tensor '{}:0' not found in graph.".format(
            args.output_name))
        print("[sanity] Run with --list_nodes to see available nodes.")
        session.close()
        sys.exit(1)

    print("[sanity] Input  tensor : {}  shape={}".format(
        in_t.name, in_t.get_shape().as_list()))
    print("[sanity] Output tensor : {}  shape={}".format(
        out_t.name, out_t.get_shape().as_list()))

    H, W = args.image_h, args.image_w

    # ---- Image 1 ----
    img1  = preprocess(args.image1, H, W)
    feat1 = session.run(out_t, {in_t: img1})[0]
    norm1 = np.linalg.norm(feat1)
    print("\n[sanity] Image 1: {}".format(args.image1))
    print("  Embedding norm   : {:.6f}  (expect ≈ 1.0)".format(norm1))
    print("  First 8 dims     : {}".format(feat1[:8].round(4)))

    norm_ok = abs(norm1 - 1.0) < 1e-3
    if norm_ok:
        print("  Norm check       : PASS")
    else:
        print("  Norm check       : FAIL — model may not be L2-normalising!")

    # ---- Image 2 (same ID) ----
    sim12: Optional[float] = None
    if args.image2:
        img2  = preprocess(args.image2, H, W)
        feat2 = session.run(out_t, {in_t: img2})[0]
        sim12 = cosine_sim(feat1, feat2)
        print("\n[sanity] Image 2 (same ID): {}".format(args.image2))
        print("  Cosine sim (1 vs 2) : {:.4f}  (expect HIGH, close to 1)".format(sim12))

    # ---- Image 3 (different ID) ----
    sim13: Optional[float] = None
    if args.image3:
        img3  = preprocess(args.image3, H, W)
        feat3 = session.run(out_t, {in_t: img3})[0]
        sim13 = cosine_sim(feat1, feat3)
        print("\n[sanity] Image 3 (diff ID): {}".format(args.image3))
        print("  Cosine sim (1 vs 3) : {:.4f}  (expect LOW, close to 0)".format(sim13))

    # ---- Overall verdict ----
    print("\n" + "-" * 40)
    if sim12 is not None and sim13 is not None:
        if sim12 > sim13:
            print("PASS: same-ID sim ({:.4f}) > diff-ID sim ({:.4f})".format(
                sim12, sim13))
        else:
            print("FAIL: same-ID sim ({:.4f}) <= diff-ID sim ({:.4f}) — "
                  "model not yet discriminative or needs more training.".format(
                      sim12, sim13))
    elif norm_ok:
        print("PASS: embedding norm is ≈ 1.0")
    else:
        print("PARTIAL: could not run full triplet check (provide --image2 / --image3)")

    session.close()


if __name__ == "__main__":
    main()
