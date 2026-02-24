#!/usr/bin/env python3
"""Export trained Keras weights to a frozen TensorFlow .pb graph.

The exported .pb is compatible with DeepSORT's generate_detections.py
(after the one-line fix in that file — see README).

Node names in the exported graph
---------------------------------
  Input  : "images"   — raw BGR float32 in [0, 255], shape (?, H, W, 3)
  Output : "features" — L2-normalised embedding,     shape (?, 128)

Usage
-----
python scripts/export_pb.py \\
    --weights runs/turkey_reid/best_model_keras.h5 \\
    --output  runs/turkey_reid/best_model.pb

# Custom image size or embedding dim
python scripts/export_pb.py \\
    --weights runs/turkey_reid/best_model_keras.h5 \\
    --output  runs/turkey_reid/best_model.pb \\
    --image_h 128 --image_w 64 --embedding_dim 128
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.model import build_embedding_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Keras ReID model to frozen TF .pb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",       required=True, help=".h5 weights file")
    p.add_argument("--output",        required=True, help="Destination .pb path")
    p.add_argument("--image_h",       type=int,   default=128)
    p.add_argument("--image_w",       type=int,   default=64)
    p.add_argument("--embedding_dim", type=int,   default=128)
    p.add_argument("--num_filters",   type=int,   default=32)
    p.add_argument("--dropout",       type=float, default=0.30,
                   help="Dropout rate (set to 0 for inference-only export).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(args: argparse.Namespace) -> None:
    input_shape = (args.image_h, args.image_w, 3)

    model = build_embedding_model(
        input_shape=input_shape,
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        dropout_rate=args.dropout,
    )
    model.load_weights(args.weights)
    print("[export] Weights loaded from: {}".format(args.weights))
    print("[export] Input shape: {}".format(input_shape))
    print("[export] Embedding dim: {}".format(args.embedding_dim))

    # ---- Wrap model in a concrete function with named I/O ----
    @tf.function(input_signature=[
        tf.TensorSpec([None] + list(input_shape), tf.float32, name="images")
    ])
    def _serving(x: tf.Tensor) -> tf.Tensor:
        emb = model(x, training=False)
        # Explicit tf.identity ensures the output op is named "features"
        return tf.identity(emb, name="features")

    concrete = _serving.get_concrete_function()
    frozen   = convert_variables_to_constants_v2(concrete)

    # ---- Write to disk ----
    out_dir  = os.path.dirname(os.path.abspath(args.output))
    out_name = os.path.basename(args.output)
    os.makedirs(out_dir, exist_ok=True)

    tf.io.write_graph(
        graph_or_graph_def=frozen.graph,
        logdir=out_dir,
        name=out_name,
        as_text=False,
    )
    print("[export] .pb saved -> {}".format(args.output))

    # ---- Verify node names ----
    node_names = {op.name for op in frozen.graph.get_operations()}
    ok_in  = "images"   in node_names
    ok_out = "features" in node_names

    print("\n[export] Node name verification:")
    print("  'images'   present : {}".format(ok_in))
    print("  'features' present : {}".format(ok_out))

    if not (ok_in and ok_out):
        print("\n[export] WARNING — expected nodes not found.")
        print("[export] Nodes containing 'image' or 'feat':")
        for n in sorted(node_names):
            if "image" in n.lower() or "feat" in n.lower():
                print("    {}".format(n))
        print(
            "\n[export] Tip: use one of the names above as input_name / "
            "output_name when calling create_box_encoder()."
        )

    # ---- Quick numeric smoke-test ----
    print("\n[export] Running smoke-test ...")
    dummy = np.random.uniform(0, 255, (2,) + input_shape).astype(np.float32)
    with tf.Graph().as_default() as g:
        tf.import_graph_def(frozen.graph.as_graph_def(), name="")
    with tf.compat.v1.Session(graph=g) as sess:
        in_t  = g.get_tensor_by_name("images:0")
        out_t = g.get_tensor_by_name("features:0")
        out   = sess.run(out_t, {in_t: dummy})
    norms = np.linalg.norm(out, axis=1)
    print("[export] Smoke-test output shape: {}".format(out.shape))
    print("[export] Embedding norms: {} (expect ~1.0)".format(
        norms.round(5)
    ))
    assert abs(norms[0] - 1.0) < 1e-4, "Norm check failed"
    print("[export] Smoke-test PASSED.")


if __name__ == "__main__":
    export(parse_args())
