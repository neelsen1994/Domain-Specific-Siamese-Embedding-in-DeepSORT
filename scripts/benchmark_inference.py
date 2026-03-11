#!/usr/bin/env python3
"""Benchmark inference time for two .pb models.

Measures per-image latency for:
  - Model v5  : runs/turkey_reid_v5/best_model.pb
  - mars-small128 : model_feature_extractor/mars-small128.pb

Usage
-----
# Default paths (both models)
conda run -n mot python scripts/benchmark_inference.py

# Custom paths
conda run -n mot python scripts/benchmark_inference.py \
    --model_v5   runs/turkey_reid_v5/best_model.pb \
    --model_mars model_feature_extractor/mars-small128.pb \
    --n_warmup 20 --n_runs 200 --batch_size 1
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark inference time for v5 vs mars-small128",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_v5",
                   default="runs/turkey_reid_v5/best_model.pb",
                   help="Path to model v5 frozen .pb")
    p.add_argument("--model_mars",
                   default="model_feature_extractor/mars-small128.pb",
                   help="Path to mars-small128 frozen .pb")
    p.add_argument("--input_name",  default="images",
                   help="Input tensor name (without :0)")
    p.add_argument("--output_name", default="features",
                   help="Output tensor name (without :0)")
    p.add_argument("--image_h",    type=int, default=128)
    p.add_argument("--image_w",    type=int, default=64)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Number of images per inference call")
    p.add_argument("--n_warmup",   type=int, default=20,
                   help="Warm-up runs (excluded from timing)")
    p.add_argument("--n_runs",     type=int, default=200,
                   help="Timed runs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pb(pb_path: str):
    """Load a frozen .pb into a TF1-compat session."""
    with tf.io.gfile.GFile(pb_path, "rb") as fh:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(fh.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    session = tf.compat.v1.Session(graph=graph)
    return session, graph


def benchmark(session, in_t, out_t, dummy_input: np.ndarray,
              n_warmup: int, n_runs: int) -> dict:
    """Run warm-up then timed inference. Returns timing stats (ms)."""
    # Warm-up
    for _ in range(n_warmup):
        session.run(out_t, {in_t: dummy_input})

    # Timed runs
    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(out_t, {in_t: dummy_input})
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms)
    return {
        "mean_ms":   float(np.mean(times_ms)),
        "std_ms":    float(np.std(times_ms)),
        "min_ms":    float(np.min(times_ms)),
        "max_ms":    float(np.max(times_ms)),
        "p50_ms":    float(np.percentile(times_ms, 50)),
        "p95_ms":    float(np.percentile(times_ms, 95)),
        "p99_ms":    float(np.percentile(times_ms, 99)),
    }


def print_results(name: str, stats: dict, batch_size: int) -> None:
    per_img_ms = stats["mean_ms"] / batch_size
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Batch size          : {batch_size}")
    print(f"  Mean  (batch)       : {stats['mean_ms']:.3f} ms")
    print(f"  Std   (batch)       : {stats['std_ms']:.3f} ms")
    print(f"  Min   (batch)       : {stats['min_ms']:.3f} ms")
    print(f"  Max   (batch)       : {stats['max_ms']:.3f} ms")
    print(f"  p50   (batch)       : {stats['p50_ms']:.3f} ms")
    print(f"  p95   (batch)       : {stats['p95_ms']:.3f} ms")
    print(f"  p99   (batch)       : {stats['p99_ms']:.3f} ms")
    print(f"  Mean  per image     : {per_img_ms:.3f} ms  "
          f"({1000.0/per_img_ms:.1f} img/s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    H, W, C = args.image_h, args.image_w, 3
    dummy = np.random.randint(0, 256, (args.batch_size, H, W, C),
                              dtype=np.uint8).astype(np.float32)

    models = [
        ("Model v5 (turkey reid)", args.model_v5),
        ("mars-small128 (baseline)", args.model_mars),
    ]

    all_stats = {}

    for name, pb_path in models:
        print(f"\n[bench] Loading: {pb_path}")
        try:
            session, graph = load_pb(pb_path)
        except Exception as e:
            print(f"[bench] SKIP — could not load {pb_path}: {e}")
            continue

        try:
            in_t  = graph.get_tensor_by_name(f"{args.input_name}:0")
            out_t = graph.get_tensor_by_name(f"{args.output_name}:0")
        except KeyError as e:
            print(f"[bench] SKIP — tensor not found: {e}")
            session.close()
            continue

        print(f"[bench] Input  : {in_t.name}  shape={in_t.get_shape().as_list()}")
        print(f"[bench] Output : {out_t.name}  shape={out_t.get_shape().as_list()}")
        print(f"[bench] Warm-up ({args.n_warmup} runs) ...")
        stats = benchmark(session, in_t, out_t, dummy, args.n_warmup, args.n_runs)
        print_results(name, stats, args.batch_size)
        all_stats[name] = stats
        session.close()

    # --- Comparison summary ---
    if len(all_stats) == 2:
        names = list(all_stats.keys())
        m0 = all_stats[names[0]]["mean_ms"]
        m1 = all_stats[names[1]]["mean_ms"]
        ratio = m1 / m0 if m0 > 0 else float("inf")
        print(f"\n{'='*55}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*55}")
        print(f"  {names[0]:<35}: {m0:.3f} ms/batch")
        print(f"  {names[1]:<35}: {m1:.3f} ms/batch")
        if ratio >= 1.0:
            print(f"  => {names[0]} is {ratio:.2f}x faster than {names[1]}")
        else:
            print(f"  => {names[1]} is {1/ratio:.2f}x faster than {names[0]}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
