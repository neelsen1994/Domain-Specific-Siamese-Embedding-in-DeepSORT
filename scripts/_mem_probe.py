#!/usr/bin/env python3
"""Internal helper: measure memory footprint of a single .pb model.

Called by benchmark_memory.py as a subprocess.  Prints one JSON line to stdout.

Usage (internal, not called directly):
    python scripts/_mem_probe.py <pb_path> <input_name> <output_name> \
        <image_h> <image_w> <batch_size> <n_warmup> <gpu_id>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import psutil
import tensorflow as tf

GPU_DEV = "GPU:0"  # TF device string for memory queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_mib() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _tf_gpu_mib(stat: str = "current") -> float:
    """Return TF allocator GPU memory (MiB) — 'current' or 'peak'."""
    try:
        info = tf.config.experimental.get_memory_info(GPU_DEV)
        return info[stat] / (1024 ** 2)
    except Exception:
        return 0.0


def _nvidia_smi_proc_mib(pid: int, gpu_id: int) -> float:
    """Per-process GPU memory from nvidia-smi (MiB).  Fallback only."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits",
             "-i", str(gpu_id)],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] == str(pid):
                return float(parts[1])
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pb_path     = sys.argv[1]
    input_name  = sys.argv[2]
    output_name = sys.argv[3]
    H           = int(sys.argv[4])
    W           = int(sys.argv[5])
    batch_size  = int(sys.argv[6])
    n_warmup    = int(sys.argv[7])
    gpu_id      = int(sys.argv[8])
    pid         = os.getpid()

    # Enable memory growth so TF allocates incrementally (not full GPU upfront)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # ---- 1. Baseline (after TF import, before model load) ----
    rss_baseline     = _rss_mib()
    tf_gpu_baseline  = _tf_gpu_mib("current")

    # ---- 2. Load frozen graph ----
    with tf.io.gfile.GFile(pb_path, "rb") as fh:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(fh.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    session = tf.compat.v1.Session(graph=graph)

    rss_after_load    = _rss_mib()
    tf_gpu_after_load = _tf_gpu_mib("current")

    # ---- 3. Warm-up inference (forces kernel JIT + buffer allocation) ----
    in_t  = graph.get_tensor_by_name(f"{input_name}:0")
    out_t = graph.get_tensor_by_name(f"{output_name}:0")
    dummy = np.random.randint(0, 256, (batch_size, H, W, 3),
                              dtype=np.uint8).astype(np.float32)

    # Reset peak stats so we measure only the inference peak, not load peak
    try:
        tf.config.experimental.reset_memory_stats(GPU_DEV)
    except Exception:
        pass

    for _ in range(n_warmup):
        session.run(out_t, {in_t: dummy})

    rss_after_warmup     = _rss_mib()
    tf_gpu_current       = _tf_gpu_mib("current")
    tf_gpu_peak          = _tf_gpu_mib("peak")    # peak VRAM during inference

    # nvidia-smi: process-level, needs brief sleep for smi to register the process
    time.sleep(0.5)
    smi_mib = _nvidia_smi_proc_mib(pid, gpu_id)

    # ---- 4. Count graph Const params (weights only) ----
    total_params = 0
    try:
        with graph.as_default():
            for op in graph.get_operations():
                if op.type == "Const":
                    t = graph.get_tensor_by_name(op.name + ":0")
                    try:
                        val = session.run(t)
                        total_params += val.size
                    except Exception:
                        pass
    except Exception:
        total_params = -1

    session.close()

    result = {
        "pb_path":              pb_path,
        "pid":                  pid,
        "batch_size":           batch_size,
        "pb_size_mib":          round(os.path.getsize(pb_path) / (1024 ** 2), 2),
        "total_const_params":   total_params,
        # CPU / RAM
        "rss_baseline_mib":          round(rss_baseline, 1),
        "rss_after_load_mib":        round(rss_after_load, 1),
        "rss_after_warmup_mib":      round(rss_after_warmup, 1),
        "rss_load_delta_mib":        round(rss_after_load   - rss_baseline, 1),
        "rss_inference_delta_mib":   round(rss_after_warmup - rss_after_load, 1),
        "rss_total_delta_mib":       round(rss_after_warmup - rss_baseline, 1),
        # GPU (TF allocator — most accurate)
        "tf_gpu_baseline_mib":       round(tf_gpu_baseline, 1),
        "tf_gpu_after_load_mib":     round(tf_gpu_after_load, 1),
        "tf_gpu_current_mib":        round(tf_gpu_current, 1),
        "tf_gpu_peak_inference_mib": round(tf_gpu_peak, 1),
        "tf_gpu_load_delta_mib":     round(tf_gpu_after_load - tf_gpu_baseline, 1),
        "tf_gpu_inference_peak_delta_mib": round(tf_gpu_peak - tf_gpu_after_load, 1),
        # GPU (nvidia-smi per-process, fallback)
        "smi_gpu_mib":               round(smi_mib, 1),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
