#!/usr/bin/env python3
"""Benchmark runtime memory footprint: model v5 vs mars-small128.

Runs each model in an isolated subprocess so TF memory is not polluted
between measurements.

Usage
-----
conda run -n mot python scripts/benchmark_memory.py

# Custom paths / GPU
conda run -n mot python scripts/benchmark_memory.py \
    --model_v5   runs/turkey_reid_v5/best_model.pb \
    --model_mars model_feature_extractor/mars-small128.pb \
    --gpu_id 0 --batch_size 1 --n_warmup 10
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Memory footprint benchmark: v5 vs mars-small128",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_v5",
                   default="runs/turkey_reid_v5/best_model.pb")
    p.add_argument("--model_mars",
                   default="model_feature_extractor/mars-small128.pb")
    p.add_argument("--input_name",  default="images")
    p.add_argument("--output_name", default="features")
    p.add_argument("--image_h",    type=int, default=128)
    p.add_argument("--image_w",    type=int, default=64)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_warmup",   type=int, default=10,
                   help="Warm-up runs to force kernel compilation & allocation")
    p.add_argument("--gpu_id",     type=int, default=0,
                   help="GPU index for nvidia-smi queries")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Run probe in isolated subprocess
# ---------------------------------------------------------------------------

def probe(pb_path: str, args: argparse.Namespace) -> dict:
    probe_script = Path(__file__).parent / "_mem_probe.py"
    cmd = [
        sys.executable, str(probe_script),
        pb_path,
        args.input_name,
        args.output_name,
        str(args.image_h),
        str(args.image_w),
        str(args.batch_size),
        str(args.n_warmup),
        str(args.gpu_id),
    ]
    print(f"\n[mem] Probing (isolated subprocess): {pb_path}")
    result = subprocess.run(
        cmd, capture_output=True, text=True
    )
    # Find the JSON line in stdout (last non-empty line)
    json_line = None
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            json_line = line
            break

    if json_line is None:
        print("[mem] ERROR: no JSON output from probe.")
        print("  stdout:", result.stdout[-500:] if result.stdout else "(empty)")
        print("  stderr:", result.stderr[-500:] if result.stderr else "(empty)")
        return {}

    return json.loads(json_line)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_stats(label: str, s: dict) -> None:
    if not s:
        print(f"\n  {label}: FAILED")
        return
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  .pb file size              : {s['pb_size_mib']:.2f} MiB")
    print(f"  Graph const params         : {s['total_const_params']:,}")
    print(f"  Batch size                 : {s['batch_size']}")
    print()
    print(f"  --- CPU / RAM (process RSS) ---")
    print(f"  Baseline RSS               : {s['rss_baseline_mib']:.1f} MiB")
    print(f"  After model load           : {s['rss_after_load_mib']:.1f} MiB  "
          f"(+{s['rss_load_delta_mib']:.1f} MiB)")
    print(f"  After warm-up inference    : {s['rss_after_warmup_mib']:.1f} MiB  "
          f"(+{s['rss_inference_delta_mib']:.1f} MiB activation overhead)")
    print(f"  Total RAM increase         : +{s['rss_total_delta_mib']:.1f} MiB")
    print()
    print(f"  --- GPU VRAM (TF allocator) ---")
    if s.get("tf_gpu_peak_inference_mib", 0) > 0:
        print(f"  Baseline VRAM              : {s['tf_gpu_baseline_mib']:.1f} MiB")
        print(f"  After model load           : {s['tf_gpu_after_load_mib']:.1f} MiB  "
              f"(+{s['tf_gpu_load_delta_mib']:.1f} MiB  weights/graph)")
        print(f"  Current (after warmup)     : {s['tf_gpu_current_mib']:.1f} MiB")
        print(f"  Peak during inference      : {s['tf_gpu_peak_inference_mib']:.1f} MiB  "
              f"(+{s['tf_gpu_inference_peak_delta_mib']:.1f} MiB  activations)")
    else:
        print(f"  TF GPU stats unavailable — model may be running on CPU.")
    if s.get("smi_gpu_mib", 0) > 0:
        print(f"  nvidia-smi process total   : {s['smi_gpu_mib']:.1f} MiB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    models = [
        ("Model v5  (turkey ReID, ~714k params)", args.model_v5),
        ("mars-small128  (baseline, ~2.8M params)", args.model_mars),
    ]

    all_stats = {}
    for label, pb_path in models:
        s = probe(pb_path, args)
        all_stats[label] = s
        print_stats(label, s)

    # ---- Comparison ----
    labels = list(all_stats.keys())
    if all(all_stats[l] for l in labels):
        s0, s1 = all_stats[labels[0]], all_stats[labels[1]]
        print(f"\n{'='*60}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*60}")
        rows = [
            ("pb file size (MiB)",
             s0["pb_size_mib"],        s1["pb_size_mib"]),
            ("Graph const params",
             s0["total_const_params"], s1["total_const_params"]),
            ("RAM: load delta (MiB)",
             s0["rss_load_delta_mib"], s1["rss_load_delta_mib"]),
            ("RAM: total increase (MiB)",
             s0["rss_total_delta_mib"], s1["rss_total_delta_mib"]),
        ]
        if s0.get("tf_gpu_peak_inference_mib", 0) > 0 and \
           s1.get("tf_gpu_peak_inference_mib", 0) > 0:
            rows += [
                ("GPU: after model load (MiB)",
                 s0["tf_gpu_after_load_mib"], s1["tf_gpu_after_load_mib"]),
                ("GPU: peak inference (MiB)",
                 s0["tf_gpu_peak_inference_mib"], s1["tf_gpu_peak_inference_mib"]),
            ]
        col0 = "Model v5"
        col1 = "mars-small128"
        print(f"  {'Metric':<30}  {col0:>14}  {col1:>14}")
        print(f"  {'-'*30}  {'-'*14}  {'-'*14}")
        for name, v0, v1 in rows:
            if isinstance(v0, int):
                print(f"  {name:<30}  {v0:>14,}  {v1:>14,}")
            else:
                ratio = v1 / v0 if v0 > 0 else float("nan")
                tag = f"  ({ratio:.1f}x)" if not (ratio != ratio) else ""
                print(f"  {name:<30}  {v0:>13.1f}  {v1:>13.1f}{tag}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
