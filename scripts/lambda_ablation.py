#!/usr/bin/env python3
"""Lambda (ID-loss weight) Sensitivity Sweep — Colab driver
=============================================================
Addresses Reviewer 1 (#6-8) and Reviewer 2 (Minor #10):
"The text states lambda = 0.5 was selected empirically ... add a minimal
sub-ablation showing performance at lambda = 0.1, 0.5, and 1.0."

This is a TRAINING task (re-runs V5 from scratch 3x), so it needs a GPU —
run this in Colab, not on the local laptop. It orchestrates the *existing*
scripts already in this repo (does not reimplement training):

  1. scripts/train_reid_v2.py   --arch v5 --id_loss_weight {lambda}
  2. scripts/eval_reid.py                     (Rank-1/5/10 + mAP)
  3. feat_vectest_exhaustive_v2.py            (ROC AUC, separability, etc.)

All three lambda runs reuse the SAME split manifest (identical train/val/
test identity split) so the comparison isolates lambda exactly the way the
paper's main ablation isolates each architectural element ("every variant
trained under identical data, sampling, augmentation, optimization
settings" — Sec 4.1).

Colab usage
-----------
    !git clone <your-repo-url> && cd Domain-Specific-Siamese-Embedding-in-DeepSORT
    !pip install -r requirements.txt   # or the Colab-specific TF/torch combo
    !python scripts/lambda_ablation.py \\
        --dataset_path dataset_siam_21 \\
        --reference_manifest runs/turkey_reid_v5/split_manifest.json \\
        --lambdas 0.1 0.5 1.0 \\
        --epochs 200 --patience 25 \\
        --output_root runs/lambda_sweep

If --reference_manifest does not exist yet (e.g. fresh Colab checkout with
no prior runs/ directory), the FIRST lambda run creates a fresh manifest
and every subsequent lambda run reuses it automatically — still a fair,
identical-split comparison, it just won't match the exact split used for
the paper's main V5 result unless you copy that manifest over first.

Output
------
runs/lambda_sweep/lambda_0.1/   (etc.)   — full train_reid_v2.py output dir
runs/lambda_sweep/lambda_ablation_summary.csv  — Table-1-style summary:
    lambda, rank1, rank5, rank10, mAP, roc_auc, separability_index
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lambda (ID-loss weight) sensitivity sweep for V5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_path", default="dataset_siam_21")
    p.add_argument("--reference_manifest", default="runs/turkey_reid_v5/split_manifest.json",
                   help="Reused across all lambda runs for a fair comparison. "
                        "If missing, the first run creates one and later runs reuse it.")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                   help="ID-loss weight values to sweep (0.5 is the paper's default).")
    p.add_argument("--output_root", default="runs/lambda_sweep")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--P", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=2e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_training", action="store_true",
                   help="Skip the train step and only (re)run eval on existing "
                        "runs/lambda_sweep/lambda_X dirs (useful if training already "
                        "finished in an earlier Colab session and only eval needs a rerun).")
    return p.parse_args()


def run(cmd: List[str]) -> str:
    print("\n[run] {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-4000:])
        raise RuntimeError("Command failed (exit {}): {}".format(
            result.returncode, " ".join(cmd)))
    return result.stdout


def parse_eval_reid_output(stdout: str) -> Dict[str, float]:
    """Pull Rank-1/5/10 and mAP out of scripts/eval_reid.py's printed table."""
    out: Dict[str, float] = {}
    for k in (1, 5, 10):
        m = re.search(r"Rank-\s*{}\s*:\s*([0-9.]+)".format(k), stdout)
        if m:
            out["rank{}".format(k)] = float(m.group(1))
    m = re.search(r"mAP\s*:\s*([0-9.]+)", stdout)
    if m:
        out["mAP"] = float(m.group(1))
    return out


def parse_exhaustive_eval_output(stdout: str) -> Dict[str, float]:
    """Pull ROC AUC / separability index out of feat_vectest_exhaustive_v2.py output."""
    out: Dict[str, float] = {}
    m = re.search(r"ROC AUC\s*:\s*([0-9.]+)", stdout)
    if m:
        out["roc_auc"] = float(m.group(1))
    m = re.search(r"Separability\s*:\s*([0-9.]+)", stdout)
    if m:
        out["separability_index"] = float(m.group(1))
    m = re.search(r"Separation.*?:\s*([0-9.]+)", stdout)
    if m:
        out["separation"] = float(m.group(1))
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    reference_manifest = Path(args.reference_manifest)
    summary_rows: List[Dict[str, float]] = []

    for lam in args.lambdas:
        tag = "lambda_{}".format(lam)
        run_dir = output_root / tag
        print("\n" + "#" * 78)
        print("# {}".format(tag))
        print("#" * 78)

        already_trained = (run_dir / "best_model.pb").exists()
        if already_trained:
            print("[lambda_ablation] {} already has best_model.pb -- skipping "
                  "training and reusing the existing run.".format(tag))

        if not args.skip_training and not already_trained:
            cmd = [
                sys.executable, "scripts/train_reid_v2.py",
                "--arch", "v5",
                "--dataset_path", args.dataset_path,
                "--output_dir", str(run_dir),
                "--id_loss_weight", str(lam),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--P", str(args.P), "--K", str(args.K),
                "--lr", str(args.lr),
                "--weight_decay", str(args.weight_decay),
                "--dropout", str(args.dropout),
                "--seed", str(args.seed),
            ]
            # Reuse a fixed manifest across all lambda runs once one exists.
            if reference_manifest.exists():
                cmd += ["--split_manifest", str(reference_manifest)]
            run(cmd)
            # First run establishes the reference manifest for subsequent runs.
            produced_manifest = run_dir / "split_manifest.json"
            if not reference_manifest.exists() and produced_manifest.exists():
                reference_manifest = produced_manifest
                print("[lambda_ablation] Using {} as the reference manifest "
                      "for all remaining lambda runs.".format(reference_manifest))

        manifest_for_eval = run_dir / "split_manifest.json"
        if not manifest_for_eval.exists():
            manifest_for_eval = reference_manifest

        pb_path = run_dir / "best_model.pb"

        # NOTE: scripts/eval_reid.py always instantiates the V1 architecture
        # internally (no --arch flag) and will fail with a "layer count
        # mismatch" when loading V5 weights. scripts/eval_reid_pb.py evaluates
        # directly from the exported frozen .pb graph instead, which is
        # architecture-agnostic (same approach feat_vectest_exhaustive_v2.py
        # already uses below) and sidesteps the issue entirely.
        eval_stdout = run([
            sys.executable, "scripts/eval_reid_pb.py",
            "--model_pb", str(pb_path),
            "--manifest", str(manifest_for_eval),
        ])
        row: Dict[str, float] = {"lambda": lam}
        row.update(parse_eval_reid_output(eval_stdout))

        exhaustive_stdout = run([
            sys.executable, "feat_vectest_exhaustive_v2.py",
            "--pb_path", str(pb_path),
            "--manifest", str(manifest_for_eval),
            "--output_dir", str(run_dir),
        ])
        row.update(parse_exhaustive_eval_output(exhaustive_stdout))

        summary_rows.append(row)
        print("[lambda_ablation] {} -> {}".format(tag, row))

    # ---- Summary CSV ----
    fieldnames = ["lambda", "rank1", "rank5", "rank10", "mAP",
                  "roc_auc", "separation", "separability_index"]
    out_csv = output_root / "lambda_ablation_summary.csv"
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print("\n" + "=" * 78)
    print("  Lambda sensitivity summary")
    print("=" * 78)
    print("{:<10}{:>10}{:>10}{:>10}{:>10}{:>12}{:>16}".format(
        "lambda", "Rank-1", "Rank-5", "Rank-10", "mAP", "ROC AUC", "Separability"))
    for row in summary_rows:
        print("{:<10}{:>10}{:>10}{:>10}{:>10}{:>12}{:>16}".format(
            row.get("lambda", ""),
            row.get("rank1", ""), row.get("rank5", ""), row.get("rank10", ""),
            row.get("mAP", ""), row.get("roc_auc", ""),
            row.get("separability_index", ""),
        ))
    print("\n[save] Summary CSV -> {}".format(out_csv))


if __name__ == "__main__":
    main()
