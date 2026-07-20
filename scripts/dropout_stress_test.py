#!/usr/bin/env python3
"""Detection-Dropout Stress Test: Custom vs Baseline Embedding
================================================================
Addresses Reviewer 1 (#4-5): "Artificially induce motion ambiguity by
randomly dropping detection boxes during inference (e.g. 10%, 20%, 30%).
This forces the tracker to rely heavily on the appearance pathway rather
than the Kalman filter. If your model recovers trajectories significantly
better than the baseline under high detection failure rates, it provides
undeniable proof of its robustness."

It also gives Reviewer 2 (#6, thin n=3 tracking evidence) a controlled,
repeatable experiment rather than more raw footage: instead of 3 sequences
x 1 condition, this produces 3 sequences x 4 dropout rates x N repeats,
which supports an actual paired comparison per dropout level.

WHY THIS USES GROUND-TRUTH BOXES AS THE DETECTION SOURCE
----------------------------------------------------------
The three evaluated sequences (seq1/seq2/seq3) have GT annotations in
TrackEval/data/gt/mot_challenge/Turkey-test/seqX/gt/gt.txt but the raw
per-frame YOLO detection dumps used for the paper's Table 5/6 don't appear
to be cached anywhere in this repo (only the final track outputs are).
Re-running YOLOv8s from scratch on the source videos is possible but adds
a detector-noise confound. Using GT boxes as a "perfect detector" and then
randomly dropping a controlled fraction is actually the CLEANER version of
what the reviewer asked for: it isolates the effect of detection dropout
from detector false positives/negatives, which is exactly what "artificially
induce motion ambiguity" calls for. State this framing explicitly in the
rebuttal — it's a feature of the design, not a shortcut, but it should be
disclosed.

If you'd rather use real YOLO detections with natural + injected dropout,
see USE_REAL_DETECTOR below and swap in deep_sort/tools/generate_detections.py
style raw boxes before the dropout step; the tracking + dropout + metrics
logic is unchanged either way.

The seq-to-video mapping below was resolved by matching each candidate
video's frame count + resolution (via cv2) against each seqinfo.ini's
seqLength/imWidth/imHeight — all three matches are exact:
    seq1: 1117 frames, 1920x1080 == data/output_video_21.mp4
    seq2:  594 frames, 2092x1160 == data/smooth_best_11.mp4
    seq3:  477 frames, 2092x1160 == data/smooth_best_21.mp4
(output_video_1.mp4 / output_video_2.mp4, at 4516/4521 frames, are unrelated
raw footage not part of the three evaluated sequences.)

Colab usage
-----------
    !python scripts/dropout_stress_test.py \\
        --dropout_rates 0.0 0.1 0.2 0.3 \\
        --n_seeds 5 \\
        --output_csv runs/stress_test/dropout_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# motmetrics (as of the versions available on PyPI at the time of writing)
# still calls the long-removed np.asfarray internally (distances.py). NumPy
# >=2.0 dropped it entirely, so patch it back in before importing motmetrics
# rather than downgrading NumPy globally, which would risk breaking whatever
# else in the Colab image is built against the NumPy 2.x ABI.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet


# ===========================================================================
# Sequence -> (video path, gt path) mapping
# ===========================================================================
# Resolved by exact frame-count + resolution match (see module docstring).
SEQUENCES: Dict[str, Dict[str, str]] = {
    "seq1": {
        "video": "data/output_video_21.mp4",
        "gt": "TrackEval/data/gt/mot_challenge/Turkey-test/seq1/gt/gt.txt",
    },
    "seq2": {
        "video": "data/smooth_best_11.mp4",
        "gt": "TrackEval/data/gt/mot_challenge/Turkey-test/seq2/gt/gt.txt",
    },
    "seq3": {
        "video": "data/smooth_best_21.mp4",
        "gt": "TrackEval/data/gt/mot_challenge/Turkey-test/seq3/gt/gt.txt",
    },
}

EMBEDDINGS: Dict[str, str] = {
    "baseline_mars": "model_feature_extractor/mars-small128.pb",
    "custom_v5": "runs/turkey_reid_v5/best_model.pb",
}

USE_REAL_DETECTOR = False  # see module docstring


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detection-dropout stress test: custom vs baseline embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dropout_rates", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3])
    p.add_argument("--n_seeds", type=int, default=5,
                   help="Repeats per dropout rate (dropout is stochastic; "
                        "report mean +/- std across seeds, not a single draw).")
    p.add_argument("--max_cosine_distance", type=float, default=0.10,
                   help="Matches the operating point already used in tracker.py.")
    p.add_argument("--nn_budget", type=int, default=100)
    p.add_argument("--output_csv", default="runs/stress_test/dropout_results.csv")
    p.add_argument("--output_dir", default="runs/stress_test",
                   help="Where per-run MOT-format hypothesis files are saved.")
    return p.parse_args()


# ===========================================================================
# GT / detections loading
# ===========================================================================

def load_gt_as_frame_dets(gt_path: str) -> Dict[int, np.ndarray]:
    """Load MOT-format gt.txt into {frame: (N,4) array of [x,y,w,h]}.

    Ground-truth rows are treated as a "perfect detector": we deliberately
    discard the GT identity column so the tracker must re-derive identity
    itself, exactly as it would from a real detector's output.
    """
    raw = np.loadtxt(gt_path, delimiter=",")
    frames: Dict[int, np.ndarray] = {}
    for row in raw:
        frame = int(row[0])
        box = row[2:6]  # x, y, w, h
        frames.setdefault(frame, []).append(box)
    return {f: np.array(v, dtype=np.float64) for f, v in frames.items()}


def apply_dropout(
    boxes: np.ndarray, rate: float, rng: np.random.Generator
) -> np.ndarray:
    if rate <= 0.0 or len(boxes) == 0:
        return boxes
    keep_mask = rng.random(len(boxes)) >= rate
    return boxes[keep_mask]


# ===========================================================================
# Tracking
# ===========================================================================

def run_tracker_on_sequence(
    video_path: str,
    frame_dets: Dict[int, np.ndarray],
    encoder,
    max_cosine_distance: float,
    nn_budget: int,
) -> List[Tuple[int, int, float, float, float, float]]:
    """Run DeepSORT over a video given a fixed dict of per-frame detections
    (already dropout-applied). Returns MOT-format rows: (frame, id, x,y,w,h).
    """
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = DeepSortTracker(metric)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("Could not open video: {}".format(video_path))

    rows: List[Tuple[int, int, float, float, float, float]] = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        boxes = frame_dets.get(frame_idx, np.zeros((0, 4)))

        if len(boxes) == 0:
            tracker.predict()
            tracker.update([])
            continue

        features = encoder(frame, boxes)
        scores = np.ones(len(boxes), dtype=np.float32)
        dets = [Detection(boxes[i], scores[i], features[i]) for i in range(len(boxes))]

        tracker.predict()
        tracker.update(dets)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1, y1, x2, y2 = track.to_tlbr()
            rows.append((frame_idx, track.track_id, x1, y1, x2 - x1, y2 - y1))

    cap.release()
    return rows


# ===========================================================================
# Metrics (mirrors evaluate_tracker.py's compute_mota, but returns a dict
# instead of only printing, and adds IDsw explicitly for Table-5-style output)
# ===========================================================================

def compute_metrics(gt_path: str, hyp_rows: List[Tuple[int, int, float, float, float, float]]) -> Dict[str, float]:
    gt = np.loadtxt(gt_path, delimiter=",")
    if len(hyp_rows) == 0:
        return dict(idf1=0.0, mota=0.0, num_switches=float(int(gt[:, 0].max())))
    hyp = np.array(hyp_rows, dtype=np.float64)

    acc = mm.MOTAccumulator(auto_id=True)
    max_frame = int(gt[:, 0].max())
    for frame in range(1, max_frame + 1):
        gt_dets = gt[gt[:, 0] == frame, 1:6]
        hyp_dets = hyp[hyp[:, 0] == frame, 1:6]

        C = mm.distances.iou_matrix(gt_dets[:, 1:], hyp_dets[:, 1:], max_iou=0.5)
        acc.update(
            gt_dets[:, 0].astype(int).tolist(),
            hyp_dets[:, 0].astype(int).tolist(),
            C,
        )

    mh = mm.metrics.create()
    summary = mh.compute(
        acc, metrics=["idf1", "mota", "num_switches", "num_fragmentations"],
        name="acc",
    )
    return dict(
        idf1=float(summary["idf1"].iloc[0]),
        mota=float(summary["mota"].iloc[0]),
        num_switches=float(summary["num_switches"].iloc[0]),
        num_fragmentations=float(summary["num_fragmentations"].iloc[0]),
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    for name, cfg in SEQUENCES.items():
        if "FILL_IN" in cfg["video"]:
            raise ValueError(
                "SEQUENCES['{}']['video'] is still a placeholder. Edit the "
                "SEQUENCES dict at the top of this script to point at the "
                "correct source video before running.".format(name)
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []

    for embed_name, pb_path in EMBEDDINGS.items():
        print("\n" + "#" * 78)
        print("# Embedding: {}  ({})".format(embed_name, pb_path))
        print("#" * 78)
        encoder = gdet.create_box_encoder(pb_path, batch_size=32)

        for seq_name, cfg in SEQUENCES.items():
            frame_dets_full = load_gt_as_frame_dets(cfg["gt"])

            for rate in args.dropout_rates:
                idf1s, motas, idsws = [], [], []
                n_seeds = 1 if rate == 0.0 else args.n_seeds  # no dropout -> deterministic

                for seed in range(n_seeds):
                    rng = np.random.default_rng(seed)
                    dropped = {
                        f: apply_dropout(boxes, rate, rng)
                        for f, boxes in frame_dets_full.items()
                    }

                    hyp_rows = run_tracker_on_sequence(
                        cfg["video"], dropped, encoder,
                        args.max_cosine_distance, args.nn_budget,
                    )
                    m = compute_metrics(cfg["gt"], hyp_rows)
                    idf1s.append(m["idf1"])
                    motas.append(m["mota"])
                    idsws.append(m["num_switches"])

                    tag = "{}_{}_drop{:.0f}_seed{}".format(embed_name, seq_name, rate * 100, seed)
                    hyp_path = out_dir / "{}.txt".format(tag)
                    with open(hyp_path, "w") as fh:
                        for r in hyp_rows:
                            fh.write("{},{},{:.2f},{:.2f},{:.2f},{:.2f},1,1,1.0\n".format(*r))

                    print("[{}] {} drop={:.0%} seed={} -> IDF1={:.4f} MOTA={:.4f} IDsw={:.0f}".format(
                        embed_name, seq_name, rate, seed, m["idf1"], m["mota"], m["num_switches"]
                    ))

                results.append(dict(
                    embedding=embed_name, sequence=seq_name, dropout_rate=rate,
                    idf1_mean=float(np.mean(idf1s)), idf1_std=float(np.std(idf1s)),
                    mota_mean=float(np.mean(motas)), mota_std=float(np.std(motas)),
                    idsw_mean=float(np.mean(idsws)), idsw_std=float(np.std(idsws)),
                    n_seeds=n_seeds,
                ))

    # ---- Save CSV ----
    import csv
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["embedding", "sequence", "dropout_rate",
                  "idf1_mean", "idf1_std", "mota_mean", "mota_std",
                  "idsw_mean", "idsw_std", "n_seeds"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("\n[save] Results CSV -> {}".format(out_csv))

    # ---- Print summary table aggregated across sequences ----
    print("\n" + "=" * 78)
    print("  Mean IDF1 across sequences, by embedding x dropout rate")
    print("=" * 78)
    for embed_name in EMBEDDINGS:
        print("\n{}:".format(embed_name))
        for rate in args.dropout_rates:
            rows = [r for r in results if r["embedding"] == embed_name and r["dropout_rate"] == rate]
            mean_idf1 = np.mean([r["idf1_mean"] for r in rows])
            mean_idsw = np.mean([r["idsw_mean"] for r in rows])
            print("  drop={:.0%}: mean IDF1={:.4f}  mean IDsw={:.1f}".format(
                rate, mean_idf1, mean_idsw))


if __name__ == "__main__":
    main()
