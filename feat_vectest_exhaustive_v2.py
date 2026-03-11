"""
Exhaustive Pairwise Evaluation using the test split from split_manifest.json
=============================================================================
Key differences from feat_vectest_exhaustive.py:
  - Uses the proper open-set test split (IDs never seen during training).
  - Embeddings are extracted ONCE per image, then reused — no repeated forward passes.
  - Negative pair budget is controlled per class-pair, not divided by class count.
  - Random seed for reproducibility.
  - Results saved correctly (metrics as separate named arrays, not a dict).

Usage
-----
python feat_vectest_exhaustive_v2.py \\
    --pb_path  runs/turkey_reid/best_model.pb \\
    --manifest runs/turkey_reid/split_manifest.json \\
    [--max_pos_pairs 100] \\
    [--max_neg_pairs_per_combo 5] \\
    [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exhaustive pairwise evaluation on the test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pb_path",  default="runs/turkey_reid/best_model.pb",
                   help="Path to frozen .pb model.")
    p.add_argument("--manifest", default="runs/turkey_reid/split_manifest.json",
                   help="Path to split manifest JSON.")
    p.add_argument("--split",    default="test", choices=["train", "val", "test"],
                   help="Which split to evaluate on.")
    p.add_argument("--input_tensor",  default="images:0")
    p.add_argument("--output_tensor", default="features:0")
    p.add_argument("--image_h", type=int, default=128)
    p.add_argument("--image_w", type=int, default=64)
    p.add_argument("--max_pos_pairs", type=int, default=100,
                   help="Max positive pairs per identity.")
    p.add_argument("--max_neg_pairs_per_combo", type=int, default=5,
                   help="Max negative pairs per (id_i, id_j) combination.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for embedding extraction.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=".",
                   help="Directory to save plots and .npz results.")
    p.add_argument("--pdf", action="store_true",
                   help="Also export the evaluation plot as a PDF (IEEE-compatible, Type-1 fonts).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading and embedding extraction
# ---------------------------------------------------------------------------

def load_frozen_pb(pb_path: str) -> tf.Graph:
    with tf.io.gfile.GFile(pb_path, "rb") as fh:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(fh.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def preprocess_image(image_path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    """Load image as BGR uint8 then cast to float32.

    Normalization (÷255) is baked inside the .pb model as a Lambda layer,
    so we must NOT divide here — pass raw [0, 255] values.
    """
    h, w = target_hw
    img = cv2.imread(image_path)      # BGR
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (w, h))     # cv2.resize takes (width, height)
    return img.astype(np.float32)


def extract_all_embeddings(
    image_paths: List[str],
    graph: tf.Graph,
    sess: tf.compat.v1.Session,
    input_tensor: tf.Tensor,
    output_tensor: tf.Tensor,
    target_hw: Tuple[int, int],
    batch_size: int = 64,
) -> np.ndarray:
    """Extract embeddings for all images in one batched pass.

    Returns array of shape (N, embedding_dim).
    """
    n = len(image_paths)
    all_embs: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        batch_paths = image_paths[start: start + batch_size]
        batch_imgs = np.stack(
            [preprocess_image(p, target_hw) for p in batch_paths], axis=0
        )
        embs = sess.run(output_tensor, {input_tensor: batch_imgs})
        all_embs.append(embs)

        if (start // batch_size) % 5 == 0:
            print(f"  Extracting embeddings: {min(start + batch_size, n)}/{n}")

    return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def build_pairs(
    class_images: Dict[str, List[str]],
    image_index: Dict[str, int],
    max_pos_pairs: int,
    max_neg_pairs_per_combo: int,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return (positive_pairs, negative_pairs) as lists of index tuples."""

    class_names = sorted(class_images.keys())
    positive_pairs: List[Tuple[int, int]] = []
    negative_pairs: List[Tuple[int, int]] = []

    # Positive pairs: same identity
    for cls in class_names:
        imgs = class_images[cls]
        all_pos = list(combinations(range(len(imgs)), 2))
        if len(all_pos) > max_pos_pairs:
            chosen = rng.choice(len(all_pos), size=max_pos_pairs, replace=False)
            all_pos = [all_pos[i] for i in chosen]
        for a, b in all_pos:
            positive_pairs.append((image_index[imgs[a]], image_index[imgs[b]]))

    # Negative pairs: different identities
    for i, cls1 in enumerate(class_names):
        for cls2 in class_names[i + 1:]:
            imgs1 = class_images[cls1]
            imgs2 = class_images[cls2]

            # Build full cross product, then sample
            all_neg = [
                (image_index[a], image_index[b])
                for a in imgs1
                for b in imgs2
            ]
            if len(all_neg) > max_neg_pairs_per_combo:
                chosen = rng.choice(len(all_neg), size=max_neg_pairs_per_combo,
                                    replace=False)
                all_neg = [all_neg[i] for i in chosen]
            negative_pairs.extend(all_neg)

    return positive_pairs, negative_pairs


# ---------------------------------------------------------------------------
# Metrics and plotting
# ---------------------------------------------------------------------------

def cosine_similarity(embs: np.ndarray, idx_a: int, idx_b: int) -> float:
    a, b = embs[idx_a], embs[idx_b]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_metrics(positive_scores: np.ndarray, negative_scores: np.ndarray) -> dict:
    metrics: dict = {
        "positive_mean": float(np.mean(positive_scores)),
        "positive_std":  float(np.std(positive_scores)),
        "positive_min":  float(np.min(positive_scores)),
        "positive_max":  float(np.max(positive_scores)),
        "negative_mean": float(np.mean(negative_scores)),
        "negative_std":  float(np.std(negative_scores)),
        "negative_min":  float(np.min(negative_scores)),
        "negative_max":  float(np.max(negative_scores)),
    }

    y_true   = np.concatenate([np.ones(len(positive_scores)),
                                np.zeros(len(negative_scores))])
    y_scores = np.concatenate([positive_scores, negative_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    metrics["roc_auc"] = float(roc_auc)

    # Optimal threshold via Youden's J
    j = tpr - fpr
    opt_idx = int(np.argmax(j))
    opt_thr = float(thresholds[opt_idx])
    metrics["optimal_threshold"] = opt_thr
    metrics["optimal_tpr"]       = float(tpr[opt_idx])
    metrics["optimal_fpr"]       = float(fpr[opt_idx])

    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)
    metrics["pr_auc"] = float(auc(recall_arr, precision_arr))

    preds    = (y_scores > opt_thr).astype(int)
    metrics["accuracy"] = float(np.mean(preds == y_true))

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    metrics["precision"] = float(prec)
    metrics["recall"]    = float(rec)
    metrics["f1_score"]  = float(f1)

    metrics["separation"]          = float(metrics["positive_mean"] - metrics["negative_mean"])
    denom = np.sqrt(metrics["positive_std"] ** 2 + metrics["negative_std"] ** 2)
    metrics["separability_index"]  = float(metrics["separation"] / denom) if denom > 0 else 0.0

    return metrics, fpr, tpr, precision_arr, recall_arr


def plot_results(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    metrics: dict,
    fpr: np.ndarray,
    tpr: np.ndarray,
    precision_arr: np.ndarray,
    recall_arr: np.ndarray,
    save_path: str,
    save_pdf: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Score distributions
    axes[0, 0].hist(positive_scores, alpha=0.5, label="Positive", bins=30, density=True)
    axes[0, 0].hist(negative_scores, alpha=0.5, label="Negative", bins=30, density=True)
    axes[0, 0].axvline(metrics["optimal_threshold"], color="red", linestyle="--",
                       label=f"Threshold: {metrics['optimal_threshold']:.3f}")
    axes[0, 0].set_xlabel("Cosine Similarity")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Score Distributions")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. ROC Curve
    axes[0, 1].plot(fpr, tpr, "b-", label=f'AUC = {metrics["roc_auc"]:.3f}')
    axes[0, 1].plot([0, 1], [0, 1], "r--", alpha=0.5)
    axes[0, 1].scatter(metrics["optimal_fpr"], metrics["optimal_tpr"], color="red",
                       s=100, label=f'Optimal (FPR={metrics["optimal_fpr"]:.3f})')
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Precision-Recall Curve
    axes[0, 2].plot(recall_arr, precision_arr, "g-",
                    label=f'AUC = {metrics["pr_auc"]:.3f}')
    axes[0, 2].set_xlabel("Recall")
    axes[0, 2].set_ylabel("Precision")
    axes[0, 2].set_title("Precision-Recall Curve")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Box plot
    axes[1, 0].boxplot([positive_scores, negative_scores],
                       labels=["Positive", "Negative"])
    axes[1, 0].set_ylabel("Cosine Similarity")
    axes[1, 0].set_title("Score Comparison")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Metrics table
    txt = (
        f"ROC AUC:            {metrics['roc_auc']:.4f}\n"
        f"PR  AUC:            {metrics['pr_auc']:.4f}\n"
        f"Accuracy:           {metrics['accuracy']:.4f}\n"
        f"F1 Score:           {metrics['f1_score']:.4f}\n"
        f"Precision:          {metrics['precision']:.4f}\n"
        f"Recall:             {metrics['recall']:.4f}\n"
        f"Optimal Threshold:  {metrics['optimal_threshold']:.4f}\n"
        f"Separation (Δμ):    {metrics['separation']:.4f}\n"
        f"Separability Index: {metrics['separability_index']:.4f}"
    )
    axes[1, 1].text(0.05, 0.5, txt, fontsize=10,
                    verticalalignment="center", fontfamily="monospace",
                    transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Performance Metrics")
    axes[1, 1].axis("off")

    # 6. Confusion matrix
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)
    tp = int(np.sum(positive_scores > metrics["optimal_threshold"]))
    tn = int(np.sum(negative_scores <= metrics["optimal_threshold"]))
    fp = n_neg - tn
    fn = n_pos - tp
    cm = np.array([[tp, fn], [fp, tn]])
    axes[1, 2].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1, 2].set_title("Confusion Matrix")
    for r in range(2):
        for c in range(2):
            axes[1, 2].text(c, r, str(cm[r, c]), ha="center", va="center",
                            color="white" if cm[r, c] > cm.max() / 2 else "black")
    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_yticks([0, 1])
    axes[1, 2].set_xticklabels(["Pred Pos", "Pred Neg"])
    axes[1, 2].set_yticklabels(["True Pos", "True Neg"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[plot] Saved to {save_path}")
    if save_pdf:
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path, bbox_inches="tight",
                    metadata={"Creator": "feat_vectest_exhaustive_v2"})
        print(f"[plot] Saved to {pdf_path}")


def print_summary(positive_scores: np.ndarray, negative_scores: np.ndarray,
                  metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY  (test split only — open-set IDs)")
    print("=" * 60)
    print(f"Positive pairs : {len(positive_scores)}")
    print(f"Negative pairs : {len(negative_scores)}")
    print(f"ROC AUC        : {metrics['roc_auc']:.4f}")
    print(f"PR AUC         : {metrics['pr_auc']:.4f}")
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"F1 Score       : {metrics['f1_score']:.4f}")
    print(f"Opt. Threshold : {metrics['optimal_threshold']:.4f}")
    print(f"Separation Δμ  : {metrics['separation']:.4f}")
    print(f"Pos  μ ± σ     : {metrics['positive_mean']:.4f} ± {metrics['positive_std']:.4f}")
    print(f"Neg  μ ± σ     : {metrics['negative_mean']:.4f} ± {metrics['negative_std']:.4f}")
    print(f"Separability   : {metrics['separability_index']:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    if args.pdf:
        import matplotlib
        matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    # ---- Load manifest and extract test split ----
    with open(args.manifest) as fh:
        manifest = json.load(fh)

    split_data: Dict[str, List[str]] = manifest["splits"][args.split]
    # Paths in the manifest are relative to the working directory (project root).
    # Resolve them against CWD so the script works regardless of manifest location.
    cwd = Path.cwd()
    class_images: Dict[str, List[str]] = {
        cls: [str(cwd / p) for p in paths]
        for cls, paths in split_data.items()
        if len(paths) >= 2          # need ≥2 images to form any positive pair
    }

    print(f"\n[data] Split '{args.split}': {len(class_images)} IDs "
          f"({sum(len(v) for v in class_images.values())} images)")
    for cls, imgs in sorted(class_images.items()):
        print(f"  {cls}: {len(imgs)} images")

    # ---- Flat list of all images + index lookup ----
    all_paths: List[str] = []
    for imgs in class_images.values():
        all_paths.extend(imgs)
    all_paths = sorted(set(all_paths))          # deduplicate, deterministic order
    image_index: Dict[str, int] = {p: i for i, p in enumerate(all_paths)}

    # ---- Load model ----
    print(f"\n[model] Loading .pb from {args.pb_path} ...")
    graph = load_frozen_pb(args.pb_path)
    sess  = tf.compat.v1.Session(graph=graph)
    in_t  = graph.get_tensor_by_name(args.input_tensor)
    out_t = graph.get_tensor_by_name(args.output_tensor)

    # ---- Extract embeddings (one forward pass per image) ----
    print("[model] Extracting embeddings ...")
    t0 = time.time()
    all_embs = extract_all_embeddings(
        all_paths, graph, sess, in_t, out_t,
        target_hw=(args.image_h, args.image_w),
        batch_size=args.batch_size,
    )
    print(f"[model] Done in {time.time() - t0:.1f}s  "
          f"| norm mean={np.linalg.norm(all_embs, axis=1).mean():.4f} (expect ~1.0)")

    # ---- Generate pairs ----
    print("\n[pairs] Generating pairs ...")
    pos_pairs, neg_pairs = build_pairs(
        class_images, image_index,
        args.max_pos_pairs,
        args.max_neg_pairs_per_combo,
        rng,
    )
    print(f"[pairs] Positive: {len(pos_pairs)}  |  Negative: {len(neg_pairs)}")

    # ---- Compute similarities (index lookup — no extra forward passes) ----
    print("\n[eval] Computing cosine similarities ...")
    positive_scores = np.array([cosine_similarity(all_embs, a, b) for a, b in pos_pairs])
    negative_scores = np.array([cosine_similarity(all_embs, a, b) for a, b in neg_pairs])

    # ---- Metrics ----
    print("[eval] Computing metrics ...")
    metrics, fpr, tpr, prec_arr, rec_arr = compute_metrics(positive_scores, negative_scores)
    print_summary(positive_scores, negative_scores, metrics)

    # ---- Save results ----
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "evaluation_results_v2.png")
    plot_results(positive_scores, negative_scores, metrics,
                 fpr, tpr, prec_arr, rec_arr, save_path=plot_path,
                 save_pdf=args.pdf)

    # Save scores + metrics as separate named arrays (np.savez can't store dicts)
    npz_path = os.path.join(args.output_dir, "evaluation_results_v2.npz")
    np.savez(npz_path,
             positive_scores=positive_scores,
             negative_scores=negative_scores,
             **{k: np.array(v) for k, v in metrics.items()})
    print(f"[save] Results saved to {npz_path}")


if __name__ == "__main__":
    main()
