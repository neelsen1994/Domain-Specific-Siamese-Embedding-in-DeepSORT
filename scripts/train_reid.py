#!/usr/bin/env python3
"""Turkey ReID Training Pipeline
==================================
Trains a DeepSORT-compatible 128-D L2-normalised embedding model using
batch-hard triplet loss with PK (P identities × K images) sampling.

Key features
------------
* Deterministic train/val/test split saved to a JSON manifest.
* PK batch sampler with with-replacement fallback for small IDs.
* Batch-hard triplet loss (Hermans et al., 2017).
* OpenCV-based augmentation tuned for barn / livestock imagery.
* Cosine LR decay with linear warm-up.
* Checkpointing of the best model by val rank-1.
* Early stopping.
* Automatic .pb export of the best weights on training completion.

Usage
-----
# Fresh training
python scripts/train_reid.py \\
    --dataset_path dataset_siam_21 \\
    --output_dir   runs/turkey_reid \\
    --epochs 150 --P 16 --K 4 --lr 1e-3 --seed 42

# Resume from a checkpoint
python scripts/train_reid.py \\
    --dataset_path dataset_siam_21 \\
    --output_dir   runs/turkey_reid \\
    --resume       runs/turkey_reid/checkpoints/best_weights.h5 \\
    --epochs 200

# Use an existing split manifest (skip re-splitting)
python scripts/train_reid.py \\
    --dataset_path dataset_siam_21 \\
    --output_dir   runs/turkey_reid \\
    --split_manifest runs/turkey_reid/split_manifest.json

Output files
------------
runs/turkey_reid/
    split_manifest.json          deterministic split (reproducible)
    checkpoints/best_weights.h5  best Keras weights (by val rank-1)
    best_model.pb                frozen .pb for DeepSORT inference
    best_model_keras.h5          alias of the best weights
    training_log.json            per-epoch metrics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

# ---- local imports (works when run from repo root) ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dataset import (
    load_dataset_dict,
    create_split,
    load_split_from_manifest,
    PKSampler,
    load_split_as_arrays,
)
from scripts.model import build_embedding_model
from scripts.losses import batch_hard_triplet_loss
from scripts.augmentation import ReIDAugmentation
from scripts.metrics import compute_rank1


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train turkey ReID embedding model (batch-hard triplet loss)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Dataset ---
    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset_path", default="dataset_siam_21",
                   help="Root dir; one sub-folder per identity.")
    g.add_argument("--split_manifest", default=None,
                   help="Path to an existing split JSON.  If omitted and no "
                        "manifest exists in output_dir, a new split is created.")
    g.add_argument("--train_ratio", type=float, default=0.70)
    g.add_argument("--val_ratio",   type=float, default=0.15)
    g.add_argument("--seed", type=int, default=42,
                   help="Random seed for split + PK sampling.")

    # --- Model ---
    g = p.add_argument_group("Model")
    g.add_argument("--image_h",       type=int,   default=128)
    g.add_argument("--image_w",       type=int,   default=64)
    g.add_argument("--embedding_dim", type=int,   default=128)
    g.add_argument("--num_filters",   type=int,   default=32,
                   help="Base channel count (doubles after first MaxPool).")
    g.add_argument("--dropout",       type=float, default=0.30)

    # --- PK Sampling ---
    g = p.add_argument_group("PK Sampling")
    g.add_argument("--P", type=int, default=16,
                   help="Identities per batch.")
    g.add_argument("--K", type=int, default=4,
                   help="Images per identity per batch.  batch_size = P * K.")
    g.add_argument("--steps_per_epoch", type=int, default=50,
                   help="Training steps per epoch (with replacement).")

    # --- Training ---
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",       type=int,   default=150)
    g.add_argument("--lr",           type=float, default=1e-3)
    g.add_argument("--margin",       type=float, default=0.30,
                   help="Triplet loss margin.")
    g.add_argument("--warmup_epochs", type=int,  default=5,
                   help="Linear LR warm-up duration in epochs.")
    g.add_argument("--patience",     type=int,   default=25,
                   help="Early-stopping patience in epochs.")
    g.add_argument("--no_augment",   action="store_true",
                   help="Disable data augmentation (for debugging).")
    g.add_argument("--resume",       default=None,
                   help="Path to .h5 weights to resume from.")

    # --- Output ---
    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", default="runs/turkey_reid",
                   help="Directory for checkpoints, logs, and exported .pb.")
    g.add_argument("--no_export_pb", action="store_true",
                   help="Skip .pb export after training.")

    return p.parse_args()


# ===========================================================================
# Learning-rate schedule: linear warm-up + cosine decay
# ===========================================================================

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up to *peak_lr*, then cosine decay to *min_lr*."""

    def __init__(
        self,
        peak_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.peak_lr      = float(peak_lr)
        self.total_steps  = float(total_steps)
        self.warmup_steps = float(warmup_steps)
        self.min_lr       = float(min_lr)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step    = tf.cast(step, tf.float32)
        warmup  = self.warmup_steps
        total   = self.total_steps
        peak    = self.peak_lr
        mn      = self.min_lr

        # Warm-up: linear 0 → peak
        warmup_lr = peak * step / tf.maximum(warmup, 1.0)

        # Cosine decay: peak → min_lr
        progress  = (step - warmup) / tf.maximum(total - warmup, 1.0)
        progress  = tf.clip_by_value(progress, 0.0, 1.0)
        cos_lr    = mn + 0.5 * (peak - mn) * (1.0 + tf.cos(np.pi * progress))

        return tf.where(step < warmup, warmup_lr, cos_lr)

    def get_config(self) -> dict:
        return dict(
            peak_lr=self.peak_lr,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            min_lr=self.min_lr,
        )


# ===========================================================================
# Frozen .pb export
# ===========================================================================

def export_frozen_pb(model: tf.keras.Model, output_path: str, input_shape: tuple) -> None:
    """Export Keras model to a frozen TF .pb graph.

    Input  node: "images"   — raw BGR float32 [0, 255]
    Output node: "features" — L2-normalised 128-D embedding

    The graph is compatible with DeepSORT's generate_detections.py when that
    file loads the graph with  tf.import_graph_def(graph_def, name="")
    (no prefix) and uses input_name="images", output_name="features".
    """
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    @tf.function(input_signature=[
        tf.TensorSpec([None] + list(input_shape), tf.float32, name="images")
    ])
    def _serving(x: tf.Tensor) -> tf.Tensor:
        emb = model(x, training=False)
        # Explicit identity op ensures the output is named "features" in the graph
        return tf.identity(emb, name="features")

    concrete = _serving.get_concrete_function()
    frozen   = convert_variables_to_constants_v2(concrete)

    out_dir  = os.path.dirname(os.path.abspath(output_path))
    out_name = os.path.basename(output_path)
    os.makedirs(out_dir, exist_ok=True)

    tf.io.write_graph(
        graph_or_graph_def=frozen.graph,
        logdir=out_dir,
        name=out_name,
        as_text=False,
    )
    print("[export] Frozen .pb saved -> {}".format(output_path))

    # Verify expected node names
    node_names = {op.name for op in frozen.graph.get_operations()}
    print("[export] 'images'   node found: {}".format("images" in node_names))
    print("[export] 'features' node found: {}".format("features" in node_names))


# ===========================================================================
# Embedding extraction + validation
# ===========================================================================

def extract_embeddings(
    model: tf.keras.Model,
    images: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Run model on *images* in mini-batches; return (N, D) embeddings."""
    embs: List[np.ndarray] = []
    for i in range(0, len(images), batch_size):
        batch = tf.constant(images[i: i + batch_size], dtype=tf.float32)
        embs.append(model(batch, training=False).numpy())
    return np.concatenate(embs, axis=0)


def validate(
    model: tf.keras.Model,
    train_data: Dict[str, List[str]],
    val_data: Dict[str, List[str]],
    image_size: Tuple[int, int],
    batch_size: int = 64,
) -> float:
    """Rank-1 accuracy: val images as queries, train images as gallery.

    The two splits are disjoint by construction, so no self-removal is needed.
    """
    q_imgs, q_lbls, _ = load_split_as_arrays(val_data,   image_size)
    g_imgs, g_lbls, _ = load_split_as_arrays(train_data, image_size)

    q_embs = extract_embeddings(model, q_imgs, batch_size)
    g_embs = extract_embeddings(model, g_imgs, batch_size)

    return compute_rank1(q_embs, q_lbls, g_embs, g_lbls, remove_self=False)


# ===========================================================================
# Main training loop
# ===========================================================================

def train(args: argparse.Namespace) -> tf.keras.Model:
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    image_size  = (args.image_h, args.image_w)
    input_shape = (args.image_h, args.image_w, 3)
    batch_size  = args.P * args.K

    # ------------------------------------------------------------------
    # 1. Dataset split
    # ------------------------------------------------------------------
    dataset       = load_dataset_dict(args.dataset_path)
    manifest_path = str(output_dir / "split_manifest.json")

    if args.split_manifest and os.path.exists(args.split_manifest):
        print("[train] Using provided manifest: {}".format(args.split_manifest))
        train_data, val_data, test_data = load_split_from_manifest(
            args.split_manifest
        )
    elif os.path.exists(manifest_path):
        print("[train] Loading existing manifest: {}".format(manifest_path))
        train_data, val_data, test_data = load_split_from_manifest(manifest_path)
    else:
        print("[train] Creating new deterministic split ...")
        train_data, val_data, test_data = create_split(
            dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            manifest_path=manifest_path,
        )

    # ------------------------------------------------------------------
    # 2. Augmentation + PK sampler
    # ------------------------------------------------------------------
    aug_fn = None if args.no_augment else ReIDAugmentation(image_size=image_size)

    sampler = PKSampler(
        data=train_data,
        P=args.P,
        K=args.K,
        image_size=image_size,
        augment_fn=aug_fn,
        seed=args.seed,
    )
    print("[train] PK sampler: P={} K={} -> batch={}  natural_steps/epoch={}".format(
        args.P, args.K, batch_size, len(sampler)
    ))

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = build_embedding_model(
        input_shape=input_shape,
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        dropout_rate=args.dropout,
    )
    model.summary(line_length=90)
    print("[train] Trainable params: {:,}".format(model.count_params()))

    if args.resume:
        print("[train] Resuming weights from: {}".format(args.resume))
        model.load_weights(args.resume)

    # ------------------------------------------------------------------
    # 4. Optimiser + LR schedule
    # ------------------------------------------------------------------
    total_steps  = args.epochs * args.steps_per_epoch
    warmup_steps = args.warmup_epochs * args.steps_per_epoch

    lr_schedule = WarmupCosineDecay(
        peak_lr=args.lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    best_rank1        = -1.0
    best_weights_path = str(ckpt_dir / "best_weights.h5")
    no_improve_count  = 0

    log = {"args": vars(args), "epochs": []}
    log_path = str(output_dir / "training_log.json")

    header = (
        "\n" + "=" * 70 + "\n"
        "Training configuration\n"
        "  Epochs:          {epochs}\n"
        "  Steps/epoch:     {steps}\n"
        "  Batch (P×K):     {P}×{K}={bs}\n"
        "  Triplet margin:  {margin}\n"
        "  Peak LR:         {lr}\n"
        "  Warmup epochs:   {wu}\n"
        "  Early-stop pat.: {pat}\n"
        "  Augmentation:    {aug}\n"
        + "=" * 70
    ).format(
        epochs=args.epochs,
        steps=args.steps_per_epoch,
        P=args.P, K=args.K, bs=batch_size,
        margin=args.margin, lr=args.lr,
        wu=args.warmup_epochs, pat=args.patience,
        aug="off" if args.no_augment else "on",
    )
    print(header)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        ep_loss, ep_frac, ep_pos = [], [], []

        for images, labels in sampler.epoch_generator(args.steps_per_epoch):
            images_tf = tf.constant(images, dtype=tf.float32)
            labels_tf = tf.constant(labels, dtype=tf.int32)

            with tf.GradientTape() as tape:
                embs = model(images_tf, training=True)
                loss, frac_active, mean_pos = batch_hard_triplet_loss(
                    embs, labels_tf, margin=args.margin
                )

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            ep_loss.append(float(loss))
            ep_frac.append(float(frac_active))
            ep_pos.append(float(mean_pos))

        # ---- end-of-epoch metrics ----
        mean_loss = float(np.mean(ep_loss))
        mean_frac = float(np.mean(ep_frac))
        mean_pos  = float(np.mean(ep_pos))
        elapsed   = time.time() - t0
        cur_lr    = float(lr_schedule(optimizer.iterations))

        # ---- validation rank-1 ----
        rank1 = validate(model, train_data, val_data, image_size,
                         batch_size=batch_size)

        improved = rank1 > best_rank1
        if improved:
            best_rank1 = rank1
            model.save_weights(best_weights_path)
            no_improve_count = 0
            marker = " *"
        else:
            no_improve_count += 1
            marker = ""

        print(
            "Epoch {:4d}/{:d} | loss={:.4f} | active={:.2f} | "
            "pos_d={:.3f} | val_r1={:.4f}{} | "
            "lr={:.2e} | {:.1f}s".format(
                epoch, args.epochs, mean_loss, mean_frac,
                mean_pos, rank1, marker, cur_lr, elapsed
            )
        )

        log["epochs"].append(dict(
            epoch=epoch, loss=mean_loss, frac_active=mean_frac,
            mean_pos_dist=mean_pos, val_rank1=rank1, lr=cur_lr,
        ))
        with open(log_path, "w") as fh:
            json.dump(log, fh, indent=2)

        # ---- early stopping ----
        if no_improve_count >= args.patience:
            print(
                "\n[train] Early stopping after {:d} epochs without "
                "improvement (patience={:d})".format(
                    no_improve_count, args.patience
                )
            )
            break

    # ------------------------------------------------------------------
    # 6. Load best weights and export
    # ------------------------------------------------------------------
    print("\n[train] Best val rank-1: {:.4f}".format(best_rank1))
    print("[train] Loading best weights: {}".format(best_weights_path))
    model.load_weights(best_weights_path)

    # Save as .h5 alias
    final_keras = str(output_dir / "best_model_keras.h5")
    model.save_weights(final_keras)
    print("[train] Keras weights -> {}".format(final_keras))

    if not args.no_export_pb:
        export_frozen_pb(model, str(output_dir / "best_model.pb"), input_shape)

    return model


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
