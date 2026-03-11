#!/usr/bin/env python3
"""Turkey ReID Training Pipeline — v2 (multi-architecture)
===========================================================
Single training entry point for all model variants.  Select the architecture
with --arch {v2, v3, v4, v5, v6}.  All other hyper-parameters are shared.

Architecture summary
--------------------
  v2  (~224k)  Pre-activation ResBlocks + BNNeck                  (baseline)
  v3  (~229k)  v2 + Squeeze-and-Excite channel attention
  v4  (~249k)  v2 + horizontal strip pooling (fixes 64-D bottleneck)
  v5  (~714k)  v2 + stage-4 channels 64→128 + SE attention        (high capacity)
  v6  (~782k)  v5 + horizontal strip pooling                      (recommended)
               = 128-ch stage-4 + SE on all blocks + 4-strip pooling head

Changes from train_reid.py (present in all variants)
------------------------------------------------------
1. --arch flag selects which model_vX.py to use
2. --P default : 8 → 16  (stronger hard-negative mining)
3. Gradient clipping: Adam clipnorm=1.0  (stability)
4. Label smoothing ε=0.1 on the ID classification CE loss
5. New CLI flags: --arch, --label_smoothing, --clip_norm,
                  --se_reduction, --num_strips, --stage4_filters

Usage
-----
# v2 baseline
python scripts/train_reid_v2.py --arch v2 --dataset_path dataset_siam_21

# v3 (SE blocks)
python scripts/train_reid_v2.py --arch v3 --dataset_path dataset_siam_21

# v4 (strip pooling, 4 strips)
python scripts/train_reid_v2.py --arch v4 --dataset_path dataset_siam_21

# v4 with 2 strips (fewer but bigger bands)
python scripts/train_reid_v2.py --arch v4 --num_strips 2

# v5 (128-ch stage 4 + SE)
python scripts/train_reid_v2.py --arch v5 --dataset_path dataset_siam_21

# v6 (v5 backbone + strip pooling) — recommended flags for the bigger model:
python scripts/train_reid_v2.py --arch v6 --dataset_path dataset_siam_21 \\
    --epochs 300 --patience 50 \\
    --weight_decay 2e-4 --dropout 0.2 \\
    --P 20 --K 4

# Override output directory
python scripts/train_reid_v2.py --arch v3 --output_dir runs/ablation_v3

# Resume from checkpoint
python scripts/train_reid_v2.py --arch v3 \\
    --resume runs/turkey_reid_v3/checkpoints/best_weights.h5 --epochs 200

Output files (per arch, unless --output_dir is given)
------------------------------------------------------
runs/turkey_reid_{arch}/
    split_manifest.json          deterministic identity split
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
    create_id_split,
    load_split_from_manifest,
    PKSampler,
    load_split_as_arrays,
)
from scripts.losses import batch_hard_triplet_loss, batch_soft_triplet_loss
from scripts.augmentation_v2 import ReIDAugmentation
from scripts.metrics import compute_rank1


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------

_ARCH_CHOICES = ["v2", "v3", "v4", "v5", "v6"]


def _load_model_builder(arch: str):
    """Import and return build_embedding_model from the requested arch module."""
    if arch == "v2":
        from scripts.model_v2 import build_embedding_model
    elif arch == "v3":
        from scripts.model_v3 import build_embedding_model
    elif arch == "v4":
        from scripts.model_v4 import build_embedding_model
    elif arch == "v5":
        from scripts.model_v5 import build_embedding_model
    elif arch == "v6":
        from scripts.model_v6 import build_embedding_model
    else:
        raise ValueError("Unknown --arch '{}'. Choose from: {}".format(
            arch, _ARCH_CHOICES))
    return build_embedding_model


def _arch_kwargs(args: "argparse.Namespace") -> dict:
    """Collect arch-specific kwargs to pass to build_embedding_model."""
    if args.arch == "v3":
        return {"se_reduction": args.se_reduction}
    if args.arch == "v4":
        return {"num_strips": args.num_strips}
    if args.arch == "v5":
        return {
            "stage4_filters": args.stage4_filters,
            "se_reduction": args.se_reduction,
        }
    if args.arch == "v6":
        return {
            "stage4_filters": args.stage4_filters,
            "se_reduction": args.se_reduction,
            "num_strips": args.num_strips,
        }
    return {}   # v2: no extra kwargs

def _configure_gpu() -> None:
    """Enable GPU if available; set memory growth to avoid OOM on shared GPUs."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[gpu] {} GPU(s) available: {}".format(
                len(gpus), [g.name for g in gpus]))
        except RuntimeError as e:
            print("[gpu] Memory growth config failed: {}".format(e))
    else:
        print("[gpu] No GPU found — training on CPU.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train turkey ReID embedding model (multi-arch: v2/v3/v4/v5/v6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Architecture ---
    g = p.add_argument_group("Architecture")
    g.add_argument("--arch", default="v2", choices=_ARCH_CHOICES,
                   help="Model variant to train.  "
                        "v2=baseline  v3=+SE  v4=+strip-pool  v5=+128ch+SE  "
                        "v6=+128ch+SE+strip-pool (recommended: use with "
                        "--patience 50 --weight_decay 2e-4 --dropout 0.2 --P 20)")
    g.add_argument("--se_reduction", type=int, default=4,
                   help="[v3/v5] SE bottleneck ratio: filters // se_reduction "
                        "hidden units per SE block.")
    g.add_argument("--num_strips", type=int, default=4,
                   help="[v4] Number of horizontal strips for strip pooling. "
                        "Must divide input_h // 4 (= 32 for default 128px input).")
    g.add_argument("--stage4_filters", type=int, default=None,
                   help="[v5] Channel count for stage-4 ResBlocks and the "
                        "stage-3 transition Conv.  Default: num_filters * 4 = 128.")

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
    g.add_argument("--P", type=int, default=16,          # v2: was 8
                   help="Identities per batch.  v2 default=16 (was 8).")
    g.add_argument("--K", type=int, default=4,
                   help="Images per identity per batch.  batch_size = P * K.")
    g.add_argument("--steps_per_epoch", type=int, default=50,
                   help="Training steps per epoch (with replacement).")

    # --- Training ---
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",       type=int,   default=150)
    g.add_argument("--lr",           type=float, default=3e-4,
                   help="Peak learning rate (cosine decay after warmup).")
    g.add_argument("--weight_decay", type=float, default=1e-4,
                   help="L2 regularisation on Conv/Dense kernels.")
    g.add_argument("--margin",       type=float, default=0.50,
                   help="Triplet loss margin.")
    g.add_argument("--loss_type",    default="soft", choices=["hard", "soft"],
                   help="'soft' uses log(1+exp(·)) instead of hinge max(0,·).")
    g.add_argument("--id_loss_weight", type=float, default=0.5,
                   help="Weight of the cross-entropy ID classification loss. "
                        "Set to 0 to disable the classification head.")
    g.add_argument("--label_smoothing", type=float, default=0.1,  # v2 new
                   help="Label smoothing epsilon for the ID classification loss. "
                        "Prevents overconfidence on 3-8 images per class.")
    g.add_argument("--warmup_epochs", type=int,  default=5,
                   help="Linear LR warm-up duration in epochs.")
    g.add_argument("--patience",     type=int,   default=25,
                   help="Early-stopping patience in epochs.")
    g.add_argument("--clip_norm",    type=float, default=1.0,      # v2 new
                   help="Gradient clip norm for Adam.  0 = disabled.")
    g.add_argument("--no_augment",   action="store_true",
                   help="Disable data augmentation (for debugging).")
    g.add_argument("--resume",       default=None,
                   help="Path to .h5 weights to resume from.")

    # --- Output ---
    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", default=None,
                   help="Directory for checkpoints, logs, and exported .pb.  "
                        "Defaults to runs/turkey_reid_{arch} if omitted.")
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

        warmup_lr = peak * step / tf.maximum(warmup, 1.0)

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
    """
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    @tf.function(input_signature=[
        tf.TensorSpec([None] + list(input_shape), tf.float32, name="images")
    ])
    def _serving(x: tf.Tensor) -> tf.Tensor:
        out = model(x, training=False)
        emb = out[0] if isinstance(out, (list, tuple)) else out
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
        out = model(batch, training=False)
        emb = out[0] if isinstance(out, (list, tuple)) else out
        embs.append(emb.numpy())
    return np.concatenate(embs, axis=0)


def validate(
    model: tf.keras.Model,
    val_data: Dict[str, List[str]],
    image_size: Tuple[int, int],
    batch_size: int = 64,
) -> float:
    """Rank-1 accuracy on entirely unseen val identities (leave-one-out)."""
    imgs, lbls, _ = load_split_as_arrays(val_data, image_size)
    embs = extract_embeddings(model, imgs, batch_size)
    return compute_rank1(embs, lbls, embs, lbls, remove_self=True)


# ===========================================================================
# Main training loop
# ===========================================================================

def train(args: argparse.Namespace) -> tf.keras.Model:
    _configure_gpu()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Auto-derive output directory from arch if not explicitly set
    if args.output_dir is None:
        args.output_dir = "runs/turkey_reid_{}".format(args.arch)

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
        print("[train] Creating new identity split (open-set) ...")
        train_data, val_data, test_data = create_id_split(
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
    num_classes = len(train_data) if args.id_loss_weight > 0 else None
    build_fn = _load_model_builder(args.arch)
    model = build_fn(
        input_shape=input_shape,
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        dropout_rate=args.dropout,
        num_classes=num_classes,
        weight_decay=args.weight_decay,
        **_arch_kwargs(args),
    )
    model.summary(line_length=90)
    print("[train] Architecture:     {}".format(args.arch))
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

    # v2: gradient clipping via clipnorm (0 = disabled)
    if args.clip_norm > 0:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, clipnorm=args.clip_norm
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # v2: label-smoothed CE loss object (reused each step)
    if args.id_loss_weight > 0 and num_classes is not None and args.label_smoothing > 0:
        ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=args.label_smoothing,
        )
    else:
        ce_loss_fn = None

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    best_rank1        = -1.0
    best_weights_path = str(ckpt_dir / "best_weights.h5")
    no_improve_count  = 0

    log = {"args": vars(args), "epochs": []}
    log_path = str(output_dir / "training_log.json")

    id_loss_active = args.id_loss_weight > 0 and num_classes is not None
    triplet_fn = batch_soft_triplet_loss if args.loss_type == "soft" else batch_hard_triplet_loss

    # Build the arch-specific extra info line for the header
    arch_extra = ""
    if args.arch == "v3":
        arch_extra = "  SE reduction:    {}\n".format(args.se_reduction)
    elif args.arch == "v4":
        arch_extra = "  Strip count:     {}\n".format(args.num_strips)
    elif args.arch == "v5":
        s4f = args.stage4_filters if args.stage4_filters else args.num_filters * 4
        arch_extra = "  Stage-4 filters: {}  SE reduction: {}\n".format(
            s4f, args.se_reduction)
    elif args.arch == "v6":
        s4f = args.stage4_filters if args.stage4_filters else args.num_filters * 4
        arch_extra = "  Stage-4 filters: {}  SE reduction: {}  Strip count: {}\n".format(
            s4f, args.se_reduction, args.num_strips)

    header = (
        "\n" + "=" * 75 + "\n"
        "Training configuration  [arch={arch}  params={params:,}]\n"
        "  Output dir:      {outdir}\n"
        "  Epochs:          {epochs}\n"
        "  Steps/epoch:     {steps}\n"
        "  Batch (P×K):     {P}×{K}={bs}\n"
        "  Loss type:       {lt}\n"
        "  Triplet margin:  {margin}\n"
        "  ID loss weight:  {idw} ({nc} classes)\n"
        "  Label smoothing: {ls}\n"
        "  Weight decay:    {wd}\n"
        "  Peak LR:         {lr}\n"
        "  Warmup epochs:   {wu}\n"
        "  Clip norm:       {cn}\n"
        "  Early-stop pat.: {pat}\n"
        "  Augmentation:    {aug}\n"
        "{arch_extra}" + "=" * 75
    ).format(
        arch=args.arch,
        params=model.count_params(),
        outdir=args.output_dir,
        epochs=args.epochs,
        steps=args.steps_per_epoch,
        P=args.P, K=args.K, bs=batch_size,
        lt=args.loss_type,
        margin=args.margin,
        idw=args.id_loss_weight,
        nc=num_classes if num_classes else 0,
        ls=args.label_smoothing,
        wd=args.weight_decay,
        lr=args.lr,
        wu=args.warmup_epochs,
        cn=args.clip_norm if args.clip_norm > 0 else "off",
        pat=args.patience,
        aug="off" if args.no_augment else "on (v2)",
        arch_extra=arch_extra,
    )
    print(header)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        ep_loss, ep_frac, ep_pos = [], [], []

        for images, labels, global_labels in sampler.epoch_generator(args.steps_per_epoch):
            images_tf        = tf.constant(images,        dtype=tf.float32)
            labels_tf        = tf.constant(labels,        dtype=tf.int32)
            global_labels_tf = tf.constant(global_labels, dtype=tf.int32)

            with tf.GradientTape() as tape:
                model_out = model(images_tf, training=True)

                if id_loss_active:
                    embs, logits = model_out
                else:
                    embs = model_out

                trip_loss, frac_active, mean_pos = triplet_fn(
                    embs, labels_tf, margin=args.margin
                )

                if id_loss_active:
                    if ce_loss_fn is not None:
                        # v2: label-smoothed CE via CategoricalCrossentropy
                        one_hot = tf.one_hot(global_labels_tf, num_classes)
                        ce_loss = ce_loss_fn(one_hot, logits)
                    else:
                        ce_loss = tf.reduce_mean(
                            tf.keras.losses.sparse_categorical_crossentropy(
                                global_labels_tf, logits, from_logits=True
                            )
                        )
                    loss = trip_loss + args.id_loss_weight * ce_loss
                else:
                    loss = trip_loss

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
        rank1 = validate(model, val_data, image_size, batch_size=batch_size)

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
