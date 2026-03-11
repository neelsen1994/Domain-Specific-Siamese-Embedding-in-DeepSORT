"""Turkey ReID embedding model — v5: expanded stage-4 channels + SE attention.

Changes from model_v2.py
-------------------------
1. Stage-4 channels: 64 → stage4_filters (default 128, i.e. num_filters * 4).
   The stage-3 transition Conv is updated to output stage4_filters channels.
   This directly targets the most parameter-starved part of the network
   (stage 4 represents 66% of total params but only at 64 channels).

2. SE channel attention (same as v3) added to all four residual blocks.
   Parameters per block scale with channel count:
     Stage 2 (32 ch, reduction=4):  Dense(32→8→32)  =    512 params
     Stage 4 (128 ch, reduction=4): Dense(128→32→128) = 8,704 params

Estimated parameters (num_filters=32, stage4_filters=128, se_reduction=4):
  Stage 1:  ~10k   (unchanged)
  Stage 2:  ~38k   (unchanged + SE ~1k)
  Stage 3:  ~37k   (Conv 32→128, was 32→64)
  Stage 4: ~612k   (ResBlocks at 128ch × 2 + SE × 2 + final BN)
  Head:     ~17k   (BNNeck 128ch + Dense 128→128)
  Total:   ~714k   (vs ~224k v2 / ~2.8M mars-small128)

No strip pooling in this variant (GAP retained) — the ablation is cleaner.
Combine with v4's strip pooling manually if desired.
"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model, layers


# ---------------------------------------------------------------------------
# SE channel attention block
# ---------------------------------------------------------------------------

def _se_block(x: tf.Tensor, filters: int, reduction: int = 4,
              name: str = "se") -> tf.Tensor:
    """Squeeze-and-Excite channel attention (Hu et al. 2018)."""
    bottleneck = max(1, filters // reduction)
    se = layers.GlobalAveragePooling2D(name="{}_squeeze".format(name))(x)
    se = layers.Dense(bottleneck, use_bias=True, activation="relu",
                      name="{}_fc1".format(name))(se)
    se = layers.Dense(filters, use_bias=True, activation="sigmoid",
                      name="{}_fc2".format(name))(se)
    se = layers.Reshape((1, 1, filters), name="{}_reshape".format(name))(se)
    return layers.Multiply(name="{}_scale".format(name))([x, se])


# ---------------------------------------------------------------------------
# Pre-activation residual block with SE
# ---------------------------------------------------------------------------

def _residual_block(x: tf.Tensor, filters: int, name: str,
                    reg=None, se_reduction: int = 4) -> tf.Tensor:
    """Pre-activation residual block + SE channel attention.

    BN → ReLU → conv → BN → ReLU → conv → Add(shortcut) → SE
    """
    shortcut = x
    x = layers.BatchNormalization(name="{}_bn1".format(name))(x)
    x = layers.ReLU(name="{}_relu1".format(name))(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="{}_conv1".format(name))(x)
    x = layers.BatchNormalization(name="{}_bn2".format(name))(x)
    x = layers.ReLU(name="{}_relu2".format(name))(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="{}_conv2".format(name))(x)
    x = layers.Add(name="{}_add".format(name))([shortcut, x])
    x = _se_block(x, filters, reduction=se_reduction, name="{}_se".format(name))
    return x


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_embedding_model(
    input_shape: tuple = (128, 64, 3),
    embedding_dim: int = 128,
    num_filters: int = 32,
    dropout_rate: float = 0.3,
    num_classes: Optional[int] = None,
    weight_decay: float = 0.0,
    stage4_filters: Optional[int] = None,
    se_reduction: int = 4,
    name: str = "turkey_reid_v5",
) -> Model:
    """Build the v5 embedding model (expanded stage-4 + SE).

    Parameters
    ----------
    input_shape    : (H, W, C)
    embedding_dim  : output embedding dimension (default 128)
    num_filters    : base channel count; stage 2 uses num_filters,
                     stage 4 uses stage4_filters.
    dropout_rate   : dropout before the embedding Dense
    num_classes    : if not None, add a classification head
    weight_decay   : L2 regularisation on Conv/Dense kernels
    stage4_filters : channels in stage-4 ResBlocks and the stage-3 transition
                     Conv.  Defaults to num_filters * 4 (= 128 for 32).
    se_reduction   : SE bottleneck ratio (filters // se_reduction hidden units)
    name           : Keras model name
    """
    if stage4_filters is None:
        stage4_filters = num_filters * 4   # 128 for default num_filters=32

    reg = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None

    inp = layers.Input(shape=input_shape, name="images")

    # ---- internal normalisation ----
    x = layers.Lambda(lambda t: t / 255.0, name="normalize")(inp)

    # ---- Stage 1: two conv layers + max-pool, 32 filters ----
    x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same", name="pool1")(x)

    # ---- Stage 2: two pre-act SE residual blocks, 32 filters ----
    x = _residual_block(x, num_filters, "res1", reg, se_reduction)
    x = _residual_block(x, num_filters, "res2", reg, se_reduction)
    x = layers.BatchNormalization(name="stage2_final_bn")(x)
    x = layers.ReLU(name="stage2_final_relu")(x)

    # ---- Stage 3: conv + max-pool, stage4_filters (128) ----
    # Transition conv now outputs stage4_filters instead of num_filters*2 (64)
    x = layers.Conv2D(stage4_filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same", name="pool2")(x)

    # ---- Stage 4: two pre-act SE residual blocks, stage4_filters (128) ----
    x = _residual_block(x, stage4_filters, "res3", reg, se_reduction)
    x = _residual_block(x, stage4_filters, "res4", reg, se_reduction)
    x = layers.BatchNormalization(name="stage4_final_bn")(x)
    x = layers.ReLU(name="stage4_final_relu")(x)

    # ---- Embedding head with BNNeck ----
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bnneck")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    # Dense projects down: stage4_filters (128) → embedding_dim (128)
    x = layers.Dense(embedding_dim, use_bias=False,
                     kernel_regularizer=reg, name="embedding")(x)

    features = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1), name="features"
    )(x)

    if num_classes is not None:
        logits = layers.Dense(num_classes, name="logits")(x)
        return Model(inp, [features, logits], name=name)

    return Model(inp, features, name=name)
