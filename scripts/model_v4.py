"""Turkey ReID embedding model — v4: pre-activation ResBlocks + BNNeck + horizontal strip pooling.

Changes from model_v2.py
-------------------------
GlobalAveragePooling2D is replaced with horizontal strip pooling.

Motivation
----------
For a full-body vertical crop (128×64), GAP treats a pixel at the turkey's
head exactly the same as one at its legs.  Different individuals may have
distinctive markings at different vertical positions (wattle at the top,
body feather patterns in the middle, leg colouring at the bottom).  GAP
erases all of this vertical spatial information.

Horizontal strip pooling divides the final feature map into N equal-height
horizontal bands, applies average pooling within each band, and concatenates
the N pooled vectors.  This preserves the vertical layout at negligible cost.

Concrete dimensions (default: 128×64 input, num_filters=32, num_strips=4):
  Feature map after stage 4:  32 (H) × 16 (W) × 64 (C)
  4 strips, each:              8  (H) × 16 (W) × 64 (C) → pool → 64-D
  Concatenated:                4 × 64 = 256-D
  Dense(256 → 128, no bias) → L2Norm

This also fixes the 64-D bottleneck in v2/v3 where GAP produced only a
64-D vector that was then *up-projected* to 128-D by the embedding Dense.
Now the Dense projects down (256 → 128), which is more conventional.

Parameter overhead vs v2:
  Dense input: 64 → 256  (+24,576 extra params in the embedding Dense)
  Net total:   ~249k   (v2 was ~224k)

No SE blocks in this variant — keeping it focused so the ablation is clean.
"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model, layers


# ---------------------------------------------------------------------------
# Pre-activation residual block (same as v2 — no SE in this variant)
# ---------------------------------------------------------------------------

def _residual_block(x: tf.Tensor, filters: int, name: str,
                    reg=None) -> tf.Tensor:
    """Pre-activation identity residual block.

    BN → ReLU → conv → BN → ReLU → conv → Add(shortcut)
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
    return x


# ---------------------------------------------------------------------------
# Horizontal strip pooling
# ---------------------------------------------------------------------------

def _strip_pool(x: tf.Tensor, num_strips: int, feat_h: int,
                name: str = "strips") -> tf.Tensor:
    """Pool N horizontal strips of the feature map and concatenate.

    Parameters
    ----------
    x         : (batch, H, W, C) feature tensor
    num_strips: number of horizontal bands to split into
    feat_h    : feature map height (must be known statically and divisible
                by num_strips)
    name      : name prefix

    Returns
    -------
    (batch, num_strips * C) concatenated pooled vectors
    """
    if feat_h % num_strips != 0:
        raise ValueError(
            "Feature map height {} is not divisible by num_strips {}.  "
            "Adjust --num_strips or --image_h.".format(feat_h, num_strips)
        )
    strip_h = feat_h // num_strips
    pooled = []
    for i in range(num_strips):
        # Default-arg capture (i=i, sh=strip_h) prevents Python closure bug
        strip = layers.Lambda(
            lambda t, s=i, sh=strip_h: t[:, s * sh: (s + 1) * sh, :, :],
            name="{}_slice{}".format(name, i),
        )(x)
        p = layers.GlobalAveragePooling2D(name="{}_gap{}".format(name, i))(strip)
        pooled.append(p)
    return layers.Concatenate(name="{}_concat".format(name))(pooled)


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
    num_strips: int = 4,
    name: str = "turkey_reid_v4",
) -> Model:
    """Build the v4 embedding model (pre-act ResBlocks + BNNeck + strip pool).

    Parameters
    ----------
    input_shape  : (H, W, C)
    embedding_dim: output embedding dimension (default 128)
    num_filters  : base channel count; doubles after the first MaxPool
    dropout_rate : dropout probability before the embedding Dense
    num_classes  : if not None, add a classification head
    weight_decay : L2 regularisation on Conv/Dense kernels
    num_strips   : number of horizontal strips for the pooling head.
                   Must divide the feature-map height (input_h // 4).
                   Default 4  →  (32×16×64) → 4 × 64 = 256-D.
    name         : Keras model name
    """
    reg = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None

    # Feature map height after two stride-2 MaxPools
    feat_h = input_shape[0] // 4

    if feat_h % num_strips != 0:
        raise ValueError(
            "input_h ({}) // 4 = {} is not divisible by num_strips ({}).".format(
                input_shape[0], feat_h, num_strips
            )
        )

    # Strip-pooled feature size before embedding Dense
    strip_feat_dim = num_filters * 2 * num_strips   # 64 * 4 = 256 by default

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

    # ---- Stage 2: two pre-act residual blocks, 32 filters ----
    x = _residual_block(x, num_filters, "res1", reg)
    x = _residual_block(x, num_filters, "res2", reg)
    x = layers.BatchNormalization(name="stage2_final_bn")(x)
    x = layers.ReLU(name="stage2_final_relu")(x)

    # ---- Stage 3: conv + max-pool, 64 filters ----
    x = layers.Conv2D(num_filters * 2, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same", name="pool2")(x)

    # ---- Stage 4: two pre-act residual blocks, 64 filters ----
    x = _residual_block(x, num_filters * 2, "res3", reg)
    x = _residual_block(x, num_filters * 2, "res4", reg)
    x = layers.BatchNormalization(name="stage4_final_bn")(x)
    x = layers.ReLU(name="stage4_final_relu")(x)

    # ---- Embedding head: horizontal strip pooling + BNNeck ----
    # Strip pool replaces GAP; fixes the 64-D up-projection bottleneck in v2
    x = _strip_pool(x, num_strips=num_strips, feat_h=feat_h, name="strip")

    # BNNeck on the concatenated strip features
    x = layers.BatchNormalization(name="bnneck")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    # Dense projects DOWN: strip_feat_dim (256) → embedding_dim (128)
    x = layers.Dense(embedding_dim, use_bias=False,
                     kernel_regularizer=reg, name="embedding")(x)

    features = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1), name="features"
    )(x)

    if num_classes is not None:
        logits = layers.Dense(num_classes, name="logits")(x)
        return Model(inp, [features, logits], name=name)

    return Model(inp, features, name=name)
