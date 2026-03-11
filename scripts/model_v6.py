"""Turkey ReID embedding model — v6: 128-ch stage-4 + SE attention + horizontal strip pooling.

Combines the best of v4 and v5:
  v5 backbone  — expanded stage-4 channels (32 → 128) with SE channel attention
                 on every residual block gives tight, stable intra-class clusters.
  v4 pooling   — horizontal strip pooling preserves vertical spatial layout
                 (head / body / leg markings) instead of collapsing everything
                 with a single GAP.

Architecture
------------
  Stage 1 :  Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool(3×3, s2)
  Stage 2 :  ResBlock_SE(32) × 2 → BN → ReLU
  Stage 3 :  Conv(stage4_filters=128) → BN → ReLU → MaxPool(3×3, s2)
  Stage 4 :  ResBlock_SE(128) × 2 → BN → ReLU
  Head    :  StripPool(num_strips=4) → BNNeck → Dropout → Dense(512→128) → L2Norm

Concrete dimensions (defaults: 128×64 input, num_filters=32, stage4_filters=128,
                                num_strips=4):
  Feature map after stage 4 :  32(H) × 16(W) × 128(C)
  4 strips, each             :   8(H) × 16(W) × 128(C)  →  pool → 128-D
  Concatenated strip vector  :  4 × 128 = 512-D
  Dense projects DOWN        :  512 → 128 (no up-projection bottleneck)

Estimated parameters (num_filters=32, stage4_filters=128, se_reduction=4,
                       num_strips=4):
  Stage 1 : ~10k
  Stage 2 : ~39k  (2× SE-ResBlocks at 32ch + SE ~1k each)
  Stage 3 : ~37k  (Conv 32→128 + BN)
  Stage 4 : ~629k (2× SE-ResBlocks at 128ch + SE ~9k each)
  Head    :  ~67k  (BNNeck(512) + Dense(512→128))
  Total   : ~782k

Recommended training flags (see train_reid_v2.py --arch v6):
  --patience 50 --weight_decay 2e-4 --dropout 0.2 --P 20 --K 4 --epochs 300
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
# Pre-activation residual block + SE (from v5)
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
# Horizontal strip pooling (from v4)
# ---------------------------------------------------------------------------

def _strip_pool(x: tf.Tensor, num_strips: int, feat_h: int,
                name: str = "strips") -> tf.Tensor:
    """Pool N horizontal strips of the feature map and concatenate.

    Parameters
    ----------
    x          : (batch, H, W, C) feature tensor
    num_strips : number of horizontal bands to split into
    feat_h     : feature map height (must be known statically and divisible
                 by num_strips)
    name       : name prefix

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
        # Default-arg capture prevents Python closure bug
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
    dropout_rate: float = 0.2,
    num_classes: Optional[int] = None,
    weight_decay: float = 0.0,
    stage4_filters: Optional[int] = None,
    se_reduction: int = 4,
    num_strips: int = 4,
    name: str = "turkey_reid_v6",
) -> Model:
    """Build the v6 embedding model (128-ch stage-4 + SE + strip pooling).

    Parameters
    ----------
    input_shape    : (H, W, C)
    embedding_dim  : output embedding dimension (default 128)
    num_filters    : base channel count for stages 1-2
    dropout_rate   : dropout before the embedding Dense (default 0.2 for v6)
    num_classes    : if not None, add a classification head
    weight_decay   : L2 regularisation on Conv/Dense kernels (recommend 2e-4)
    stage4_filters : channels in stage-4 ResBlocks and stage-3 transition Conv.
                     Defaults to num_filters * 4 (= 128 for num_filters=32).
    se_reduction   : SE bottleneck ratio (filters // se_reduction hidden units)
    num_strips     : horizontal strips for the pooling head. Must divide
                     input_h // 4 (= 32 for default 128px input). Default 4.
    name           : Keras model name
    """
    if stage4_filters is None:
        stage4_filters = num_filters * 4   # 128 by default

    feat_h = input_shape[0] // 4           # 32 for 128px input

    if feat_h % num_strips != 0:
        raise ValueError(
            "input_h ({}) // 4 = {} is not divisible by num_strips ({}).".format(
                input_shape[0], feat_h, num_strips
            )
        )

    # Strip-pooled feature size before embedding Dense
    strip_feat_dim = stage4_filters * num_strips   # 128 * 4 = 512 by default

    reg = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None

    inp = layers.Input(shape=input_shape, name="images")

    # ---- internal normalisation ----
    x = layers.Lambda(lambda t: t / 255.0, name="normalize")(inp)

    # ---- Stage 1: two conv layers + max-pool, num_filters (32) channels ----
    x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same", name="pool1")(x)

    # ---- Stage 2: two pre-act SE residual blocks, num_filters (32) channels ----
    x = _residual_block(x, num_filters, "res1", reg, se_reduction)
    x = _residual_block(x, num_filters, "res2", reg, se_reduction)
    x = layers.BatchNormalization(name="stage2_final_bn")(x)
    x = layers.ReLU(name="stage2_final_relu")(x)

    # ---- Stage 3: transition conv + max-pool, stage4_filters (128) channels ----
    x = layers.Conv2D(stage4_filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same", name="pool2")(x)

    # ---- Stage 4: two pre-act SE residual blocks, stage4_filters (128) channels ----
    x = _residual_block(x, stage4_filters, "res3", reg, se_reduction)
    x = _residual_block(x, stage4_filters, "res4", reg, se_reduction)
    x = layers.BatchNormalization(name="stage4_final_bn")(x)
    x = layers.ReLU(name="stage4_final_relu")(x)

    # ---- Embedding head: strip pool + BNNeck + Dropout + Dense + L2Norm ----
    # 4 strips × 128ch = 512-D; Dense projects DOWN: 512 → 128
    x = _strip_pool(x, num_strips=num_strips, feat_h=feat_h, name="strip")
    x = layers.BatchNormalization(name="bnneck")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Dense(embedding_dim, use_bias=False,
                     kernel_regularizer=reg, name="embedding")(x)

    features = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1), name="features"
    )(x)

    if num_classes is not None:
        logits = layers.Dense(num_classes, name="logits")(x)
        return Model(inp, [features, logits], name=name)

    return Model(inp, features, name=name)
