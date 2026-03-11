"""DeepSORT-style compact CNN embedding model for turkey re-identification — v2.

Changes from model.py
---------------------
1. Pre-activation residual blocks (He et al. 2016 "Identity Mappings in Deep
   Residual Networks"):
     v1 (post-act): conv → BN → ReLU → conv → BN → Add → ReLU
     v2 (pre-act) : BN → ReLU → conv → BN → ReLU → conv → Add
   A final BN→ReLU is added after each residual stage so the output is always
   activated before the next stage's convolution.

2. BNNeck (Luo et al. 2019 "Bag of Tricks for Person Re-Identification"):
   A BatchNorm layer is inserted between GlobalAveragePooling and the Dropout
   before the embedding Dense.  This normalises the scale of the GAP feature
   before projection and is a consistent rank-1 improver in ReID literature.

All other aspects (input node "images", output node "features", raw BGR [0,255]
normalisation inside the model, .pb compatibility) are identical to v1.
"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model, layers


# ---------------------------------------------------------------------------
# Building block  (pre-activation)
# ---------------------------------------------------------------------------

def _residual_block(x: tf.Tensor, filters: int, name: str, reg=None) -> tf.Tensor:
    """Pre-activation identity residual block (He et al. 2016 v2).

    BN → ReLU → conv → BN → ReLU → conv → Add(shortcut)

    The output is the raw sum (no trailing ReLU).  Callers must add an
    explicit BN→ReLU after the last block in a stage before feeding into
    the next non-residual layer.
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
# Model factory
# ---------------------------------------------------------------------------

def build_embedding_model(
    input_shape: tuple = (128, 64, 3),
    embedding_dim: int = 128,
    num_filters: int = 32,
    dropout_rate: float = 0.3,
    num_classes: Optional[int] = None,
    weight_decay: float = 0.0,
    name: str = "turkey_reid_v2",
) -> Model:
    """Build the v2 embedding model.

    Parameters
    ----------
    input_shape   : (H, W, C) – must match your DeepSORT patch size.
    embedding_dim : dimensionality of the output embedding (default 128).
    num_filters   : base channel count; doubles after the first MaxPool.
    dropout_rate  : dropout probability before the embedding Dense layer.
    num_classes   : if not None, add a classification head and return
                    [features, logits].  Used for softmax-baseline runs.
    name          : Keras model name.

    Returns
    -------
    tf.keras.Model with:
        input  node "images"   – raw BGR float32 [0, 255]
        output node "features" – L2-normalised (D,) embedding
        (optionally also "logits" when num_classes is set)
    """
    reg = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None

    inp = layers.Input(shape=input_shape, name="images")

    # ---- internal normalisation: .pb accepts raw [0, 255] BGR ----
    x = layers.Lambda(lambda t: t / 255.0, name="normalize")(inp)

    # ---- Stage 1: two conv layers + max-pool, 32 filters ----
    # (standard post-activation; these are not residual blocks)
    x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)

    x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same",
                            name="pool1")(x)

    # ---- Stage 2: two pre-activation residual blocks, 32 filters ----
    x = _residual_block(x, num_filters, "res1", reg)
    x = _residual_block(x, num_filters, "res2", reg)
    # Final activation for stage 2 (pre-act block output is unactivated)
    x = layers.BatchNormalization(name="stage2_final_bn")(x)
    x = layers.ReLU(name="stage2_final_relu")(x)

    # ---- Stage 3: conv + max-pool, 64 filters ----
    x = layers.Conv2D(num_filters * 2, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same",
                            name="pool2")(x)

    # ---- Stage 4: two pre-activation residual blocks, 64 filters ----
    x = _residual_block(x, num_filters * 2, "res3", reg)
    x = _residual_block(x, num_filters * 2, "res4", reg)
    # Final activation for stage 4
    x = layers.BatchNormalization(name="stage4_final_bn")(x)
    x = layers.ReLU(name="stage4_final_relu")(x)

    # ---- Embedding head with BNNeck ----
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    # BNNeck: normalises GAP output scale before the embedding projection
    x = layers.BatchNormalization(name="bnneck")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Dense(embedding_dim, use_bias=False,
                     kernel_regularizer=reg, name="embedding")(x)

    # L2-normalise; named "features" for DeepSORT compatibility
    features = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name="features",
    )(x)

    if num_classes is not None:
        # Classification logits from pre-normalised embedding (more capacity)
        logits = layers.Dense(num_classes, name="logits")(x)
        return Model(inp, [features, logits], name=name)

    return Model(inp, features, name=name)
