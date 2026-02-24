"""DeepSORT-style compact CNN embedding model for turkey re-identification.

Design
------
* Input  : (H=128, W=64, C=3)  raw BGR float32 in [0, 255].
  Normalization is performed INSIDE the model (÷255 Lambda layer) so that
  the exported .pb is directly compatible with DeepSORT inference code that
  passes raw uint8/float image patches without pre-normalizing.

* Output : (D=128,) L2-normalised float32 embedding.
  Output node is named "features" in the frozen graph.

Architecture (mirrors the original DeepSORT residual feature extractor):
    normalize (÷255)
    Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool(3×3, s=2)
    ResBlock(32) × 2
    Conv(64) → BN → ReLU → MaxPool(3×3, s=2)
    ResBlock(64) × 2
    GlobalAveragePool
    Dropout
    Dense(128, no bias)
    L2Normalize  →  "features"
    [optional: Dense(num_classes) → "logits"]
"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model, layers


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

def _residual_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Post-activation identity residual block.

    conv→BN→ReLU→conv→BN→Add(shortcut)→ReLU
    Shortcut is the identity (no projection needed because filter count
    matches throughout each stage).
    """
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name="{}_conv1".format(name))(x)
    x = layers.BatchNormalization(name="{}_bn1".format(name))(x)
    x = layers.ReLU(name="{}_relu1".format(name))(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name="{}_conv2".format(name))(x)
    x = layers.BatchNormalization(name="{}_bn2".format(name))(x)
    x = layers.Add(name="{}_add".format(name))([shortcut, x])
    x = layers.ReLU(name="{}_out".format(name))(x)
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
    name: str = "turkey_reid",
) -> Model:
    """Build the DeepSORT-style embedding model.

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
    inp = layers.Input(shape=input_shape, name="images")

    # ---- internal normalisation: .pb accepts raw [0, 255] BGR ----
    x = layers.Lambda(lambda t: t / 255.0, name="normalize")(inp)

    # ---- Stage 1: two conv layers + max-pool, 32 filters ----
    x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False,
                      name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)

    x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False,
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same",
                            name="pool1")(x)

    # ---- Stage 2: two residual blocks, 32 filters ----
    x = _residual_block(x, num_filters, "res1")
    x = _residual_block(x, num_filters, "res2")

    # ---- Stage 3: conv + max-pool, 64 filters ----
    x = layers.Conv2D(num_filters * 2, 3, padding="same", use_bias=False,
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same",
                            name="pool2")(x)

    # ---- Stage 4: two residual blocks, 64 filters ----
    x = _residual_block(x, num_filters * 2, "res3")
    x = _residual_block(x, num_filters * 2, "res4")

    # ---- Embedding head ----
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Dense(embedding_dim, use_bias=False, name="embedding")(x)

    # L2-normalise; named "features" for DeepSORT compatibility
    features = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name="features",
    )(x)

    if num_classes is not None:
        logits = layers.Dense(num_classes, name="logits")(features)
        return Model(inp, [features, logits], name=name)

    return Model(inp, features, name=name)
