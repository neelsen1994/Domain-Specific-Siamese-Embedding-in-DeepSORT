"""Turkey ReID embedding model — v3: pre-activation ResBlocks + BNNeck + SE attention.

Changes from model_v2.py
-------------------------
Squeeze-and-Excite (SE) channel attention (Hu et al. 2018 CVPR) is added
after every residual block.

  SE block:
    block_output → GlobalAveragePool (squeeze)
                 → Dense(C // reduction, relu)  (excite bottleneck)
                 → Dense(C, sigmoid)             (excite gate)
                 → reshape (1, 1, C)
                 → multiply channel-wise

Placement: after the residual Add, before returning from each block.
This is the standard SE-ResNet arrangement and is compatible with
pre-activation (identity-mapping) residual connections.

Parameter overhead (reduction=4, default):
  Stage 2 blocks (32 ch): Dense(32→8→32) = 512 params × 2 = 1,024
  Stage 4 blocks (64 ch): Dense(64→16→64) = 2,048 params × 2 = 4,096
  Total SE overhead: ~5,120 params on top of v2's ~224k

All other aspects (input "images", output "features", raw BGR [0,255],
BNNeck, .pb compatibility) are identical to v2.
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
    """Squeeze-and-Excite channel attention (Hu et al. 2018).

    Parameters
    ----------
    x         : (batch, H, W, filters) feature map
    filters   : number of channels (must equal x.shape[-1])
    reduction : bottleneck ratio  (filters // reduction hidden units)
    name      : name prefix for all sub-layers
    """
    bottleneck = max(1, filters // reduction)
    # Squeeze: global context vector
    se = layers.GlobalAveragePooling2D(name="{}_squeeze".format(name))(x)
    # Excite: two fully-connected layers
    se = layers.Dense(bottleneck, use_bias=True, activation="relu",
                      name="{}_fc1".format(name))(se)
    se = layers.Dense(filters, use_bias=True, activation="sigmoid",
                      name="{}_fc2".format(name))(se)
    # Reshape for broadcast multiplication: (batch, C) -> (batch, 1, 1, C)
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
    x = _se_block(x, filters, reduction=se_reduction,
                  name="{}_se".format(name))
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
    se_reduction: int = 4,
    name: str = "turkey_reid_v3",
) -> Model:
    """Build the v3 embedding model (pre-act ResBlocks + BNNeck + SE).

    Parameters
    ----------
    input_shape  : (H, W, C) – must match DeepSORT patch size.
    embedding_dim: output embedding dimension (default 128).
    num_filters  : base channel count; doubles after the first MaxPool.
    dropout_rate : dropout probability before the embedding Dense.
    num_classes  : if not None, add a classification head → [features, logits].
    weight_decay : L2 regularisation on Conv/Dense kernels.
    se_reduction : SE bottleneck ratio (filters // se_reduction hidden units).
    name         : Keras model name.
    """
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

    # ---- Stage 3: conv + max-pool, 64 filters ----
    x = layers.Conv2D(num_filters * 2, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same", name="pool2")(x)

    # ---- Stage 4: two pre-act SE residual blocks, 64 filters ----
    x = _residual_block(x, num_filters * 2, "res3", reg, se_reduction)
    x = _residual_block(x, num_filters * 2, "res4", reg, se_reduction)
    x = layers.BatchNormalization(name="stage4_final_bn")(x)
    x = layers.ReLU(name="stage4_final_relu")(x)

    # ---- Embedding head with BNNeck ----
    x = layers.GlobalAveragePooling2D(name="gap")(x)
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
