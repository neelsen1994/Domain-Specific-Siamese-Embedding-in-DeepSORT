"""Batch-hard triplet loss for metric-learning ReID training.

Reference: "In Defense of the Triplet Loss for Person Re-Identification"
           Hermans, Beyer, Leibe (2017)  arXiv:1703.07737
"""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def pairwise_l2_distance(
    embeddings: tf.Tensor,
    squared: bool = False,
) -> tf.Tensor:
    """Compute the (N × N) pairwise L2 distance matrix.

    Parameters
    ----------
    embeddings : (N, D) float32 tensor (need not be L2-normalised)
    squared    : if True return squared distances (faster, avoids sqrt)

    Returns
    -------
    dist : (N, N) float32 — dist[i, j] = ||emb_i - emb_j||₂
    """
    # ||a - b||² = ||a||² - 2·aᵀb + ||b||²
    dot = tf.matmul(embeddings, tf.transpose(embeddings))   # (N, N)
    sq  = tf.linalg.diag_part(dot)                          # (N,)
    d2  = (
        tf.expand_dims(sq, 1)       # (N, 1)
        - 2.0 * dot                 # (N, N)
        + tf.expand_dims(sq, 0)     # (1, N)
    )
    d2 = tf.maximum(d2, 0.0)        # numerical clamp (avoid tiny negatives)
    if squared:
        return d2
    return tf.sqrt(d2 + 1e-12)      # epsilon avoids NaN gradient at 0


def batch_hard_triplet_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    margin: float = 0.3,
    squared: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Batch-hard triplet loss.

    For every anchor i in the batch:
        hardest positive = the same-class sample j≠i with the LARGEST distance
        hardest negative = the different-class sample j with the SMALLEST distance
        loss_i = max(d_pos - d_neg + margin, 0)

    The loss is averaged over anchors that have at least one valid positive
    partner (i.e. at least one other sample with the same label in the batch).

    Parameters
    ----------
    embeddings : (N, D) float32 — model outputs for the current batch
    labels     : (N,)   int32  — identity labels (consecutive 0..P-1)
    margin     : triplet margin δ (default 0.3)
    squared    : use squared L2 distances instead of L2

    Returns
    -------
    loss        : scalar float32 — mean batch-hard triplet loss
    frac_active : scalar float32 — fraction of non-zero (active) triplets
    mean_pos    : scalar float32 — mean hardest-positive distance (monitoring)
    """
    dist = pairwise_l2_distance(embeddings, squared=squared)   # (N, N)

    labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    N = tf.shape(embeddings)[0]

    labels_row = tf.reshape(labels, [1, -1])                   # (1, N)
    labels_col = tf.reshape(labels, [-1, 1])                   # (N, 1)

    pos_mask = tf.equal(labels_col, labels_row)                # (N, N) same class
    neg_mask = tf.logical_not(pos_mask)                        # (N, N) diff class
    # Remove diagonal: an anchor is not its own positive
    eye = tf.eye(N, dtype=tf.bool)
    valid_pos_mask = tf.logical_and(pos_mask, tf.logical_not(eye))

    # ---- Hardest positive: max distance among valid positives ----
    # Invalid entries zeroed out (max over a set of non-negatives is safe)
    hardest_pos = tf.reduce_max(
        dist * tf.cast(valid_pos_mask, tf.float32), axis=1
    )   # (N,)

    # ---- Hardest negative: min distance among negatives ----
    # Replace non-negatives with a large sentinel so they are never chosen
    large = tf.ones_like(dist) * 1e9
    neg_dist = tf.where(neg_mask, dist, large)
    hardest_neg = tf.reduce_min(neg_dist, axis=1)              # (N,)

    # ---- Per-anchor triplet loss ----
    per_anchor = tf.maximum(hardest_pos - hardest_neg + margin, 0.0)   # (N,)

    # Only count anchors that actually have a valid positive in this batch
    has_pos = tf.cast(tf.reduce_any(valid_pos_mask, axis=1), tf.float32)  # (N,)
    denom   = tf.reduce_sum(has_pos) + 1e-9

    loss        = tf.reduce_sum(per_anchor * has_pos) / denom
    frac_active = tf.reduce_sum(
        tf.cast(per_anchor > 0.0, tf.float32) * has_pos
    ) / denom
    mean_pos    = tf.reduce_sum(hardest_pos * has_pos) / denom

    return loss, frac_active, mean_pos


def batch_soft_triplet_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    margin: float = 0.5,
    squared: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Soft-margin batch-hard triplet loss.

    Uses the same hardest-positive / hardest-negative mining as batch-hard,
    but replaces the hinge  max(0, d_pos - d_neg + margin)  with the smooth
    soft-plus  log(1 + exp(d_pos - d_neg + margin)).

    Benefits over hard-margin:
    * Always provides non-zero gradients — avoids the "dead zone" where the
      model gets no signal once all triplets satisfy the margin.
    * Much harder to collapse: even when d_pos ≈ d_neg ≈ 0 the loss is
      log(1 + exp(margin)) > 0 with a non-zero gradient that pushes the
      model to actually separate the embeddings.

    Parameters / Returns: same as batch_hard_triplet_loss.
    """
    dist = pairwise_l2_distance(embeddings, squared=squared)

    labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    N = tf.shape(embeddings)[0]

    labels_row = tf.reshape(labels, [1, -1])
    labels_col = tf.reshape(labels, [-1, 1])

    pos_mask = tf.equal(labels_col, labels_row)
    neg_mask = tf.logical_not(pos_mask)
    eye = tf.eye(N, dtype=tf.bool)
    valid_pos_mask = tf.logical_and(pos_mask, tf.logical_not(eye))

    hardest_pos = tf.reduce_max(
        dist * tf.cast(valid_pos_mask, tf.float32), axis=1
    )

    large = tf.ones_like(dist) * 1e9
    neg_dist = tf.where(neg_mask, dist, large)
    hardest_neg = tf.reduce_min(neg_dist, axis=1)

    # Soft-plus instead of hinge — always differentiable
    per_anchor = tf.math.softplus(hardest_pos - hardest_neg + margin)

    has_pos = tf.cast(tf.reduce_any(valid_pos_mask, axis=1), tf.float32)
    denom = tf.reduce_sum(has_pos) + 1e-9

    loss = tf.reduce_sum(per_anchor * has_pos) / denom
    # "active" fraction: triplets where hard-margin would also be > 0
    frac_active = tf.reduce_sum(
        tf.cast((hardest_pos - hardest_neg + margin) > 0.0, tf.float32) * has_pos
    ) / denom
    mean_pos = tf.reduce_sum(hardest_pos * has_pos) / denom

    return loss, frac_active, mean_pos
