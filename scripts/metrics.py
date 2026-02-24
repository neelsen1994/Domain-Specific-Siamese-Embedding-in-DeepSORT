"""ReID evaluation metrics: CMC (rank-k accuracy) and mean Average Precision.

All functions are pure numpy — no TensorFlow dependency.
Embeddings are expected to be L2-normalised; cosine similarity is then
equivalent to the dot product.

Typical usage
-------------
    cmc, mAP = compute_cmc_map(
        query_embs, query_labels,
        gallery_embs, gallery_labels,
        ranks=(1, 5, 10),
        remove_self=True,
    )
    print(f"Rank-1: {cmc['rank1']:.4f}  mAP: {mAP:.4f}")
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def cosine_sim_matrix(
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """(Q × G) cosine similarity matrix for L2-normalised embeddings."""
    return a @ b.T   # dot product equals cosine sim when norms = 1


def compute_cmc_map(
    query_embs: np.ndarray,
    query_labels: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
    ranks: Tuple[int, ...] = (1, 5, 10),
    remove_self: bool = True,
) -> Tuple[Dict[str, float], float]:
    """Compute CMC and mAP for ReID evaluation.

    Parameters
    ----------
    query_embs    : (Q, D) float32  L2-normalised embeddings for queries
    query_labels  : (Q,)   int      identity label for each query
    gallery_embs  : (G, D) float32  L2-normalised embeddings for gallery
    gallery_labels: (G,)   int      identity label for each gallery item
    ranks         : CMC rank cutoffs to report
    remove_self   : if True, remove the query from the gallery result list
                    when the query's own embedding is inside the gallery
                    (detected by cosine sim ≈ 1.0, i.e. > 1 - 1e-5)

    Returns
    -------
    cmc : dict {"rank1": float, "rank5": float, ...}
    mAP : float
    """
    Q = len(query_embs)
    sim = cosine_sim_matrix(query_embs, gallery_embs)   # (Q, G)
    sorted_idx = np.argsort(-sim, axis=1)               # descending sim

    all_ap: List[float] = []
    hits: Dict[int, int] = {k: 0 for k in ranks}

    for q in range(Q):
        ql    = query_labels[q]
        order = sorted_idx[q]                           # gallery indices sorted
        gl    = gallery_labels[order]                   # sorted gallery labels

        # Remove the query itself (exact cosine-sim match)
        if remove_self:
            keep  = sim[q, order] < (1.0 - 1e-5)
            order = order[keep]
            gl    = gallery_labels[order]

        matches = (gl == ql)                            # boolean match array

        # CMC: rank-k hit if any correct match in top-k
        for k in ranks:
            if matches[:k].any():
                hits[k] += 1

        # Average Precision
        n_rel = int(matches.sum())
        if n_rel == 0:
            all_ap.append(0.0)
            continue
        cumsum  = np.cumsum(matches).astype(np.float64)
        prec_at = cumsum / (np.arange(len(matches)) + 1.0)
        ap      = float((prec_at * matches).sum() / n_rel)
        all_ap.append(ap)

    cmc = {"rank{}".format(k): hits[k] / Q for k in ranks}
    mAP = float(np.mean(all_ap))
    return cmc, mAP


def compute_rank1(
    query_embs: np.ndarray,
    query_labels: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
    remove_self: bool = True,
) -> float:
    """Convenience wrapper: return only rank-1 accuracy."""
    cmc, _ = compute_cmc_map(
        query_embs, query_labels,
        gallery_embs, gallery_labels,
        ranks=(1,),
        remove_self=remove_self,
    )
    return cmc["rank1"]


def mean_intra_inter_distance(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Diagnostic: mean intra-class and inter-class cosine distances.

    For a well-trained model, intra < inter.
    """
    sim = cosine_sim_matrix(embeddings, embeddings)   # (N, N)
    # cosine distance = 1 - cosine similarity
    dist = 1.0 - sim

    N = len(labels)
    intra, inter = [], []
    for i in range(N):
        for j in range(i + 1, N):
            d = dist[i, j]
            if labels[i] == labels[j]:
                intra.append(d)
            else:
                inter.append(d)

    mean_intra = float(np.mean(intra)) if intra else float("nan")
    mean_inter = float(np.mean(inter)) if inter else float("nan")
    return mean_intra, mean_inter
