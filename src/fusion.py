"""Rank fusion for systems C and D (and the ablation E), plus score-level convex combination (post-hoc).

RRF (Cormack et al., 2009): fused(d) = sum over systems s of 1 / (k + rank_s(d)), rank 1-based.
Ranks use the same order as the evaluator (score descending, ties broken by page id descending), so a tie
inside one system is resolved exactly as when that system is evaluated alone.
"""

from __future__ import annotations

import numpy as np


def ranks_from_scores(scores: np.ndarray, page_ids: list[str]) -> np.ndarray:
    """[n_queries, n_pages] scores -> 1-based ranks in trec_eval order (score desc, page id desc)."""
    n_q, n_p = scores.shape
    # lexsort sorts ascending by the last key first: primary = -score, secondary = -page-id order.
    id_order = np.argsort(np.argsort(np.array(page_ids, dtype=object)))  # position of each id in ascending id order
    ranks = np.empty((n_q, n_p), dtype=np.int64)
    for i in range(n_q):
        order = np.lexsort((-id_order, -scores[i]))
        ranks[i, order] = np.arange(1, n_p + 1)
    return ranks


def rrf(score_matrices: list[np.ndarray], page_ids: list[str], k: float = 60.0) -> np.ndarray:
    """Reciprocal rank fusion of several systems' full score matrices (same queries x pages)."""
    if not score_matrices:
        raise ValueError("need at least one system")
    fused = np.zeros_like(score_matrices[0], dtype=np.float64)
    for scores in score_matrices:
        fused += 1.0 / (k + ranks_from_scores(scores, page_ids))
    return fused


def weighted_rrf(score_matrices: list[np.ndarray], weights: list[float], page_ids: list[str], k: float = 60.0) -> np.ndarray:
    """Weighted RRF: fused(d) = sum_s w_s / (k + rank_s(d)). Equal weights give plain RRF (up to a constant factor)."""
    if len(score_matrices) != len(weights):
        raise ValueError("one weight per system")
    fused = np.zeros_like(score_matrices[0], dtype=np.float64)
    for scores, w in zip(score_matrices, weights):
        fused += w / (k + ranks_from_scores(scores, page_ids))
    return fused


def minmax_per_query(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise each query's scores over all pages to [0, 1]; a constant row becomes all zeros."""
    lo = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - lo
    return np.where(span > 0, (scores - lo) / np.where(span > 0, span, 1.0), 0.0)


def convex_combination(colpali: np.ndarray, text: np.ndarray, alpha: float) -> np.ndarray:
    """Score-level fusion (post-hoc, after Bruch et al. 2024): alpha * mm(ColPali) + (1 - alpha) * mm(text).

    alpha = 1 keeps ColPali's ranking exactly (min-max is order-preserving; ties are broken by page id as usual).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * minmax_per_query(colpali) + (1.0 - alpha) * minmax_per_query(text)
