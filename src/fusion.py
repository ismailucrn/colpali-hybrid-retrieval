"""Rank fusion for systems C and D (and the ablation E).

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
