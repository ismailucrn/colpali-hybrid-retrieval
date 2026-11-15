"""Two-stage retrieval (post-hoc set 2): rerank the top N of a first-stage ranking with a second-stage scorer.

The output is a full [n_queries, n_pages] score matrix, so the usual evaluator and statistics apply unchanged:
- the first stage's top N pages (in trec_eval order: score desc, page id desc) are reordered by the second-stage
  score (ties keep the first-stage order);
- every page below N keeps its first-stage position.
Scores are -(final position), so they are distinct and the evaluator's tie-breaking never matters.
"""

from __future__ import annotations

import numpy as np

from src.fusion import ranks_from_scores


def rerank_top_n(first: np.ndarray, second: np.ndarray, n: int, page_ids: list[str]) -> np.ndarray:
    """Rerank the first stage's top n pages per query by `second` (higher = better).

    `second` may contain NaN outside the candidate pool; every page inside the top n must have a value.
    n = 1 (or 0) leaves the first-stage ranking unchanged.
    """
    if first.shape != second.shape:
        raise ValueError("first and second must have the same shape")
    ranks = ranks_from_scores(first, page_ids)  # 1-based
    out = np.empty_like(first, dtype=np.float64)
    n_pages = first.shape[1]
    for i in range(first.shape[0]):
        order = np.argsort(ranks[i])  # page indices in first-stage order
        top, rest = order[:n], order[n:]
        if np.isnan(second[i, top]).any():
            raise ValueError(f"query {i}: second-stage score missing for a top-{n} page")
        # stable sort by -second keeps first-stage order among equal second-stage scores
        top = top[np.argsort(-second[i, top], kind="stable")]
        final = np.concatenate([top, rest])
        out[i, final] = -np.arange(1, n_pages + 1, dtype=np.float64)
    return out


def top_n_pool(score_matrices: list[np.ndarray], n: int, page_ids: list[str]) -> np.ndarray:
    """Boolean mask [n_queries, n_pages]: page is in the top n of at least one of the given rankings."""
    mask = np.zeros(score_matrices[0].shape, dtype=bool)
    for s in score_matrices:
        mask |= ranks_from_scores(s, page_ids) <= n
    return mask
