"""Selective fusion (post-hoc set 3, H2): use a fused ranking only for queries where the text retriever is confident.

Confidence of a lexical retriever on one query: g = (s1 - s2) / s1, the relative margin between its two highest
scores (0 when s1 <= 0). A large margin means one page stands out lexically.
"""

from __future__ import annotations

import numpy as np


def relative_margin(scores: np.ndarray) -> np.ndarray:
    """[n_queries, n_pages] -> [n_queries] relative margin between each row's two highest scores."""
    top2 = -np.sort(-scores, axis=1)[:, :2]
    s1, s2 = top2[:, 0], top2[:, 1]
    return np.where(s1 > 0, (s1 - s2) / np.where(s1 > 0, s1, 1.0), 0.0)


def gated(fused: np.ndarray, base: np.ndarray, confidence: np.ndarray, tau: float) -> np.ndarray:
    """Per query: the fused row where confidence >= tau, else the base row. tau = inf always keeps the base."""
    if fused.shape != base.shape or confidence.shape != (base.shape[0],):
        raise ValueError("shape mismatch")
    use = confidence >= tau
    # Each query's row comes from one system only, so the two systems' score scales never mix within a ranking.
    return np.where(use[:, None], fused, base)
