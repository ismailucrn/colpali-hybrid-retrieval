"""Tests for two-stage reranking (CPU, synthetic scores)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fusion import ranks_from_scores  # noqa: E402
from src.rerank import rerank_top_n, top_n_pool  # noqa: E402

PAGES = ["p1", "p2", "p3", "p4", "p5"]


def order(scores):
    return [PAGES[j] for j in np.argsort(ranks_from_scores(scores, PAGES)[0])]


def test_rerank_reorders_only_top_n():
    first = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])           # p1 > p2 > p3 > p4 > p5
    second = np.array([[0.0, 1.0, 9.0, 100.0, np.nan]])     # p4 is best, but it is outside the top 3
    out = rerank_top_n(first, second, 3, PAGES)
    assert order(out) == ["p3", "p2", "p1", "p4", "p5"]


def test_n_one_keeps_first_stage():
    first = np.array([[1.0, 5.0, 3.0, 2.0, 4.0]])
    second = np.random.default_rng(0).normal(size=(1, 5))
    assert order(rerank_top_n(first, second, 1, PAGES)) == order(first)


def test_ties_in_second_stage_keep_first_stage_order():
    first = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    second = np.array([[1.0, 2.0, 2.0, 0.0, 0.0]])
    assert order(rerank_top_n(first, second, 5, PAGES)) == ["p2", "p3", "p1", "p4", "p5"]


def test_missing_score_inside_top_n_fails():
    first = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    second = np.array([[1.0, np.nan, 2.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        rerank_top_n(first, second, 2, PAGES)


def test_pool_is_union_of_top_n():
    a = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    b = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    assert top_n_pool([a, b], 2, PAGES)[0].tolist() == [True, True, False, True, True]
