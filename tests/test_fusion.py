"""Tests for RRF fusion (CPU, synthetic scores)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import gold_ranks, scores_to_run  # noqa: E402
from src.fusion import ranks_from_scores, rrf  # noqa: E402


def test_ranks_match_evaluator_order_including_ties():
    pages = ["a", "c", "b", "d"]
    scores = np.array([[1.0, 2.0, 2.0, 0.5]])  # tie between "c" and "b" -> "c" first (id descending)
    ranks = ranks_from_scores(scores, pages)
    assert ranks.tolist() == [[3, 1, 2, 4]]
    run = scores_to_run(scores, ["q"], pages)
    for page, r in zip(pages, ranks[0]):
        assert gold_ranks({"q": {page: 1}}, run)["q"] == r


def test_rrf_formula():
    pages = ["p1", "p2", "p3"]
    s1 = np.array([[3.0, 2.0, 1.0]])  # ranks 1, 2, 3
    s2 = np.array([[1.0, 3.0, 2.0]])  # ranks 3, 1, 2
    fused = rrf([s1, s2], pages, k=60)
    expected = [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62]
    assert fused[0] == pytest.approx(expected)


def test_rrf_of_one_system_preserves_its_ranking():
    rng = np.random.default_rng(0)
    pages = [f"p{i}" for i in range(20)]
    s = rng.normal(size=(5, 20))
    assert (ranks_from_scores(rrf([s], pages, k=10), pages) == ranks_from_scores(s, pages)).all()


def test_weighted_rrf_equal_weights_matches_plain_rrf_ranking():
    from src.fusion import weighted_rrf

    rng = np.random.default_rng(1)
    pages = [f"p{i}" for i in range(15)]
    a, b = rng.normal(size=(4, 15)), rng.normal(size=(4, 15))
    plain = ranks_from_scores(rrf([a, b], pages, k=5), pages)
    weighted = ranks_from_scores(weighted_rrf([a, b], [0.5, 0.5], pages, k=5), pages)
    assert (plain == weighted).all()


def test_weighted_rrf_full_weight_reproduces_one_system():
    from src.fusion import weighted_rrf

    rng = np.random.default_rng(2)
    pages = [f"p{i}" for i in range(15)]
    a, b = rng.normal(size=(4, 15)), rng.normal(size=(4, 15))
    fused = weighted_rrf([a, b], [1.0, 0.0], pages, k=1)
    assert (ranks_from_scores(fused, pages) == ranks_from_scores(a, pages)).all()


def test_minmax_and_convex_combination():
    from src.fusion import convex_combination, minmax_per_query

    s = np.array([[2.0, 4.0, 3.0], [1.0, 1.0, 1.0]])
    assert minmax_per_query(s).tolist() == [[0.0, 1.0, 0.5], [0.0, 0.0, 0.0]]
    colpali = np.array([[10.0, 30.0, 20.0]])
    text = np.array([[5.0, 0.0, 1.0]])
    pages = ["p1", "p2", "p3"]
    # alpha = 1 reproduces ColPali's ranking exactly
    assert ranks_from_scores(convex_combination(colpali, text, 1.0), pages).tolist() == \
        ranks_from_scores(colpali, pages).tolist()
    fused = convex_combination(colpali, text, 0.5)
    assert fused[0].tolist() == pytest.approx([0.5 * 0.0 + 0.5 * 1.0, 0.5 * 1.0 + 0.0, 0.5 * 0.5 + 0.5 * 0.2])
    with pytest.raises(ValueError):
        convex_combination(colpali, text, 1.5)
