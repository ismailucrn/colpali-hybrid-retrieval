"""Fast CPU tests for the official-protocol task construction and the metrics. Run: python -m pytest tests"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_task  # noqa: E402
from src.evaluation import evaluate, gold_ranks, scores_to_run  # noqa: E402


def test_build_task_matches_official_dedup():
    queries = ["a", "b", "a", None, "c"]
    pages = ["p0", "p1", "p2", "p3", "p4"]
    task = build_task(queries, pages)
    assert task.queries == ["a", "b", "c"]  # first-occurrence order, None dropped
    assert task.qrels["a"] == {"p2": 1}  # last occurrence wins, as in the official dict comprehension
    assert task.duplicated_queries == {"a"}


def test_build_task_rejects_duplicate_pages():
    with pytest.raises(ValueError):
        build_task(["a", "b"], ["p0", "p0"])


def test_metrics_hand_computed():
    queries, pages = ["q1", "q2"], ["d1", "d2", "d3"]
    scores = np.array([[0.9, 0.5, 0.1],   # gold d2 at rank 2
                       [0.2, 0.3, 0.8]])  # gold d3 at rank 1
    qrels = {"q1": {"d2": 1}, "q2": {"d3": 1}}
    agg, per_query = evaluate(qrels, scores_to_run(scores, queries, pages), k_values=(1, 5))
    assert per_query["q1"]["gold_rank"] == 2
    assert per_query["q1"]["ndcg_at_5"] == pytest.approx(1 / math.log2(3))
    assert per_query["q1"]["recall_at_1"] == 0.0
    assert per_query["q1"]["mrr_at_5"] == pytest.approx(0.5)
    assert agg["ndcg_at_5"] == pytest.approx((1 / math.log2(3) + 1) / 2)
    assert agg["recall_at_1"] == pytest.approx(0.5)


@pytest.mark.parametrize("gold", ["d1", "d2", "d3"])
def test_tie_breaking_consistent_with_pytrec_eval(gold):
    # All scores tied: the gold rank we compute must imply the same nDCG pytrec_eval reports.
    run = scores_to_run(np.array([[1.0, 1.0, 1.0]]), ["q"], ["d1", "d2", "d3"])
    qrels = {"q": {gold: 1}}
    _, per_query = evaluate(qrels, run, k_values=(5,))
    rank = gold_ranks(qrels, run)["q"]
    assert per_query["q"]["ndcg_at_5"] == pytest.approx(1 / math.log2(rank + 1))
