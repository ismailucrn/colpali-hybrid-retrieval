"""Retrieval metrics matching the ViDoRe v1 / MTEB evaluator.

The official evaluator (vidore-benchmark -> MTEB 1.16 `RetrievalEvaluator.evaluate`) computes nDCG, Recall and
Precision with pytrec_eval; we call pytrec_eval directly with the same measures so the numbers are
comparable, including its tie-breaking (equal scores are ordered by document id, descending).
MRR is computed here with the same ordering (MTEB uses its own MRR code, which may order ties differently).
"""

from __future__ import annotations

import numpy as np
import pytrec_eval

K_VALUES = (1, 3, 5, 10, 20, 50, 100)


def scores_to_run(scores: np.ndarray, queries: list[str], page_ids: list[str]) -> dict[str, dict[str, float]]:
    """Score matrix [n_queries, n_pages] -> TREC-style run {query: {page_id: score}}."""
    if scores.shape != (len(queries), len(page_ids)):
        raise ValueError(f"scores shape {scores.shape} != ({len(queries)}, {len(page_ids)})")
    return {q: {pid: float(s) for pid, s in zip(page_ids, row)} for q, row in zip(queries, scores)}


def ranking(doc_scores: dict[str, float]) -> list[str]:
    """Documents in trec_eval order: score descending, ties broken by document id descending."""
    return sorted(doc_scores, key=lambda d: (doc_scores[d], d), reverse=True)


def gold_ranks(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]]) -> dict[str, int]:
    """1-based rank of the (single) relevant page for each query."""
    ranks = {}
    for q, rel in qrels.items():
        order = ranking(run[q])
        relevant = [d for d, r in rel.items() if r > 0]
        ranks[q] = min(order.index(d) + 1 for d in relevant)
    return ranks


def evaluate(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]],
             k_values: tuple[int, ...] = K_VALUES) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return (aggregate metrics, per-query metrics). Keys follow ViDoRe naming, e.g. `ndcg_at_5`."""
    qrels = {q: qrels[q] for q in run}
    ks = ",".join(str(k) for k in k_values)
    measures = {f"ndcg_cut.{ks}", f"recall.{ks}", f"P.{ks}", f"map_cut.{ks}"}
    raw = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)

    ranks = gold_ranks(qrels, run)
    per_query: dict[str, dict[str, float]] = {}
    for q, m in raw.items():
        row = {}
        for k in k_values:
            row[f"ndcg_at_{k}"] = m[f"ndcg_cut_{k}"]
            row[f"recall_at_{k}"] = m[f"recall_{k}"]
            row[f"precision_at_{k}"] = m[f"P_{k}"]
            row[f"map_at_{k}"] = m[f"map_cut_{k}"]
            row[f"mrr_at_{k}"] = 1.0 / ranks[q] if ranks[q] <= k else 0.0
        row["gold_rank"] = ranks[q]
        per_query[q] = row

    metric_names = [k for k in next(iter(per_query.values())) if k != "gold_rank"]
    aggregate = {name: float(np.mean([row[name] for row in per_query.values()])) for name in metric_names}
    aggregate["n_queries"] = len(per_query)
    return aggregate, per_query
