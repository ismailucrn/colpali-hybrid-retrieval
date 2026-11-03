"""Evaluate a saved score matrix (scores.npz) with the official ViDoRe protocol, on CPU.

Rebuilds queries and qrels from the dataset (without images), checks that the saved query/page order
matches, and prints the metrics. Used to verify GPU outputs locally and to evaluate every system
(BM25, fusion, ...) the same way.

Usage:
    python scripts/evaluate_scores.py --scores results/colpali_v1.3_docvqa/scores.npz \
        --dataset vidore/docvqa_test_subsampled --revision 49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", required=True)
    parser.add_argument("--dataset", default="vidore/docvqa_test_subsampled")
    parser.add_argument("--revision", default="49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    saved = np.load(args.scores, allow_pickle=False)
    scores, queries, page_ids = saved["scores"], list(saved["queries"]), list(saved["page_ids"])
    task = load_task(args.dataset, args.revision, args.split)
    if queries != task.queries or page_ids != task.page_ids:
        raise SystemExit("saved query/page order does not match the dataset")

    run = scores_to_run(scores, queries, page_ids)
    metrics, _ = evaluate(task.qrels, run)
    print(json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()
                      if k.split("_at_")[0] in ("ndcg", "recall", "mrr") or k == "n_queries"}, indent=1))


if __name__ == "__main__":
    main()
