"""Check that src/evaluation.py gives the same numbers as MTEB 1.16 (the evaluator vidore-benchmark uses).

Uses a synthetic 451 x 500 score matrix quantised to bf16 (many ties, like real ColPali scores), with the
gold page forced into a top tie for a third of the queries. Needs MTEB in its own environment:
    python -m venv .venv_mteb && .venv_mteb/bin/pip install "mteb>=1.16.3,<1.17"
    .venv_mteb/bin/python scripts/checks/check_metrics_against_mteb.py

Expected: max |diff| = 0 for nDCG/Recall/P/MAP. MRR may differ on ties (documented in src/evaluation.py).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from mteb.evaluation.evaluators import RetrievalEvaluator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation import evaluate, scores_to_run  # noqa: E402

rng = np.random.default_rng(0)
nq, nd = 451, 500
pages = [f"page_{i:03d}.png" for i in range(nd)]
queries = [f"query {i}" for i in range(nq)]
scores = torch.tensor(rng.normal(10, 2, (nq, nd)), dtype=torch.bfloat16).float().numpy()
gold = rng.integers(0, nd, nq)
for i in range(0, nq, 3):
    scores[i, gold[i]] = scores[i].max()
qrels = {q: {pages[g]: 1} for q, g in zip(queries, gold)}
run = scores_to_run(scores, queries, pages)

k = [1, 3, 5, 10, 20, 50, 100]
ndcg, _map, recall, precision, _ = RetrievalEvaluator.evaluate(qrels, run, k, ignore_identical_ids=True)
mrr = RetrievalEvaluator.evaluate_custom(qrels, run, k, "mrr")
ours, _ = evaluate(qrels, run)

worst = 0.0
for kk in k:
    pairs = [("ndcg", ndcg[f"NDCG@{kk}"]), ("recall", recall[f"Recall@{kk}"]),
             ("precision", precision[f"P@{kk}"]), ("map", _map[f"MAP@{kk}"])]
    for name, theirs in pairs:
        worst = max(worst, abs(round(ours[f"{name}_at_{kk}"], 5) - theirs))
    print(f"@{kk}: ndcg ours={ours[f'ndcg_at_{kk}']:.5f} mteb={ndcg[f'NDCG@{kk}']:.5f} | "
          f"mrr ours={ours[f'mrr_at_{kk}']:.5f} mteb={mrr[0][f'MRR@{kk}']:.5f}")
print(f"max |diff| over nDCG/Recall/P/MAP (MTEB rounds to 5 decimals): {worst}")
sys.exit(0 if worst == 0 else 1)
