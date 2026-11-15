"""Diagnostic found during post-hoc set 2 (2026-11-15, NOT pre-registered; dev split only, test not touched).

The four pages with empty OCR text get an empty-string BGE-M3 embedding, which lands in BGE-M3's top 5 for most
queries. This check counts how often, and measures on DEV how much BGE-M3, D = RRF(BGE-M3, ColPali) with k = 1 and
T_hyb = RRF(BM25, BGE-M3) with k = 5 change when those pages are moved to the bottom of BGE-M3's ranking.

Usage:
    python scripts/checks/bge_empty_pages.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run, subset  # noqa: E402
from src.fusion import rrf  # noqa: E402


def main() -> None:
    task = load_task("vidore/docvqa_test_subsampled", "49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6", "test",
                     text_dataset="vidore/docvqa_test_subsampled_tesseract",
                     text_revision="9a183a7237d678eaaf68dc0f640a42f46016802f")
    dev = json.loads((REPO / "results/splits/docvqa_split.json").read_text())["dev"]
    bge = load_scores("results/bge_m3_docvqa/scores.npz", task.queries, task.page_ids)
    bm25 = load_scores("results/bm25_docvqa/scores.npz", task.queries, task.page_ids)
    colpali = load_scores("results/colpali_v1.3_docvqa/scores.npz", task.queries, task.page_ids)

    empty = [j for j, t in enumerate(task.page_texts) if not t.strip()]
    top5 = np.argsort(-bge, axis=1)[:, :5]
    in_top5 = np.isin(top5, empty).any(axis=1)
    print(f"pages with empty OCR text: {[task.page_ids[j] for j in empty]}")
    print(f"all {len(task.queries)} queries: an empty page is in BGE-M3's top 5 for {int(in_top5.sum())}; "
          f"per page: {[int(np.isin(top5, [j]).any(axis=1).sum()) for j in empty]}")

    demoted = bge.copy()
    demoted[:, empty] = bge.min() - 1.0

    def dev_ndcg5(scores):
        return evaluate(*subset(task.qrels, scores_to_run(scores, task.queries, task.page_ids), dev))[0]["ndcg_at_5"]

    gold_empty = [q for q in dev if task.page_ids.index(next(iter(task.qrels[q]))) in empty]
    print(f"\nDEV only ({len(dev)} queries; {len(gold_empty)} have an empty-OCR gold page), nDCG@5 as is -> empty pages demoted:")
    rows = {"BGE-M3": (bge, demoted),
            "D = RRF(BGE-M3, ColPali), k=1": (rrf([bge, colpali], task.page_ids, k=1), rrf([demoted, colpali], task.page_ids, k=1)),
            "T_hyb = RRF(BM25, BGE-M3), k=5": (rrf([bm25, bge], task.page_ids, k=5), rrf([bm25, demoted], task.page_ids, k=5))}
    record = {"label": "diagnostic, not pre-registered; dev split only", "empty_pages": [task.page_ids[j] for j in empty],
              "queries_with_empty_page_in_bge_top5": int(in_top5.sum()), "n_queries": len(task.queries),
              "dev_n": len(dev), "dev_queries_with_empty_gold": len(gold_empty), "dev_ndcg_at_5": {}}
    for name, (a, b) in rows.items():
        record["dev_ndcg_at_5"][name] = {"as_is": dev_ndcg5(a), "empty_demoted": dev_ndcg5(b)}
        print(f"  {name}: {dev_ndcg5(a):.4f} -> {dev_ndcg5(b):.4f}")
    print(f"  (B ColPali dev: {dev_ndcg5(colpali):.4f}; A BM25 dev: {dev_ndcg5(bm25):.4f})")
    out = REPO / "results/checks/bge_empty_pages.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
