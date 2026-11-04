"""Dev-split breakdown by query type for single systems and RRF fusions (never touches the test split).

Usage:
    python scripts/analyze_dev_by_type.py --k 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run, subset  # noqa: E402
from src.fusion import rrf  # noqa: E402
from src.runinfo import REPO_ROOT  # noqa: E402

SCORES = {
    "A BM25": "results/bm25_docvqa/scores.npz",
    "BGE-M3": "results/bge_m3_docvqa/scores.npz",
    "B ColPali": "results/colpali_v1.3_docvqa/scores.npz",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=float, default=1.0)
    args = parser.parse_args()

    task = load_task("vidore/docvqa_test_subsampled", "49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6", "test")
    dev = json.loads((REPO_ROOT / "results/splits/docvqa_split.json").read_text())["dev"]
    tags = {r["query"]: r["tag"] for r in csv.DictReader(open(REPO_ROOT / "results/splits/docvqa_query_tags.csv"))}

    systems = {name: np.load(REPO_ROOT / p, allow_pickle=False)["scores"].astype(np.float64) for name, p in SCORES.items()}
    systems[f"C RRF(BM25,ColPali) k={args.k:g}"] = rrf([systems["A BM25"], systems["B ColPali"]], task.page_ids, k=args.k)
    systems[f"D RRF(BGE-M3,ColPali) k={args.k:g}"] = rrf([systems["BGE-M3"], systems["B ColPali"]], task.page_ids, k=args.k)

    groups = {"all dev": dev, **{t: [q for q in dev if tags[q] == t] for t in ("exact", "general", "visual")}}
    print(f"{'system':32s}" + "".join(f"{g + f' (n={len(qs)})':>20s}" for g, qs in groups.items()))
    for name, scores in systems.items():
        run = scores_to_run(scores, task.queries, task.page_ids)
        cells = [evaluate(*subset(task.qrels, run, qs))[0]["ndcg_at_5"] for qs in groups.values()]
        print(f"{name:32s}" + "".join(f"{c:20.4f}" for c in cells))


if __name__ == "__main__":
    main()
