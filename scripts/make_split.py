"""Tag every query by type and split queries into dev/test, stratified by tag. Run once, then freeze.

Uses only the dataset's query texts (no images, no retrieval results). Writes:
  results/splits/docvqa_query_tags.csv  query, tag, rule, duplicated, split
  results/splits/docvqa_split.json      seed, fractions, counts per split x tag, query lists

Usage:
    python scripts/make_split.py --seed 0 --dev-fraction 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_task  # noqa: E402
from src.query_types import tag_query  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402


def stratified_split(queries: list[str], tags: dict[str, str], dev_fraction: float, seed: int) -> dict[str, str]:
    """Shuffle each tag group with a fixed seed and put the first round(n * dev_fraction) into dev."""
    rng = np.random.RandomState(seed)
    split = {}
    for tag in sorted(set(tags.values())):
        group = [q for q in queries if tags[q] == tag]  # dataset order, then shuffled
        order = rng.permutation(len(group))
        n_dev = int(round(len(group) * dev_fraction))
        for rank, idx in enumerate(order):
            split[group[idx]] = "dev" if rank < n_dev else "test"
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="vidore/docvqa_test_subsampled")
    parser.add_argument("--revision", default="49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-fraction", type=float, default=0.5)
    parser.add_argument("--out-dir", default="results/splits")
    args = parser.parse_args()

    task = load_task(args.dataset, args.revision, "test")
    tagged = {q: tag_query(q) for q in task.queries}
    tags = {q: t for q, (t, _) in tagged.items()}
    split = stratified_split(task.queries, tags, args.dev_fraction, args.seed)

    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "docvqa_query_tags.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "tag", "rule", "duplicated", "split"])
        for q in task.queries:
            w.writerow([q, tags[q], tagged[q][1], int(q in task.duplicated_queries), split[q]])

    counts = {s: dict(Counter(tags[q] for q in task.queries if split[q] == s)) for s in ("dev", "test")}
    record = collect_metadata(vars(args), "cli args", "cpu", args.seed)
    record.update({
        "counts": counts,
        "n_dev": sum(counts["dev"].values()),
        "n_test": sum(counts["test"].values()),
        "duplicated_by_split": {s: sum(1 for q in task.duplicated_queries if split[q] == s) for s in ("dev", "test")},
        "dev": [q for q in task.queries if split[q] == "dev"],
        "test": [q for q in task.queries if split[q] == "test"],
    })
    (out / "docvqa_split.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(json.dumps({k: record[k] for k in ("counts", "n_dev", "n_test", "duplicated_by_split")}, indent=1))
    print("tag totals:", dict(Counter(tags.values())))


if __name__ == "__main__":
    main()
