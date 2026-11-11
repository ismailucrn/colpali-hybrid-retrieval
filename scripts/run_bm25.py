"""System A: BM25 over OCR page text on a ViDoRe QA task, official protocol, CPU.

Builds the BM25 index over the `_tesseract` text, scores every (deduplicated) query against every page,
evaluates on the full query set (comparable to the paper) and on the dev split, and measures
indexing time, index size, single-query latency and peak process memory.
Test-split metrics are deliberately not computed here: the test split is touched once, at the end.

Saves to <output_dir>/: scores.npz, metrics.json, per_query.csv (same formats as scripts/run_colpali.py).

Usage:
    python scripts/run_bm25.py --config configs/bm25_docvqa.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bm25 import BM25Index, Tokenizer  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run, subset, write_per_query_csv  # noqa: E402
from src.runinfo import REPO_ROOT, MemoryTracker, collect_metadata, hub_revision  # noqa: E402


def latency(index: BM25Index, queries: list[str], n: int, warmup: int) -> dict:
    times = []
    for i in range(-warmup, min(n, len(queries))):
        start = time.perf_counter()
        index.score(queries[max(i, 0)])
        if i >= 0:
            times.append(time.perf_counter() - start)
    ms = np.asarray(times) * 1000
    return {"n_queries": len(times), "warmup": warmup, "mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "note": "tokenise one query + BM25 scores for all pages, CPU, 1 thread"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/bm25_docvqa.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    dcfg = config["data"]
    memory = MemoryTracker("cpu")
    timings = {}

    start = time.perf_counter()
    task = load_task(dcfg["dataset"], dcfg["revision"], dcfg["split"],
                     text_dataset=dcfg["text_dataset"], text_revision=dcfg["text_revision"])
    timings["data_load_s"] = time.perf_counter() - start
    print(f"{len(task.queries)} queries, {len(task.page_ids)} pages, "
          f"{sum(not t.strip() for t in task.page_texts)} pages with empty OCR text")

    tokenizer = Tokenizer(config.get("tokenizer", "official"))  # "codes" = post-hoc variant A_code
    start = time.perf_counter()
    index = BM25Index(task.page_texts, tokenizer)
    timings["index_pages_s"] = time.perf_counter() - start
    timings["index_per_page_s"] = timings["index_pages_s"] / len(task.page_ids)

    start = time.perf_counter()
    scores = index.score_matrix(task.queries)
    timings["score_all_s"] = time.perf_counter() - start

    empty_queries = [q for q in task.queries if not tokenizer(q)]
    lat = latency(index, task.queries, config["latency"]["n_queries"], config["latency"]["warmup"])
    index_info = {
        "n_pages": len(task.page_ids),
        "total_tokens": int(sum(len(d) for d in index.tokenized_corpus)),
        "vocabulary_size": len(index.bm25.idf),
        "pickled_size_mb": len(pickle.dumps(index.bm25)) / 1024**2,
        "note": "pickled BM25Okapi object (term frequencies + idf) as a proxy for index size",
    }

    run = scores_to_run(scores, task.queries, task.page_ids)
    metrics, per_query = evaluate(task.qrels, run)
    clean = [q for q in task.queries if q not in task.duplicated_queries]
    metrics_clean, _ = evaluate(*subset(task.qrels, run, clean))

    split = json.loads((REPO_ROOT / config["splits"]).read_text())
    metrics_dev, _ = evaluate(*subset(task.qrels, run, split["dev"]))

    print(f"\nfull set: nDCG@5 = {metrics['ndcg_at_5']:.4f}  (paper, BM25 on Unstructured + OCR: 0.368; "
          f"tokenizer: {tokenizer.mode})")
    print(f"Recall@5 = {metrics['recall_at_5']:.4f}  Recall@10 = {metrics['recall_at_10']:.4f}  "
          f"nDCG@10 = {metrics['ndcg_at_10']:.4f}  MRR@10 = {metrics['mrr_at_10']:.4f}")
    print(f"dev ({metrics_dev['n_queries']} queries): nDCG@5 = {metrics_dev['ndcg_at_5']:.4f}")
    print(f"post-hoc, without {len(task.duplicated_queries)} duplicated query texts: nDCG@5 = {metrics_clean['ndcg_at_5']:.4f}")
    print(f"queries with no tokens left after preprocessing: {len(empty_queries)}")
    print(f"timings: {json.dumps({k: round(v, 3) for k, v in timings.items()})}")
    print(f"latency per query (median): {lat['median_ms']:.2f} ms; index: {index_info}")

    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "scores.npz", scores=scores.astype(np.float32),
                        queries=np.array(task.queries, dtype=str), page_ids=np.array(task.page_ids, dtype=str))
    split_of = {q: s for s in ("dev", "test") for q in split[s]}
    write_per_query_csv(out_dir / "per_query.csv", task.queries, task.qrels, per_query,
                        extra={"split": split_of, "duplicated": {q: int(q in task.duplicated_queries) for q in task.queries}})

    record = collect_metadata(config, args.config, "cpu", config["seed"])
    record.update({
        "system": config.get("system_name", "A: BM25"),
        "tokenizer": tokenizer.mode,
        "resolved": {
            "dataset_sha": hub_revision(dcfg["dataset"], dcfg["revision"], "dataset"),
            "text_dataset_sha": hub_revision(dcfg["text_dataset"], dcfg["text_revision"], "dataset"),
        },
        "task": {"n_queries": len(task.queries), "n_pages": len(task.page_ids),
                 "n_duplicated_query_texts": len(task.duplicated_queries),
                 "n_empty_pages": sum(not t.strip() for t in task.page_texts), "n_empty_queries": len(empty_queries)},
        "metrics": metrics,
        "metrics_dev": metrics_dev,
        "metrics_without_duplicated_queries_posthoc": metrics_clean,
        "published": {"ndcg_at_5_paper_table2_bm25_unstructured_ocr": 0.368},
        "timings": timings,
        "latency": lat,
        "index": index_info,
        "peak_memory": memory.report(),
    })
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out_dir}/{{scores.npz, metrics.json, per_query.csv}}")


if __name__ == "__main__":
    main()
