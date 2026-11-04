"""Dense text retriever for control D: BGE-M3 dense vectors over OCR page text, official protocol.

Encodes every page text and every (deduplicated) query with FlagEmbedding's BGEM3FlagModel, scores with a
dot product (vectors are L2-normalised), evaluates on the full query set and the dev split, and measures
indexing time, index size, single-query latency and peak memory. Test-split metrics are not computed here.

Saves to <output_dir>/: scores.npz, metrics.json, per_query.csv (same formats as the other systems).

Usage:
    python scripts/run_bge_m3.py --config configs/bge_m3_docvqa.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run, subset, write_per_query_csv  # noqa: E402
from src.runinfo import REPO_ROOT, MemoryTracker, Timer, collect_metadata, get_device, hub_revision  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/bge_m3_docvqa.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    set_seed(config["seed"])
    mcfg, dcfg = config["model"], config["data"]
    device = get_device(mcfg["device"])
    use_fp16 = bool(mcfg["use_fp16"]) and device != "cpu"
    print(f"device={device} fp16={use_fp16}")

    from FlagEmbedding import BGEM3FlagModel
    from huggingface_hub import snapshot_download

    memory = MemoryTracker(device)
    timings = {}

    start = time.perf_counter()
    task = load_task(dcfg["dataset"], dcfg["revision"], dcfg["split"],
                     text_dataset=dcfg["text_dataset"], text_revision=dcfg["text_revision"])
    timings["data_load_s"] = time.perf_counter() - start
    print(f"{len(task.queries)} queries, {len(task.page_ids)} pages")

    with Timer(device) as t:
        model_path = snapshot_download(mcfg["name"], revision=mcfg["revision"], allow_patterns=mcfg["allow_patterns"])
        model = BGEM3FlagModel(model_path, use_fp16=use_fp16, devices=device,
                               query_max_length=mcfg["query_max_length"],
                               passage_max_length=mcfg["passage_max_length"])
    timings["model_load_s"] = t.seconds
    memory.sample()

    bs = mcfg["batch_size"]
    with Timer(device) as t:
        page_emb = model.encode(task.page_texts, batch_size=bs, max_length=mcfg["passage_max_length"])["dense_vecs"]
    timings["index_pages_s"] = t.seconds
    timings["index_per_page_s"] = t.seconds / len(task.page_ids)
    memory.sample()

    with Timer(device) as t:
        query_emb = model.encode(task.queries, batch_size=bs, max_length=mcfg["query_max_length"])["dense_vecs"]
    timings["encode_queries_s"] = t.seconds
    memory.sample()

    page_emb = np.asarray(page_emb, dtype=np.float32)
    query_emb = np.asarray(query_emb, dtype=np.float32)
    scores = query_emb @ page_emb.T

    # Online latency: encode one query, dot product against all page vectors.
    times = []
    for i in range(-config["latency"]["warmup"], min(config["latency"]["n_queries"], len(task.queries))):
        start = time.perf_counter()
        q = np.asarray(model.encode([task.queries[max(i, 0)]], batch_size=1,
                                    max_length=mcfg["query_max_length"])["dense_vecs"], dtype=np.float32)
        _ = q @ page_emb.T
        if i >= 0:
            times.append(time.perf_counter() - start)
    ms = np.asarray(times) * 1000
    latency = {"n_queries": len(times), "warmup": config["latency"]["warmup"], "mean_ms": float(ms.mean()),
               "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
               "note": f"batch size 1; encode on {device}, dot product on CPU (numpy)"}
    peak = memory.report()

    index = {"n_pages": len(task.page_ids), "dim": int(page_emb.shape[1]), "dtype": "float32 (stored)",
             "size_mb": page_emb.nbytes / 1024**2}

    run = scores_to_run(scores, task.queries, task.page_ids)
    metrics, per_query = evaluate(task.qrels, run)
    clean = [q for q in task.queries if q not in task.duplicated_queries]
    metrics_clean, _ = evaluate(*subset(task.qrels, run, clean))
    split = json.loads((REPO_ROOT / config["splits"]).read_text())
    metrics_dev, _ = evaluate(*subset(task.qrels, run, split["dev"]))

    print(f"\nfull set: nDCG@5 = {metrics['ndcg_at_5']:.4f}  (paper, BGE-M3 on Unstructured + OCR: 0.257)")
    print(f"Recall@5 = {metrics['recall_at_5']:.4f}  Recall@10 = {metrics['recall_at_10']:.4f}  "
          f"nDCG@10 = {metrics['ndcg_at_10']:.4f}  MRR@10 = {metrics['mrr_at_10']:.4f}")
    print(f"dev ({metrics_dev['n_queries']} queries): nDCG@5 = {metrics_dev['ndcg_at_5']:.4f}")
    print(f"post-hoc, without {len(task.duplicated_queries)} duplicated query texts: nDCG@5 = {metrics_clean['ndcg_at_5']:.4f}")
    print(f"timings: {json.dumps({k: round(v, 2) for k, v in timings.items()})}")
    print(f"latency per query (median): {latency['median_ms']:.1f} ms; peak memory: {peak}")

    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "scores.npz", scores=scores.astype(np.float32),
                        queries=np.array(task.queries, dtype=str), page_ids=np.array(task.page_ids, dtype=str))
    split_of = {q: s for s in ("dev", "test") for q in split[s]}
    write_per_query_csv(out_dir / "per_query.csv", task.queries, task.qrels, per_query,
                        extra={"split": split_of, "duplicated": {q: int(q in task.duplicated_queries) for q in task.queries}})

    record = collect_metadata(config, args.config, device, config["seed"])
    record["packages"]["FlagEmbedding"] = __import__("importlib.metadata").metadata.version("FlagEmbedding")
    record.update({
        "system": "D component: BGE-M3 dense text",
        "resolved": {
            "use_fp16": use_fp16,
            "model_sha": hub_revision(mcfg["name"], mcfg["revision"], "model"),
            "dataset_sha": hub_revision(dcfg["dataset"], dcfg["revision"], "dataset"),
            "text_dataset_sha": hub_revision(dcfg["text_dataset"], dcfg["text_revision"], "dataset"),
        },
        "task": {"n_queries": len(task.queries), "n_pages": len(task.page_ids),
                 "n_duplicated_query_texts": len(task.duplicated_queries)},
        "metrics": metrics,
        "metrics_dev": metrics_dev,
        "metrics_without_duplicated_queries_posthoc": metrics_clean,
        "published": {"ndcg_at_5_paper_table2_bge_m3_unstructured_ocr": 0.257},
        "timings": timings,
        "latency": latency,
        "index": index,
        "peak_memory": peak,
    })
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out_dir}/{{scores.npz, metrics.json, per_query.csv}}")


if __name__ == "__main__":
    main()
