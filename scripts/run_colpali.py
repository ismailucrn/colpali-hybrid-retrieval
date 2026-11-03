"""System B: ColPali on a full ViDoRe QA task with the official protocol.

Encodes every page and every (deduplicated) query once, computes the full query x page score
matrix exactly as vidore-benchmark does (`processor.score`, CPU, float32 output), evaluates it, and
measures resource use: indexing time, embedding size, peak GPU memory and single-query latency.

Saves to <output_dir>/:
  scores.npz    float32 scores [n_queries, n_pages] + query texts + page ids (input to fusion experiments)
  metrics.json  official-protocol metrics, resource metrics and full run metadata
  per_query.csv per-query metrics and gold-page rank

Usage:
    python scripts/run_colpali.py --config configs/colpali_v1.3_docvqa.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run  # noqa: E402
from src.runinfo import (  # noqa: E402
    REPO_ROOT,
    MemoryTracker,
    Timer,
    collect_metadata,
    get_device,
    hub_revision,
    resolve_dtype,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(model, process, items: list, batch_size: int, memory: MemoryTracker) -> list[torch.Tensor]:
    """Same loop as vidore-benchmark's ColPaliRetriever.forward_*: batch, forward, move to CPU, unbind."""
    embeddings = []
    for start in range(0, len(items), batch_size):
        batch = process(items[start : start + batch_size]).to(model.device)
        with torch.no_grad():
            out = model(**batch).to("cpu")
        embeddings.extend(torch.unbind(out))
        memory.sample()
    return embeddings


def measure_latency(model, processor, queries: list[str], page_emb: list[torch.Tensor], cfg: dict, device: str) -> dict:
    """Online-search latency for one query: encode it, then MaxSim against all page embeddings on the device."""
    pages_on_device = [p.to(device) for p in page_emb]
    n, warmup = min(cfg["n_queries"], len(queries)), cfg["warmup"]
    encode_s, score_s = [], []
    for i in range(-warmup, n):
        query = queries[max(i, 0)]
        with Timer(device) as t_enc:
            batch = processor.process_queries([query]).to(model.device)
            with torch.no_grad():
                q = model(**batch)
        with Timer(device) as t_score:
            processor.score([q[0]], pages_on_device, batch_size=len(pages_on_device), device=device)
        if i >= 0:
            encode_s.append(t_enc.seconds)
            score_s.append(t_score.seconds)
    total = np.array(encode_s) + np.array(score_s)

    def summary(values) -> dict:
        values = np.asarray(values) * 1000
        return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95))}

    return {"n_queries": n, "warmup": warmup, "encode": summary(encode_s), "score": summary(score_s),
            "total": summary(total), "note": "batch size 1; scoring against all pages on the device"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/colpali_v1.3_docvqa.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    set_seed(config["seed"])
    mcfg, dcfg = config["model"], config["data"]
    device = get_device(mcfg["device"])
    dtype = resolve_dtype(mcfg["dtype"], device)
    print(f"device={device} dtype={dtype}")

    from colpali_engine.models import ColPali, ColPaliProcessor

    memory = MemoryTracker(device)
    timings = {}

    with Timer(device) as t:
        task = load_task(dcfg["dataset"], dcfg["revision"], dcfg["split"], with_images=True)
    timings["data_load_s"] = t.seconds
    print(f"{len(task.queries)} queries, {len(task.page_ids)} pages, {len(task.duplicated_queries)} duplicated query texts")

    with Timer(device) as t:
        model = ColPali.from_pretrained(
            mcfg["name"], revision=mcfg["revision"], torch_dtype=dtype, device_map=device
        ).eval()
        processor = ColPaliProcessor.from_pretrained(mcfg["name"], revision=mcfg["revision"])
    timings["model_load_s"] = t.seconds
    memory.sample()

    with Timer(device) as t:
        page_emb = encode(model, processor.process_images, task.images, config["batch_passage"], memory)
    timings["index_pages_s"] = t.seconds
    timings["index_per_page_s"] = t.seconds / len(page_emb)

    with Timer(device) as t:
        query_emb = encode(model, processor.process_queries, task.queries, config["batch_query"], memory)
    timings["encode_queries_s"] = t.seconds

    with Timer(device) as t:
        scores = processor.score(query_emb, page_emb, batch_size=config["batch_score"], device="cpu")
    timings["score_all_cpu_s"] = t.seconds
    scores = scores.numpy().astype(np.float32)

    latency = measure_latency(model, processor, task.queries, page_emb, config["latency"], device)
    peak = memory.report()

    # Index size: what has to be kept to search (all page vectors), at the stored precision.
    n_vectors = sum(p.shape[0] for p in page_emb)
    index = {
        "n_pages": len(page_emb),
        "vectors_per_page": sorted({p.shape[0] for p in page_emb}),
        "dim": int(page_emb[0].shape[1]),
        "dtype": str(page_emb[0].dtype),
        "total_vectors": n_vectors,
        "size_mb": sum(p.numel() * p.element_size() for p in page_emb) / 1024**2,
    }

    run = scores_to_run(scores, task.queries, task.page_ids)
    metrics, per_query = evaluate(task.qrels, run)
    clean = [q for q in task.queries if q not in task.duplicated_queries]
    metrics_clean, _ = evaluate({q: task.qrels[q] for q in clean}, {q: run[q] for q in clean})

    print(f"\nnDCG@5 = {metrics['ndcg_at_5']:.4f}  (model card v1.3: 0.5845; paper v1.0: 0.544)")
    print(f"Recall@5 = {metrics['recall_at_5']:.4f}  Recall@10 = {metrics['recall_at_10']:.4f}  "
          f"nDCG@10 = {metrics['ndcg_at_10']:.4f}  MRR@10 = {metrics['mrr_at_10']:.4f}")
    print(f"post-hoc, without {len(task.duplicated_queries)} duplicated query texts: nDCG@5 = {metrics_clean['ndcg_at_5']:.4f}")
    print(f"timings: {json.dumps({k: round(v, 2) for k, v in timings.items()})}")
    print(f"latency per query (median): {latency['total']['median_ms']:.1f} ms; peak memory: {peak}")

    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "scores.npz", scores=scores, queries=np.array(task.queries, dtype=str),
                        page_ids=np.array(task.page_ids, dtype=str))

    with open(out_dir / "per_query.csv", "w", newline="") as f:
        cols = ["query", "gold_page", "duplicated", "gold_rank", "ndcg_at_5", "recall_at_5", "recall_at_10",
                "ndcg_at_10", "mrr_at_10"]
        writer = csv.writer(f)
        writer.writerow(cols)
        for q in task.queries:
            row = per_query[q]
            writer.writerow([q, next(iter(task.qrels[q])), int(q in task.duplicated_queries), row["gold_rank"],
                             *(round(row[c], 6) for c in cols[4:])])

    record = collect_metadata(config, args.config, device, config["seed"])
    record.update({
        "system": "B: ColPali",
        "resolved": {
            "dtype": str(dtype),
            "model_sha": hub_revision(mcfg["name"], mcfg["revision"], "model"),
            "dataset_sha": hub_revision(dcfg["dataset"], dcfg["revision"], "dataset"),
        },
        "task": {"n_queries": len(task.queries), "n_pages": len(task.page_ids),
                 "n_duplicated_query_texts": len(task.duplicated_queries)},
        "metrics": metrics,
        "metrics_without_duplicated_queries_posthoc": metrics_clean,
        "published": {"ndcg_at_5_model_card_v1.3": 0.58452, "ndcg_at_5_paper_table2_v1.0": 0.544},
        "timings": timings,
        "latency": latency,
        "index": index,
        "peak_memory": peak,
    })
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out_dir}/{{scores.npz, metrics.json, per_query.csv}}")


if __name__ == "__main__":
    main()
