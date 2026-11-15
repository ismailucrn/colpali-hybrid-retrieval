"""POST-HOC set 2: cross-encoder scores (bge-reranker-v2-m3) over a candidate pool of OCR page texts.

The pool per query is the union of the top `pool_top` pages of T_hyb (dev-selected k) and of ColPali; no relevance
labels are used. max_length is chosen from token counts only (pre-registered rule), before any score exists.
Also measures per-query reranking latency for several N and the model's memory.

Saves to <reranker_output_dir>/: ce_scores.npz (NaN outside the pool), metrics.json.

Usage:
    python scripts/run_reranker.py --config configs/posthoc2_docvqa.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.data import load_task  # noqa: E402
from src.fusion import ranks_from_scores, rrf  # noqa: E402
from src.rerank import top_n_pool  # noqa: E402
from src.runinfo import REPO_ROOT, MemoryTracker, Timer, collect_metadata, get_device, hub_revision  # noqa: E402

N_SPECIAL = 4  # XLM-R pair encoding: <s> query </s></s> passage </s>


def choose_max_length(tokenizer, queries: list[str], pages: list[str], candidates: list[int], coverage: float) -> dict:
    q_len = max(len(tokenizer(q, add_special_tokens=False)["input_ids"]) for q in queries)
    p_len = np.array([len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in pages])
    pair_len = p_len + q_len + N_SPECIAL
    fits = {int(c): float((pair_len <= c).mean()) for c in candidates}
    chosen = next((c for c in candidates if fits[c] >= coverage), candidates[-1])
    return {"longest_query_tokens": int(q_len), "page_tokens_median": float(np.median(p_len)),
            "page_tokens_p95": float(np.percentile(p_len, 95)), "page_tokens_max": int(p_len.max()),
            "share_of_pairs_fitting": fits, "coverage_rule": coverage, "chosen_max_length": int(chosen),
            "pages_truncated_at_chosen": int((pair_len > chosen).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/posthoc2_docvqa.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    dcfg, rcfg = config["data"], config["reranker"]

    import torch
    from FlagEmbedding import FlagReranker
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    torch.manual_seed(config["seed"])
    device = get_device(rcfg["device"])
    use_fp16 = bool(rcfg["use_fp16"]) and device != "cpu"
    print(f"device={device} fp16={use_fp16}")

    task = load_task(dcfg["dataset"], dcfg["revision"], dcfg["split"],
                     text_dataset=dcfg["text_dataset"], text_revision=dcfg["text_revision"])
    print(f"{len(task.queries)} queries, {len(task.page_ids)} pages")

    # Candidate pool: union of T_hyb's and ColPali's top pool_top (labels not used).
    out2 = REPO_ROOT / config["output_dir"]
    k_hyb = json.loads((out2 / "dev_selection_hybrid.json").read_text())["systems"]["T_hyb"]["selected_k"]
    base = {k: load_scores(p, task.queries, task.page_ids) for k, p in config["scores"].items()}
    t_hyb = rrf([base[i] for i in config["hybrid"]["inputs"]], task.page_ids, k=k_hyb)
    pool = top_n_pool([t_hyb, base["B"]], rcfg["pool_top"], task.page_ids)
    print(f"pool: T_hyb (k={k_hyb}) top {rcfg['pool_top']} ∪ ColPali top {rcfg['pool_top']}; "
          f"{int(pool.sum())} pairs, {pool.sum(1).mean():.1f} pages/query on average")

    memory = MemoryTracker(device)
    timings = {}
    with Timer(device) as t:
        path = snapshot_download(rcfg["name"], revision=rcfg["revision"],
                                 allow_patterns=["*.json", "*.safetensors", "sentencepiece.bpe.model"])
    timings["download_s"] = t.seconds

    tokenizer = AutoTokenizer.from_pretrained(path)
    length = choose_max_length(tokenizer, task.queries, task.page_texts, rcfg["max_length_candidates"], rcfg["coverage"])
    max_length = length["chosen_max_length"]
    print(f"max_length rule: {json.dumps(length)}")

    with Timer(device) as t:
        model = FlagReranker(path, use_fp16=use_fp16, devices=device, batch_size=rcfg["batch_size"],
                             max_length=max_length, normalize=rcfg["normalize"])
    timings["model_load_s"] = t.seconds
    memory.sample()

    qi, pj = np.nonzero(pool)
    pairs = [(task.queries[i], task.page_texts[j]) for i, j in zip(qi, pj)]
    with Timer(device) as t:
        values = model.compute_score(pairs, batch_size=rcfg["batch_size"], max_length=max_length)
    timings["score_pool_s"] = t.seconds
    timings["score_per_pair_ms"] = t.seconds / len(pairs) * 1000
    memory.sample()
    ce = np.full(pool.shape, np.nan, dtype=np.float32)
    ce[qi, pj] = np.asarray(values, dtype=np.float32)
    print(f"scored {len(pairs)} pairs in {timings['score_pool_s']:.1f} s ({timings['score_per_pair_ms']:.1f} ms/pair)")

    # Online latency: rerank T_hyb's top N for one query (tokenise + forward), batch = N pairs.
    ranks = ranks_from_scores(t_hyb, task.page_ids)
    lcfg = rcfg["latency"]
    latency = {}
    for n in lcfg["n_values"]:
        times = []
        for i in range(-lcfg["warmup"], lcfg["n_queries"]):
            q = max(i, 0)
            top = np.argsort(ranks[q])[:n]
            batch = [(task.queries[q], task.page_texts[j]) for j in top]
            start = time.perf_counter()
            model.compute_score(batch, batch_size=n, max_length=max_length)
            if device == "mps":
                torch.mps.synchronize()
            if i >= 0:
                times.append(time.perf_counter() - start)
        ms = np.asarray(times) * 1000
        latency[str(n)] = {"median_ms": float(np.median(ms)), "mean_ms": float(ms.mean()), "p95_ms": float(np.percentile(ms, 95))}
        print(f"latency, rerank top {n}: median {latency[str(n)]['median_ms']:.1f} ms")
    memory.sample()
    peak = memory.report()

    out_dir = REPO_ROOT / config["reranker_output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "ce_scores.npz", scores=ce, queries=np.array(task.queries, dtype=str),
                        page_ids=np.array(task.page_ids, dtype=str))
    record = collect_metadata(config, args.config, device, config["seed"])
    record["packages"]["FlagEmbedding"] = __import__("importlib.metadata").metadata.version("FlagEmbedding")
    record.update({
        "label": "POST-HOC set 2 (exploratory): cross-encoder scores over a candidate pool",
        "resolved": {"use_fp16": use_fp16, "model_sha": hub_revision(rcfg["name"], rcfg["revision"], "model"),
                     "dataset_sha": hub_revision(dcfg["dataset"], dcfg["revision"], "dataset"),
                     "text_dataset_sha": hub_revision(dcfg["text_dataset"], dcfg["text_revision"], "dataset"),
                     "T_hyb_k": k_hyb},
        "pool": {"top": rcfg["pool_top"], "n_pairs": int(pool.sum()), "mean_pages_per_query": float(pool.sum(1).mean()),
                 "min_pages_per_query": int(pool.sum(1).min()), "max_pages_per_query": int(pool.sum(1).max())},
        "max_length": length,
        "timings": timings,
        "latency_by_n": latency,
        "latency_note": f"rerank T_hyb's top N for one query on {device}: tokenisation + forward, batch = N pairs; "
                        f"{lcfg['n_queries']} queries after {lcfg['warmup']} warm-up",
        "peak_memory": peak,
    })
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"wrote {out_dir}/{{ce_scores.npz, metrics.json}}; peak memory {peak}")


if __name__ == "__main__":
    main()
