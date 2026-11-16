"""POST-HOC set 3, GPU stages (Colab L4). Pre-registered in e70f2a7. No metric is computed here: evaluation happens on
CPU, dev first (scripts/run_posthoc3_dev.py), test once (scripts/run_test_evaluation.py).

Stages:
    --stage embed    H4: Qwen3-VL-Embedding-2B over the 500 page images and 451 queries -> full score matrix (cosine)
    --stage rerank   H1: Qwen3-VL-Reranker-2B on (question, page image) for ColPali's top 20 of every query
    --stage ocr      H3: Qwen3.5-4B (vLLM) transcription of every page -> page_texts.json

Both Qwen3-VL models are run with the authors' own helper classes (scripts/qwen3_vl_*.py in each model repo, downloaded
at the pinned revision), so image resolution, prompt format and pooling follow the model cards.

--limit N runs on the first N queries / pages only and writes to <output_dir>_smoke (a smoke test, never reported).

Usage:
    python scripts/run_posthoc3_gpu.py --stage embed  [--limit 8]
    python scripts/run_posthoc3_gpu.py --stage rerank [--limit 4]
    python scripts/run_posthoc3_gpu.py --stage ocr    [--limit 4] [--fallback]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.data import load_task  # noqa: E402
from src.fusion import ranks_from_scores  # noqa: E402
from src.runinfo import REPO_ROOT, MemoryTracker, collect_metadata  # noqa: E402


class ErrorCounter(logging.Handler):
    """The helpers catch vision-processing errors, log them and silently score a 'NULL' input instead; we fail."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages = []

    def emit(self, rec):
        self.messages.append(rec.getMessage())


def load_helper(repo: str, revision: str, filename: str):
    """Import the model repo's own helper script at the pinned revision; return (module, local path, error counter)."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo, revision=revision)
    path = Path(local) / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    errors = ErrorCounter()
    module.logger.addHandler(errors)
    return module, local, errors


def fail_on_errors(errors: "ErrorCounter") -> None:
    if errors.messages:
        raise SystemExit(f"{len(errors.messages)} helper errors (inputs replaced by NULL); first: {errors.messages[0]}")


def latency_stats(times_s: list[float]) -> dict:
    ms = np.asarray(times_s) * 1000
    return {"n": len(ms), "median_ms": float(np.median(ms)), "mean_ms": float(ms.mean()), "p95_ms": float(np.percentile(ms, 95))}


def save_scores(path: Path, scores: np.ndarray, queries: list[str], page_ids: list[str]) -> None:
    np.savez_compressed(path, scores=scores.astype(np.float32), queries=np.array(queries, dtype=str),
                        page_ids=np.array(page_ids, dtype=str))


def stage_embed(cfg: dict, task, out_dir: Path, record: dict) -> None:
    import torch

    e = cfg["embedding"]
    helper, local, errors = load_helper(e["model"], e["revision"], "scripts/qwen3_vl_embedding.py")
    model = helper.Qwen3VLEmbedder(model_name_or_path=local, torch_dtype=getattr(torch, e["dtype"]))
    memory = MemoryTracker("cuda")
    bs = e["batch_size"]

    def encode(items: list[dict]) -> np.ndarray:
        out = []
        for i in range(0, len(items), bs):
            with torch.no_grad():
                out.append(model.process(items[i:i + bs]).float().cpu().numpy())
        return np.concatenate(out)

    t = time.perf_counter()
    pages = encode([{"image": img} for img in task.images])
    torch.cuda.synchronize()
    index_s = time.perf_counter() - t
    t = time.perf_counter()
    queries = encode([{"text": q, "instruction": e["query_instruction"]} for q in task.queries])
    torch.cuda.synchronize()
    query_s = time.perf_counter() - t
    fail_on_errors(errors)
    n_unique = len(np.unique(pages.round(4), axis=0))
    if n_unique < len(pages):
        raise SystemExit(f"only {n_unique} distinct page vectors for {len(pages)} pages")
    scores = queries @ pages.T  # both L2-normalised by the helper -> cosine

    lat = []  # one query at a time: encode + score against the stored page matrix
    for q in task.queries[:50]:
        t = time.perf_counter()
        v = encode([{"text": q, "instruction": e["query_instruction"]}])
        _ = v @ pages.T
        lat.append(time.perf_counter() - t)

    save_scores(out_dir / "scores.npz", scores, task.queries, task.page_ids)
    record.update({"system": "Q3E: Qwen3-VL-Embedding-2B (page images)", "embedding_dim": int(pages.shape[1]),
                   "timings": {"index_pages_s": index_s, "index_per_page_s": index_s / len(task.page_ids),
                               "encode_all_queries_s": query_s},
                   "latency": latency_stats(lat) | {"note": "encode one query + cosine against 500 stored page vectors"},
                   "index": {"size_mb": pages.astype(np.float32).nbytes / 1024**2, "note": "float32 page vectors"},
                   "peak_memory": memory.report()})
    print(f"pages {pages.shape}, queries {queries.shape}; index {index_s:.1f} s; query latency median "
          f"{record['latency']['median_ms']:.1f} ms")


def stage_rerank(cfg: dict, task, out_dir: Path, record: dict) -> None:
    import torch

    r = cfg["reranker"]
    helper, local, errors = load_helper(r["model"], r["revision"], "scripts/qwen3_vl_reranker.py")
    model = helper.Qwen3VLReranker(model_name_or_path=local, torch_dtype=getattr(torch, r["dtype"]))
    # Compatibility fix (transformers >= 5): the helper pads `input_ids` itself but leaves the processor's
    # `mm_token_type_ids` as an unpadded list, which Qwen3-VL's M-RoPE now requires as a tensor (smoke test,
    # 2026-11-16: TypeError in get_rope_index). Rebuild it from the padded ids exactly as the processor defines it:
    # 0 text, 1 image token, 2 video token. Nothing else in the helper changes.
    proc = model.processor
    image_id, video_id = proc.image_token_id, proc.video_token_id
    original_tokenize = model.tokenize

    def tokenize(pairs, **kwargs):
        inputs = original_tokenize(pairs, **kwargs)
        ids = inputs["input_ids"]
        inputs["mm_token_type_ids"] = (ids == image_id).long() + 2 * (ids == video_id).long()
        return inputs

    model.tokenize = tokenize
    memory = MemoryTracker("cuda")
    first = load_scores(cfg["scores"][r["pool_first"]], task.queries_all, task.page_ids)[task.query_index]
    ranks = ranks_from_scores(first, task.page_ids)
    scores = np.full(first.shape, np.nan)

    def score_pages(q: str, page_idx) -> list[float]:
        return model.process({"instruction": r["instruction"], "query": {"text": q},
                              "documents": [{"image": task.images[j]} for j in page_idx]})

    t0 = time.perf_counter()
    for i, q in enumerate(task.queries):
        top = np.argsort(ranks[i])[:r["pool_top"]]
        scores[i, top] = score_pages(q, top)
        if (i + 1) % 25 == 0 or i + 1 == len(task.queries):
            print(f"  {i + 1}/{len(task.queries)} queries, {time.perf_counter() - t0:.0f} s", flush=True)
    total_s = time.perf_counter() - t0
    fail_on_errors(errors)

    latency_by_n = {}
    lq = r["latency"]["n_queries"]
    for n in r["latency"]["n_values"]:
        times = []
        for i, q in enumerate(task.queries[:lq]):
            top = np.argsort(ranks[i])[:n]
            torch.cuda.synchronize()
            t = time.perf_counter()
            score_pages(q, top)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        latency_by_n[str(n)] = latency_stats(times)
        print(f"  latency N={n}: median {latency_by_n[str(n)]['median_ms']:.0f} ms/query")

    save_scores(out_dir / "scores.npz", scores, task.queries, task.page_ids)
    record.update({"system": "Qwen3-VL-Reranker-2B on ColPali's top pages (page images)",
                   "pool": {"first": r["pool_first"], "top": r["pool_top"], "pairs": int(np.isfinite(scores).sum())},
                   "timings": {"score_pool_s": total_s, "per_pair_s": total_s / max(1, int(np.isfinite(scores).sum()))},
                   "latency_by_n": latency_by_n, "peak_memory": memory.report(),
                   "note": "the helper scores one (query, page) pair per forward pass, as in the model card"})


def stage_ocr(cfg: dict, task, out_dir: Path, record: dict, fallback: bool) -> None:
    from vllm import LLM, SamplingParams

    o = cfg["ocr"]
    name, rev = (o["fallback_model"], o["fallback_revision"]) if fallback else (o["model"], o["revision"])
    llm = LLM(model=name, revision=rev, max_model_len=o["max_model_len"], limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.90, seed=cfg["seed"])
    params = SamplingParams(temperature=o["temperature"], max_tokens=o["max_new_tokens"], seed=cfg["seed"])
    messages = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": img},
                                              {"type": "text", "text": o["prompt"]}]}] for img in task.images]
    t0 = time.perf_counter()
    outputs = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
    total_s = time.perf_counter() - t0
    texts, finish, n_tokens = {}, [], []
    for pid, out in zip(task.page_ids, outputs):
        c = out.outputs[0]
        texts[pid] = c.text.strip()
        finish.append(c.finish_reason)
        n_tokens.append(len(c.token_ids))
    (out_dir / "page_texts.json").write_text(json.dumps(texts, indent=1, ensure_ascii=False))
    import vllm

    record.update({"system": f"VLM transcription with {name}", "model": name, "model_revision": rev,
                   "fallback_used": fallback, "vllm_version": vllm.__version__,
                   "timings": {"transcribe_all_s": total_s, "per_page_s": total_s / len(task.page_ids)},
                   "output": {"pages": len(texts), "empty": sum(not t for t in texts.values()),
                              "stopped_at_max_tokens": sum(f == "length" for f in finish),
                              "median_new_tokens": float(np.median(n_tokens)),
                              "median_words": float(np.median([len(t.split()) for t in texts.values()]))},
                   "note": "batched offline inference (vLLM); per_page_s is throughput, not single-page latency"})
    print(json.dumps(record["output"]), f"{total_s:.0f} s total")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/posthoc3_docvqa.yaml")
    parser.add_argument("--stage", choices=["embed", "rerank", "ocr"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="smoke test on the first N queries / pages")
    parser.add_argument("--fallback", action="store_true", help="H3 only: the pre-registered fallback model")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    d = cfg["data"]

    task = load_task(d["dataset"], d["revision"], d["split"], with_images=True)
    task.queries_all = list(task.queries)
    task.query_index = list(range(len(task.queries)))
    if args.limit:
        task.queries, task.query_index = task.queries[:args.limit], task.query_index[:args.limit]
        if args.stage in ("ocr", "embed"):
            task.page_ids, task.images = task.page_ids[:args.limit], task.images[:args.limit]
    key = {"embed": "embedding", "rerank": "reranker", "ocr": "ocr"}[args.stage]
    out_dir = REPO_ROOT / (cfg[key]["output_dir"] + ("_smoke" if args.limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    record = collect_metadata(cfg, args.config, "cuda", cfg["seed"])
    record["label"] = "POST-HOC set 3 (exploratory, pre-registered in e70f2a7)" + (" -- SMOKE TEST" if args.limit else "")
    record["n_queries"], record["n_pages"] = len(task.queries), len(task.page_ids)
    print(f"stage {args.stage}: {len(task.queries)} queries, {len(task.page_ids)} pages -> {out_dir}")

    if args.stage == "embed":
        stage_embed(cfg, task, out_dir, record)
    elif args.stage == "rerank":
        stage_rerank(cfg, task, out_dir, record)
    else:
        stage_ocr(cfg, task, out_dir, record, args.fallback)
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"wrote {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
