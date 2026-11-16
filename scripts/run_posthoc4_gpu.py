"""POST-HOC set 4, GPU stages (Colab L4). Pre-registered in docs/experiment_log.md (2026-11-16) before any run.
Computes score matrices only; no metric is computed here (dev selection and the test run happen on CPU).

Stages:
    --stage multivector --model colqwen25|tomoro|colpali --task docvqa|infovqa
    --stage embed8b  --task docvqa|infovqa       Qwen3-VL-Embedding-8B (helper and settings as the 2B model in set 3)
    --stage rerank8b --task docvqa|infovqa       Qwen3-VL-Reranker-8B on ColPali's top `pool_top` pages
--limit N: smoke test on the first N queries / pages, written to <output_dir>_smoke (never reported).

Usage:
    python scripts/run_posthoc4_gpu.py --stage multivector --model colqwen25 --task docvqa [--limit 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_posthoc3_gpu import latency_stats, save_scores, stage_embed, stage_rerank  # noqa: E402
from src.data import load_task  # noqa: E402
from src.runinfo import REPO_ROOT, MemoryTracker, collect_metadata  # noqa: E402


def stage_multivector(sec: dict, task, out_dir: Path, record: dict) -> None:
    import torch

    memory = MemoryTracker("cuda")
    bs = sec["batch_size"]
    if sec["family"] == "colpali_engine":
        import colpali_engine.models as M
        from huggingface_hub import snapshot_download

        path = snapshot_download(sec["model"], revision=sec["revision"])  # adapter at its pinned revision
        model = getattr(M, sec["model_class"]).from_pretrained(
            path, revision=sec["base_revision"], torch_dtype=torch.bfloat16, device_map="cuda").eval()
        proc = getattr(M, sec["processor_class"]).from_pretrained(path)

        def run(items, process):
            out = []
            for i in range(0, len(items), bs):
                with torch.no_grad():
                    batch = process(items[i:i + bs]).to(model.device)
                    out.extend(torch.unbind(model(**batch).to("cpu")))
            return out

        enc_pages = lambda: run(task.images, proc.process_images)  # noqa: E731
        enc_queries = lambda qs: run(qs, proc.process_queries)  # noqa: E731
        score = lambda q, p: proc.score(q, p, batch_size=4, device="cpu").float().numpy()  # noqa: E731
    else:
        from sentence_transformers import MultiVectorEncoder

        # flash-attn is not installed on Colab; the card says to use sdpa then.
        model = MultiVectorEncoder(sec["model"], revision=sec["revision"], trust_remote_code=True, device="cuda",
                                   model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})
        enc_pages = lambda: list(model.encode_document(task.images, batch_size=bs))  # noqa: E731
        enc_queries = lambda qs: list(model.encode_query(qs, batch_size=bs))  # noqa: E731
        score = lambda q, p: np.asarray(model.similarity(q, p).float().cpu())  # noqa: E731

    t = time.perf_counter()
    pages = enc_pages()
    torch.cuda.synchronize()
    index_s = time.perf_counter() - t
    queries = enc_queries(task.queries)
    scores = score(queries, pages)
    if scores.shape != (len(task.queries), len(task.page_ids)):
        raise SystemExit(f"unexpected score shape {scores.shape}")
    lat = []
    for q in task.queries[:50]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        score(enc_queries([q]), pages)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t)
    save_scores(out_dir / "scores.npz", scores, task.queries, task.page_ids)

    def as_np(x):
        return x.float().cpu().numpy() if hasattr(x, "float") else np.asarray(x, dtype=np.float32)

    sizes = [as_np(p).shape for p in pages]
    record.update({"system": sec["model"], "family": sec["family"],
                   "timings": {"index_pages_s": index_s, "index_per_page_s": index_s / len(task.page_ids)},
                   "latency": latency_stats(lat) | {"note": "encode one query + MaxSim against all stored pages (CPU scoring for colpali_engine)"},
                   "index": {"vectors_per_page_min_max": [min(s[0] for s in sizes), max(s[0] for s in sizes)],
                             "dim": int(sizes[0][1]),
                             "size_mb": sum(s[0] * s[1] for s in sizes) * 2 / 1024**2,
                             "note": "page vectors at bf16 (2 bytes)"},
                   "peak_memory": memory.report()})
    print(f"pages {len(pages)} (vectors/page {record['index']['vectors_per_page_min_max']}, dim {sizes[0][1]}); "
          f"index {index_s:.0f} s; latency median {record['latency']['median_ms']:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/posthoc4_docvqa.yaml")
    parser.add_argument("--stage", choices=["multivector", "embed8b", "rerank8b"], required=True)
    parser.add_argument("--model", help="multivector stage: colqwen25 | tomoro | colpali")
    parser.add_argument("--task", choices=["docvqa", "infovqa"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tk = cfg["tasks"][args.task]
    sec = {"multivector": lambda: cfg["multivector"][args.model], "embed8b": lambda: cfg["embedding8b"],
           "rerank8b": lambda: cfg["reranker8b"]}[args.stage]()

    task = load_task(tk["dataset"], tk["revision"], tk["split"], with_images=True)
    task.queries_all = list(task.queries)
    task.query_index = list(range(len(task.queries)))
    if args.limit:
        task.queries, task.query_index = task.queries[:args.limit], task.query_index[:args.limit]
        if args.stage != "rerank8b":
            task.page_ids, task.images = task.page_ids[:args.limit], task.images[:args.limit]
    out_dir = REPO_ROOT / (sec["output_dirs"][args.task] + ("_smoke" if args.limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    record = collect_metadata(cfg, args.config, "cuda", cfg["seed"])
    record["label"] = "POST-HOC set 4 (pre-registered 2026-11-16)" + (" -- SMOKE TEST" if args.limit else "")
    record.update({"task": args.task, "dataset": tk, "n_queries": len(task.queries), "n_pages": len(task.page_ids)})
    print(f"stage {args.stage} {args.model or ''} on {args.task}: {len(task.queries)} queries, {len(task.page_ids)} pages")

    if args.stage == "multivector":
        stage_multivector(sec, task, out_dir, record)
    elif args.stage == "embed8b":
        stage_embed({"embedding": sec, "seed": cfg["seed"]}, task, out_dir, record)
    else:
        first = sec["first_scores"][args.task]
        stage_rerank({"reranker": sec, "scores": {sec["pool_first"]: first}}, task, out_dir, record)
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"wrote {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
