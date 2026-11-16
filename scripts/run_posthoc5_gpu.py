"""POST-HOC set 5, GPU stage (Colab L4): ColPali v1.3 score matrices on further ViDoRe v1 tasks.

Pre-registered in docs/experiment_log.md (2026-11-16) before any run. Same model, revisions and code path
(scripts/run_posthoc4_gpu.py: stage_multivector) as the InfoVQA run of set 4. No metric is computed here.

Rows that share an image filename are one page (official evaluator: max over those rows). We hash the image bytes of
every row and refuse to run if two rows with the same filename hold different images; otherwise each page is encoded
once, which gives the same score as the official max over identical rows.

Usage:
    python scripts/run_posthoc5_gpu.py --tasks tatdqa shiftproject [--limit 4]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_posthoc4_gpu import stage_multivector  # noqa: E402
from src.data import build_task  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402


def load_task_with_pages(dataset: str, revision: str, split: str):
    from datasets import Image, load_dataset
    from PIL import Image as PILImage

    ds = load_dataset(dataset, split=split, revision=revision).select_columns(["query", "image_filename", "image"])
    ds = ds.cast_column("image", Image(decode=False))
    task = build_task(ds["query"], ds["image_filename"], allow_duplicate_pages=True)
    first, digest = {}, {}
    for r, (fn, img) in enumerate(zip(ds["image_filename"], ds["image"])):
        h = hashlib.sha256(img["bytes"]).hexdigest()
        if digest.setdefault(fn, h) != h:
            raise SystemExit(f"{dataset}: rows sharing {fn} hold different images; the max-merge must be done per row")
        first.setdefault(fn, r)
    task.images = [PILImage.open(io.BytesIO(ds[first[pid]]["image"]["bytes"])).convert("RGB") for pid in task.page_ids]
    task.n_rows = len(ds)
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/posthoc5_vidore.yaml")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sec = cfg["colpali"]
    for name in args.tasks:
        tk = cfg["tasks"][name]
        task = load_task_with_pages(tk["base"], tk["base_rev"], "test")
        n_rows = task.n_rows
        if args.limit:
            task.queries, task.page_ids, task.images = task.queries[:args.limit], task.page_ids[:args.limit], task.images[:args.limit]
        out_dir = REPO_ROOT / (sec["output_dir"].format(task=name) + ("_smoke" if args.limit else ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        record = collect_metadata(cfg, args.config, "cuda", cfg["seed"])
        record["label"] = "POST-HOC set 5 (pre-registered 2026-11-16)" + (" -- SMOKE TEST" if args.limit else "")
        record.update({"task": name, "dataset": {"name": tk["base"], "revision": tk["base_rev"], "split": "test"},
                       "n_rows": n_rows, "n_queries": len(task.queries), "n_pages": len(task.page_ids)})
        print(f"\n=== {name}: {n_rows} rows -> {len(task.page_ids)} pages, {len(task.queries)} queries")
        stage_multivector(sec, task, out_dir, record)
        (out_dir / "metrics.json").write_text(json.dumps(record, indent=2, default=str))
        print(f"wrote {out_dir.relative_to(REPO_ROOT)}")
        del task


if __name__ == "__main__":
    main()
