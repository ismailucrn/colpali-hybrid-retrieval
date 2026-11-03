"""ColPali smoke test.

Loads the checkpoint, encodes a handful of ViDoRe pages and their queries, scores them with
late interaction (MaxSim), and checks that each query ranks its own page first. Records
timing, peak memory and full run metadata to results/smoke_test/.

Usage:
    python scripts/smoke_test.py --config configs/smoke_test.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


def load_samples(cfg: dict) -> tuple[list[str], list, list[int]]:
    """Return queries, unique page images, and the gold page index of each query."""
    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], split=cfg["split"], revision=cfg["revision"], streaming=True)
    rows = list(islice(ds, cfg["n_samples"]))

    page_index: dict[str, int] = {}
    images, queries, gold = [], [], []
    for row in rows:
        key = row["image_filename"]
        if key not in page_index:
            page_index[key] = len(images)
            images.append(row["image"].convert("RGB"))
        queries.append(row["query"])
        gold.append(page_index[key])
    return queries, images, gold


def encode(model, processor, items: list, kind: str, batch_size: int) -> list[torch.Tensor]:
    process = processor.process_images if kind == "images" else processor.process_queries
    embeddings = []
    for start in range(0, len(items), batch_size):
        batch = process(items[start : start + batch_size]).to(model.device)
        with torch.no_grad():
            out = model(**batch)
        embeddings.extend(torch.unbind(out.to("cpu")))
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/smoke_test.yaml")
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
        queries, images, gold = load_samples(dcfg)
    timings["data_load_s"] = t.seconds
    print(f"{len(queries)} queries, {len(images)} unique pages")

    with Timer(device) as t:
        model = ColPali.from_pretrained(
            mcfg["name"], revision=mcfg["revision"], torch_dtype=dtype, device_map=device
        ).eval()
        processor = ColPaliProcessor.from_pretrained(mcfg["name"], revision=mcfg["revision"])
    timings["model_load_s"] = t.seconds
    memory.sample()

    with Timer(device) as t:
        page_emb = encode(model, processor, images, "images", config["batch_size"])
    timings["encode_pages_s"] = t.seconds
    timings["encode_per_page_s"] = t.seconds / len(images)
    memory.sample()

    with Timer(device) as t:
        query_emb = encode(model, processor, queries, "queries", config["batch_size"])
    timings["encode_queries_s"] = t.seconds
    timings["encode_per_query_s"] = t.seconds / len(queries)
    memory.sample()

    with Timer(device) as t:
        scores = processor.score_multi_vector(query_emb, page_emb).float()
    timings["scoring_s"] = t.seconds

    top1 = scores.argmax(dim=1).tolist()
    hits = [int(p == g) for p, g in zip(top1, gold)]

    print("\nscores (rows = queries, cols = pages):")
    print(np.array2string(scores.numpy(), precision=2, suppress_small=True))
    print(f"\ntop-1 sanity check: {sum(hits)}/{len(hits)} queries rank their own page first")
    print(f"timings: {json.dumps({k: round(v, 3) for k, v in timings.items()})}")
    peak = memory.report()
    print(f"peak memory: {peak}")

    record = collect_metadata(config, args.config, device, config["seed"])
    record.update(
        {
            "resolved": {
                "dtype": str(dtype),
                "model_sha": hub_revision(mcfg["name"], mcfg["revision"], "model"),
                "dataset_sha": hub_revision(dcfg["dataset"], dcfg["revision"], "dataset"),
            },
            "n_queries": len(queries),
            "n_pages": len(images),
            "page_embedding_shape": list(page_emb[0].shape),
            "query_embedding_shapes": [list(q.shape) for q in query_emb],
            "timings": timings,
            "peak_memory": peak,
            "top1_hits": hits,
            "top1_accuracy": sum(hits) / len(hits),
            "queries": queries,
            "gold_page_index": gold,
            "scores": scores.tolist(),
        }
    )

    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{device}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
