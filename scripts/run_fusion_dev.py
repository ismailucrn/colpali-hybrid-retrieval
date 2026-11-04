"""Choose the RRF k for systems C and D on the dev split only (equal tuning budget for both).

For every k in the grid, fuses the text retriever's scores with ColPali's and evaluates on dev.
Writes the full dev sweep (this is also ablation E on dev) and the selected k per system.
Never evaluates the test split.

Usage:
    python scripts/run_fusion_dev.py --config configs/fusion_docvqa.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run, subset  # noqa: E402
from src.fusion import rrf  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402


def load_scores(path: str, queries: list[str], page_ids: list[str]) -> np.ndarray:
    saved = np.load(REPO_ROOT / path, allow_pickle=False)
    if list(saved["queries"]) != queries or list(saved["page_ids"]) != page_ids:
        raise SystemExit(f"{path}: query/page order does not match the dataset")
    return saved["scores"].astype(np.float64)


def select_k(sweep: dict[float, float], tie_break: str) -> float:
    best = max(sweep.values())
    candidates = [k for k, v in sweep.items() if v == best]
    if tie_break == "closest_to_60":
        return min(candidates, key=lambda k: (abs(k - 60), k))
    return min(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/fusion_docvqa.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    dcfg, rcfg = config["data"], config["rrf"]

    task = load_task(dcfg["dataset"], dcfg["revision"], dcfg["split"])
    dev = json.loads((REPO_ROOT / config["splits"]).read_text())["dev"]
    colpali = load_scores(config["colpali_scores"], task.queries, task.page_ids)

    def dev_metrics(scores: np.ndarray) -> dict:
        run = scores_to_run(scores, task.queries, task.page_ids)
        return evaluate(*subset(task.qrels, run, dev))[0]

    record = {"dev_n_queries": len(dev), "baselines_dev": {"B: ColPali": dev_metrics(colpali)}, "systems": {}}
    metric = rcfg["selection_metric"]
    for sid, scfg in config["systems"].items():
        text = load_scores(scfg["text_scores"], task.queries, task.page_ids)
        record["baselines_dev"][f"{sid} text component"] = dev_metrics(text)
        sweep, sweep_metrics = {}, {}
        for k in rcfg["k_grid"]:
            m = dev_metrics(rrf([text, colpali], task.page_ids, k=k))
            sweep[k], sweep_metrics[str(k)] = m[metric], m
        k_sel = select_k(sweep, rcfg["tie_break"])
        record["systems"][sid] = {"name": scfg["name"], "selected_k": k_sel, "dev_sweep": sweep_metrics}
        line = "  ".join(f"k={k}: {v:.4f}" for k, v in sweep.items())
        print(f"{sid} {scfg['name']} dev {metric}: {line}\n  -> selected k = {k_sel}")

    print(f"\nB ColPali dev {metric}: {record['baselines_dev']['B: ColPali'][metric]:.4f}")
    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = collect_metadata(config, args.config, "cpu", 0)
    meta.update(record)
    (out_dir / "dev_selection.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir}/dev_selection.json")


if __name__ == "__main__":
    main()
