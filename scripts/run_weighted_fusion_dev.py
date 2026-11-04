"""Ablation E: choose (w, k) of weighted RRF for C' and D' on the dev split only.

fused = w / (k + rank_ColPali) + (1 - w) / (k + rank_text). Same grid and rule for both systems.
Writes the full dev surface (w x k) and the selected configuration per system. Never evaluates test.

Usage:
    python scripts/run_weighted_fusion_dev.py --config configs/fusion_weighted_docvqa.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run, subset  # noqa: E402
from src.fusion import weighted_rrf  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402


def select(surface: dict[tuple[float, float], float]) -> tuple[float, float]:
    best = max(surface.values())
    ties = [wk for wk, v in surface.items() if v == best]
    return min(ties, key=lambda wk: (abs(wk[0] - 0.5), abs(wk[1] - 60), wk[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/fusion_weighted_docvqa.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    dcfg, gcfg = config["data"], config["weighted_rrf"]

    task = load_task(dcfg["dataset"], dcfg["revision"], dcfg["split"])
    dev = json.loads((REPO_ROOT / config["splits"]).read_text())["dev"]
    colpali = load_scores(config["colpali_scores"], task.queries, task.page_ids)
    metric = gcfg["selection_metric"]

    def dev_score(scores: np.ndarray) -> float:
        run = scores_to_run(scores, task.queries, task.page_ids)
        return evaluate(*subset(task.qrels, run, dev))[0][metric]

    record = {"dev_n_queries": len(dev), "B_ColPali_dev": dev_score(colpali), "systems": {}}
    for sid, scfg in config["systems"].items():
        text = load_scores(scfg["text_scores"], task.queries, task.page_ids)
        surface = {}
        for w in gcfg["w_grid"]:
            for k in gcfg["k_grid"]:
                surface[(w, k)] = dev_score(weighted_rrf([colpali, text], [w, 1 - w], task.page_ids, k=k))
        w_sel, k_sel = select(surface)
        record["systems"][sid] = {
            "name": scfg["name"], "selected_w": w_sel, "selected_k": k_sel,
            "dev_" + metric: surface[(w_sel, k_sel)],
            "dev_surface": {f"w={w}": {str(k): surface[(w, k)] for k in gcfg["k_grid"]} for w in gcfg["w_grid"]},
        }
        print(f"\n{sid} {scfg['name']} — dev {metric} (rows w, columns k)")
        print("w\\k   " + "".join(f"{k:>8}" for k in gcfg["k_grid"]))
        for w in gcfg["w_grid"]:
            print(f"{w:<6}" + "".join(f"{surface[(w, k)]:8.4f}" for k in gcfg["k_grid"]))
        print(f"  -> selected w = {w_sel}, k = {k_sel}: {surface[(w_sel, k_sel)]:.4f}")

    print(f"\nB ColPali dev {metric}: {record['B_ColPali_dev']:.4f}")
    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = collect_metadata(config, args.config, "cpu", 0)
    meta.update(record)
    (out_dir / "dev_selection_weighted.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir}/dev_selection_weighted.json")


if __name__ == "__main__":
    main()
