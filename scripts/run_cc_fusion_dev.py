"""Post-hoc: choose alpha of the score-level convex combination for C_cc and D_cc on the dev split only.

fused = alpha * minmax(ColPali) + (1 - alpha) * minmax(text), per query. Same grid and rule for both systems.
Writes the dev curve and the selected alpha per system. Never evaluates test.

Usage:
    python scripts/run_cc_fusion_dev.py --config configs/fusion_cc_docvqa.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run, subset  # noqa: E402
from src.fusion import convex_combination  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402


def select(curve: dict[float, float]) -> float:
    best = max(curve.values())
    return min((a for a, v in curve.items() if v == best), key=lambda a: (abs(a - 0.5), a))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/fusion_cc_docvqa.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    dcfg, gcfg = config["data"], config["convex_combination"]

    task = load_task(dcfg["dataset"], dcfg["revision"], dcfg["split"])
    dev = json.loads((REPO_ROOT / config["splits"]).read_text())["dev"]
    colpali = load_scores(config["colpali_scores"], task.queries, task.page_ids)
    metric = gcfg["selection_metric"]

    def dev_score(scores) -> float:
        return evaluate(*subset(task.qrels, scores_to_run(scores, task.queries, task.page_ids), dev))[0][metric]

    record = {"dev_n_queries": len(dev), "B_ColPali_dev": dev_score(colpali), "systems": {}}
    for sid, scfg in config["systems"].items():
        text = load_scores(scfg["text_scores"], task.queries, task.page_ids)
        curve = {a: dev_score(convex_combination(colpali, text, a)) for a in gcfg["alpha_grid"]}
        a_sel = select(curve)
        record["systems"][sid] = {"name": scfg["name"], "selected_alpha": a_sel, "dev_" + metric: curve[a_sel],
                                  "dev_curve": {str(a): v for a, v in curve.items()}}
        print(f"\n{sid} {scfg['name']} — dev {metric} by alpha (weight on ColPali)")
        for a, v in curve.items():
            print(f"  alpha={a:<5} {v:.4f}" + ("   <- selected" if a == a_sel else ""))

    print(f"\nB ColPali dev {metric}: {record['B_ColPali_dev']:.4f}")
    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = collect_metadata(config, args.config, "cpu", 0)
    meta.update(record)
    (out_dir / "dev_selection_cc.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir}/dev_selection_cc.json")


if __name__ == "__main__":
    main()
