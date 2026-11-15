"""POST-HOC set 2, dev split only: select T_hyb's RRF k, then N for the two reranked systems.

Stages (run in this order; the reranker runs between them):
    python scripts/run_posthoc2_dev.py --stage hybrid   # k for T_hyb = RRF(BM25, BGE-M3)
    python scripts/run_reranker.py                      # cross-encoder scores over the candidate pool
    python scripts/run_posthoc2_dev.py --stage rerank   # N for T_rr and B_ce

Never evaluates the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores, select_k  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run, subset  # noqa: E402
from src.fusion import rrf  # noqa: E402
from src.rerank import rerank_top_n  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402


def load_ce(path: Path, queries: list[str], page_ids: list[str]) -> np.ndarray:
    saved = np.load(path, allow_pickle=False)
    if list(saved["queries"]) != queries or list(saved["page_ids"]) != page_ids:
        raise SystemExit(f"{path}: query/page order does not match the dataset")
    return saved["scores"].astype(np.float64)


def hybrid_scores(config: dict, base: dict, page_ids: list[str], k: float) -> np.ndarray:
    return rrf([base[i] for i in config["hybrid"]["inputs"]], page_ids, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/posthoc2_docvqa.yaml")
    parser.add_argument("--stage", choices=["hybrid", "rerank"], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    dcfg = config["data"]

    task = load_task(dcfg["dataset"], dcfg["revision"], dcfg["split"])
    dev = json.loads((REPO_ROOT / config["splits"]).read_text())["dev"]
    base = {k: load_scores(p, task.queries, task.page_ids) for k, p in config["scores"].items()}
    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def dev_metrics(scores: np.ndarray) -> dict:
        return evaluate(*subset(task.qrels, scores_to_run(scores, task.queries, task.page_ids), dev))[0]

    meta = collect_metadata(config, args.config, "cpu", config["seed"])
    meta["label"] = "POST-HOC set 2 (exploratory), dev split only"
    meta["dev_n_queries"] = len(dev)
    meta["baselines_dev"] = {sid: dev_metrics(s) for sid, s in base.items()}

    if args.stage == "hybrid":
        hcfg = config["hybrid"]
        sweep, sweep_metrics = {}, {}
        for k in hcfg["k_grid"]:
            m = dev_metrics(hybrid_scores(config, base, task.page_ids, k))
            sweep[k], sweep_metrics[str(k)] = m[hcfg["selection_metric"]], m
        k_sel = select_k(sweep, hcfg["tie_break"])
        print("T_hyb = RRF(BM25, BGE-M3) dev nDCG@5: " + "  ".join(f"k={k}: {v:.4f}" for k, v in sweep.items()))
        print(f"  -> selected k = {k_sel}")
        for sid in ("A", "BGE", "B"):
            print(f"{sid} dev nDCG@5: {meta['baselines_dev'][sid]['ndcg_at_5']:.4f}")
        meta["systems"] = {"T_hyb": {"name": "T_hyb: RRF(BM25, BGE-M3)", "selected_k": k_sel, "dev_sweep": sweep_metrics}}
        (out_dir / "dev_selection_hybrid.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote {out_dir}/dev_selection_hybrid.json")
        return

    # stage == rerank
    k_sel = json.loads((out_dir / "dev_selection_hybrid.json").read_text())["systems"]["T_hyb"]["selected_k"]
    firsts = {"T_hyb": hybrid_scores(config, base, task.page_ids, k_sel), **base}
    ce = load_ce(REPO_ROOT / config["reranker_output_dir"] / "ce_scores.npz", task.queries, task.page_ids)
    rcfg = config["rerank"]
    meta["systems"] = {}
    for sid, scfg in rcfg["systems"].items():
        first = firsts[scfg["first"]]
        sweep, sweep_metrics = {}, {}
        for n in rcfg["n_grid"]:
            m = dev_metrics(rerank_top_n(first, ce, n, task.page_ids))
            sweep[n], sweep_metrics[str(n)] = m[rcfg["selection_metric"]], m
        best = max(sweep.values())
        n_sel = min(n for n, v in sweep.items() if v == best)  # ties -> smaller N
        meta["systems"][sid] = {"name": scfg["name"], "first": scfg["first"], "selected_n": n_sel,
                                "dev_sweep": sweep_metrics}
        print(f"{sid} dev nDCG@5: " + "  ".join(f"N={n}: {v:.4f}" for n, v in sweep.items()) + f"\n  -> selected N = {n_sel}")
    meta["T_hyb_k"] = k_sel
    (out_dir / "dev_selection_rerank.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir}/dev_selection_rerank.json")


if __name__ == "__main__":
    main()
