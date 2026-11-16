"""Post-hoc set 4, confirmation on a fresh task (pre-registered 2026-11-16): the frozen DocVQA-dev candidate vs
ColPali v1.3 on the full InfoVQA query set, official protocol. One paired comparison; no setting is tuned here.

The candidate and its setting are read from results/posthoc4/dev_selection_candidate.json and the DocVQA dev
selections; this script refuses to run if the needed InfoVQA score files are missing.
Writes results/posthoc4/fresh_task_infovqa.json and per_query_infovqa.csv.

Usage:
    python scripts/evaluate_fresh_task.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run  # noqa: E402
from src.fusion import rrf  # noqa: E402
from src.rerank import rerank_top_n  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402
from src.stats import bootstrap_ci, paired_bootstrap_ci, paired_permutation_test  # noqa: E402

SCORE_DIRS = {"B": "colpali_v1.3_infovqa", "CQ": "colqwen25_infovqa", "TM": "tomoro_infovqa",
              "Q8": "qwen3vl_embedding8b_infovqa", "VR8": "visual_reranker8b_infovqa"}


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs/posthoc4_docvqa.yaml").read_text())
    tk = cfg["tasks"][cfg["fresh_task"]["task"]]
    task = load_task(tk["dataset"], tk["revision"], tk["split"])
    p4 = REPO_ROOT / "results/posthoc4"
    cand = json.loads((p4 / "dev_selection_candidate.json").read_text())["candidate"]
    ret = json.loads((p4 / "dev_selection_retrievers.json").read_text())["systems"]
    rr8 = json.loads((p4 / "dev_selection_rerank8b.json").read_text())["systems"]["B_vr8"]

    def scores(key: str) -> np.ndarray:
        return load_scores(f"results/{SCORE_DIRS[key]}/scores.npz", task.queries, task.page_ids)

    B = scores("B")
    if cand in ("CQ", "TM"):
        C, setting = scores(cand), "alone"
    elif cand in ("F_cq", "F_tm", "F_q8"):
        comp = ret[cand]["inputs"][1]
        C, setting = rrf([B, scores(comp)], task.page_ids, k=ret[cand]["selected_k"]), f"RRF k={ret[cand]['selected_k']}"
    elif cand == "B_vr8":
        C, setting = rerank_top_n(B, scores("VR8"), rr8["selected_n"], task.page_ids), f"N={rr8['selected_n']}"
    else:
        raise SystemExit(f"candidate {cand}: InfoVQA inputs for set-3 systems were not computed; extend the GPU run first")

    pb = evaluate(task.qrels, scores_to_run(B, task.queries, task.page_ids))
    pc = evaluate(task.qrels, scores_to_run(C, task.queries, task.page_ids))
    a = np.array([pc[1][q]["ndcg_at_5"] for q in task.queries])
    b = np.array([pb[1][q]["ndcg_at_5"] for q in task.queries])
    res = {
        "task": tk, "n_queries": len(task.queries), "n_pages": len(task.page_ids), "candidate": cand, "setting": setting,
        "B": {"metrics": pb[0], "ndcg_at_5_ci95": bootstrap_ci(b)},
        "candidate_metrics": {"metrics": pc[0], "ndcg_at_5_ci95": bootstrap_ci(a)},
        "delta": float(a.mean() - b.mean()), "delta_ci95": paired_bootstrap_ci(a, b),
        "p_permutation": paired_permutation_test(a, b),
        "wins": int((a > b).sum()), "ties": int((a == b).sum()), "losses": int((a < b).sum()),
        "published_colpali_v1_3_ndcg_at_5": cfg["fresh_task"]["published_colpali_v1_3_ndcg_at_5"],
    }
    record = collect_metadata(cfg, "configs/posthoc4_docvqa.yaml", "cpu", 0)
    record["label"] = "POST-HOC set 4: confirmation on a fresh task (pre-registered), one comparison, evaluated once"
    record.update(res)
    (p4 / "fresh_task_infovqa.json").write_text(json.dumps(record, indent=2, default=float))
    with open(p4 / "per_query_infovqa.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "B_ndcg_at_5", "B_gold_rank", f"{cand}_ndcg_at_5", f"{cand}_gold_rank"])
        for q in task.queries:
            w.writerow([q, round(pb[1][q]["ndcg_at_5"], 6), pb[1][q]["gold_rank"], round(pc[1][q]["ndcg_at_5"], 6),
                        pc[1][q]["gold_rank"]])
    print(f"InfoVQA ({len(task.queries)} queries): ColPali {pb[0]['ndcg_at_5']:.4f} (published {res['published_colpali_v1_3_ndcg_at_5']}); "
          f"{cand} ({setting}) {pc[0]['ndcg_at_5']:.4f}; delta {res['delta']:+.4f} "
          f"[{res['delta_ci95'][0]:+.4f}, {res['delta_ci95'][1]:+.4f}], p = {res['p_permutation']:.4f}, "
          f"W/T/L {res['wins']}/{res['ties']}/{res['losses']}")


if __name__ == "__main__":
    main()
