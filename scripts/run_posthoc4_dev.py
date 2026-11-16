"""POST-HOC set 4, dev split only (pre-registered 2026-11-16): settings for the stronger-model systems, and the
frozen candidate for the InfoVQA confirmation.

Stages:
    python scripts/run_posthoc4_dev.py --stage retrievers   # CQ, TM, Q8 alone; k for F_cq, F_tm, F_q8
    python scripts/run_posthoc4_dev.py --stage rerank8b     # N for B_vr8
    python scripts/run_posthoc4_dev.py --stage candidate    # highest dev nDCG@5 among the set-3 and set-4 systems

Candidate pool (clarified before any set-4 dev result was seen): every system in the Holm families of sets 3 and 4,
i.e. B_vr, C_sel, C_vlm, E_ens, CQ, F_cq, TM, F_tm, F_q8, B_vr8. Ties -> lower query latency (median, ms).
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

RETRIEVERS = {"CQ": "colqwen25_docvqa", "TM": "tomoro_docvqa", "Q8": "qwen3vl_embedding8b_docvqa"}
FUSED = {"F_cq": "CQ", "F_tm": "TM", "F_q8": "Q8"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/posthoc4_docvqa.yaml")
    parser.add_argument("--stage", choices=["retrievers", "rerank8b", "candidate"], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    d = config["data"]
    task = load_task(d["dataset"], d["revision"], d["split"])
    dev = json.loads((REPO_ROOT / config["splits"]).read_text())["dev"]
    B = load_scores(config["scores"]["B"], task.queries, task.page_ids)
    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def dev_metrics(scores: np.ndarray) -> dict:
        return evaluate(*subset(task.qrels, scores_to_run(scores, task.queries, task.page_ids), dev))[0]

    meta = collect_metadata(config, args.config, "cpu", config["seed"])
    meta["label"] = "POST-HOC set 4 (exploratory, pre-registered 2026-11-16), dev split only"
    meta["B_dev_ndcg_at_5"] = dev_metrics(B)["ndcg_at_5"]
    print(f"B (ColPali) dev nDCG@5 = {meta['B_dev_ndcg_at_5']:.4f}")

    if args.stage == "retrievers":
        S = {k: load_scores(f"results/{v}/scores.npz", task.queries, task.page_ids) for k, v in RETRIEVERS.items()}
        meta["single"] = {k: dev_metrics(v)["ndcg_at_5"] for k, v in S.items()}
        for k, v in meta["single"].items():
            print(f"{k} alone: dev nDCG@5 {v:.4f}")
        meta["systems"] = {}
        for fid, comp in FUSED.items():
            sweep = {k: dev_metrics(rrf([B, S[comp]], task.page_ids, k=k))["ndcg_at_5"] for k in config["rrf"]["k_grid"]}
            k_sel = select_k(sweep, config["rrf"]["tie_break"])
            meta["systems"][fid] = {"inputs": ["B", comp], "selected_k": k_sel, "dev_ndcg_at_5": sweep[k_sel],
                                    "dev_ndcg_at_5_by_k": {str(k): v for k, v in sweep.items()}}
            print(f"{fid} = RRF(B, {comp}): k = {k_sel}, dev {sweep[k_sel]:.4f}  "
                  + " ".join(f"{k}:{v:.4f}" for k, v in sweep.items()))
    elif args.stage == "rerank8b":
        r = config["reranker8b"]
        V = load_scores(f"{r['output_dirs']['docvqa']}/scores.npz", task.queries, task.page_ids)
        nd = {n: dev_metrics(rerank_top_n(B, V, n, task.page_ids))["ndcg_at_5"] for n in r["n_grid"]}
        n_sel = min(n for n, v in nd.items() if v == max(nd.values()))
        for n, v in nd.items():
            print(f"B_vr8 N={n:<3} dev nDCG@5 {v:.4f}{'  <- selected' if n == n_sel else ''}")
        meta["systems"] = {"B_vr8": {"first": "B", "selected_n": n_sel, "dev_ndcg_at_5": nd[n_sel],
                                     "dev_ndcg_at_5_by_n": {str(n): v for n, v in nd.items()}}}
    else:
        p3 = {k: json.loads((REPO_ROOT / f"results/posthoc3/dev_selection_{k}.json").read_text())
              for k in ("selective", "rerank", "ocr", "embed")}
        p4r = json.loads((out_dir / "dev_selection_retrievers.json").read_text())
        p4v = json.loads((out_dir / "dev_selection_rerank8b.json").read_text())
        sel = p3["selective"]["systems"]["C_sel"]
        tau_q = next(q for q, x in sel["sweep"].items() if x["tau"] == sel["selected_tau"])
        lat = lambda path: json.loads((REPO_ROOT / path).read_text())  # noqa: E731
        colpali_ms = lat("results/colpali_v1.3_docvqa/metrics.json")["latency"]["total"]["median_ms"]
        vr2 = lat("results/visual_reranker_docvqa/metrics.json")["latency_by_n"]
        vr8 = lat("results/visual_reranker8b_docvqa/metrics.json")["latency_by_n"]
        mv_ms = {k: lat(f"results/{v}/metrics.json")["latency"]["median_ms"] for k, v in RETRIEVERS.items()}
        n2, n8 = p3["rerank"]["systems"]["B_vr"]["selected_n"], p4v["systems"]["B_vr8"]["selected_n"]
        pool = {
            "B_vr": (p3["rerank"]["systems"]["B_vr"]["dev_ndcg_at_5_by_n"][str(n2)], colpali_ms + vr2.get(str(n2), {"median_ms": 0})["median_ms"]),
            "C_sel": (sel["sweep"][tau_q]["dev_ndcg_at_5"], colpali_ms),
            "C_vlm": (p3["ocr"]["systems"]["C_vlm"]["selected_dev_metrics"]["ndcg_at_5"], colpali_ms),
            "E_ens": (p3["embed"]["systems"]["E_ens"]["selected_dev_metrics"]["ndcg_at_5"],
                      colpali_ms + lat("results/qwen3vl_embedding_docvqa/metrics.json")["latency"]["median_ms"]),
            "CQ": (p4r["single"]["CQ"], mv_ms["CQ"]),
            "TM": (p4r["single"]["TM"], mv_ms["TM"]),
            "F_cq": (p4r["systems"]["F_cq"]["dev_ndcg_at_5"], colpali_ms + mv_ms["CQ"]),
            "F_tm": (p4r["systems"]["F_tm"]["dev_ndcg_at_5"], colpali_ms + mv_ms["TM"]),
            "F_q8": (p4r["systems"]["F_q8"]["dev_ndcg_at_5"], colpali_ms + mv_ms["Q8"]),
            "B_vr8": (p4v["systems"]["B_vr8"]["dev_ndcg_at_5"], colpali_ms + (vr8[str(n8)]["median_ms"] if n8 > 1 else 0)),
        }
        best = max(v[0] for v in pool.values())
        cand = min((k for k, v in pool.items() if v[0] == best), key=lambda k: pool[k][1])
        for k, (nd, ms) in sorted(pool.items(), key=lambda x: -x[1][0]):
            print(f"  {k:<6} dev nDCG@5 {nd:.4f}  latency ~{ms:.0f} ms{'  <- candidate' if k == cand else ''}")
        meta["candidate"] = cand
        meta["pool"] = {k: {"dev_ndcg_at_5": v[0], "latency_sequential_ms_approx": v[1]} for k, v in pool.items()}
        meta["note"] = "latency: sequential sum of component medians (ColPali on L4 + the second component)"
    path = out_dir / f"dev_selection_{args.stage}.json"
    path.write_text(json.dumps(meta, indent=2, default=float))
    print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
