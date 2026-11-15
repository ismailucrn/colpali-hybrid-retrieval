"""POST-HOC set 3, dev split only: select the setting of each system (pre-registered in e70f2a7).

Stages (each needs its inputs; the GPU ones come from scripts/run_posthoc3_gpu.py on Colab):
    python scripts/run_posthoc3_dev.py --stage selective   # H2: tau for C_sel (CPU only)
    python scripts/run_posthoc3_dev.py --stage rerank      # H1: N for B_vr
    python scripts/run_posthoc3_dev.py --stage ocr         # H3: k for C_vlm (needs results/bm25_vlm_docvqa/scores.npz)
    python scripts/run_posthoc3_dev.py --stage embed       # H4: k for E_ens

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
from src.selective import gated, relative_margin  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/posthoc3_docvqa.yaml")
    parser.add_argument("--stage", choices=["selective", "rerank", "ocr", "embed"], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    dcfg = config["data"]

    task = load_task(dcfg["dataset"], dcfg["revision"], dcfg["split"])
    dev = json.loads((REPO_ROOT / config["splits"]).read_text())["dev"]
    dev_idx = [task.queries.index(q) for q in dev]
    base = {k: load_scores(p, task.queries, task.page_ids) for k, p in config["scores"].items()}
    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def dev_metrics(scores: np.ndarray) -> dict:
        return evaluate(*subset(task.qrels, scores_to_run(scores, task.queries, task.page_ids), dev))[0]

    meta = collect_metadata(config, args.config, "cpu", config["seed"])
    meta["label"] = "POST-HOC set 3 (exploratory, pre-registered in e70f2a7), dev split only"
    meta["dev_n_queries"] = len(dev)
    b_dev = dev_metrics(base["B"])["ndcg_at_5"]
    meta["B_dev_ndcg_at_5"] = b_dev
    print(f"B (ColPali) dev nDCG@5 = {b_dev:.4f}")

    def k_sweep(name: str, other: np.ndarray) -> dict:
        sweep = {k: dev_metrics(rrf([other, base["B"]], task.page_ids, k=k)) for k in config["rrf"]["k_grid"]}
        nd = {k: m["ndcg_at_5"] for k, m in sweep.items()}
        k_sel = select_k(nd, config["rrf"]["tie_break"])
        for k, v in nd.items():
            print(f"  {name} k={k:<4} dev nDCG@5 {v:.4f}{'  <- selected' if k == k_sel else ''}")
        return {"selected_k": k_sel, "dev_ndcg_at_5_by_k": {str(k): v for k, v in nd.items()},
                "selected_dev_metrics": sweep[k_sel]}

    if args.stage == "selective":
        s = config["selective"]
        fused = rrf([base[i] for i in s["fused"]["inputs"]], task.page_ids, k=s["fused"]["k"])
        conf = relative_margin(base[s["confidence_of"]])
        taus = [float(np.quantile(conf[dev_idx], q)) for q in s["tau_quantiles"]] + [float("inf")]
        res = {}
        for q, tau in zip(s["tau_quantiles"] + ["inf"], taus):
            m = dev_metrics(gated(fused, base["B"], conf, tau))
            n_fused = int((conf[dev_idx] >= tau).sum())
            res[str(q)] = {"tau": tau, "dev_queries_fused": n_fused, "dev_ndcg_at_5": m["ndcg_at_5"]}
            print(f"  quantile {q!s:<5} tau={tau:.4f} fused on {n_fused:3d} dev queries: nDCG@5 {m['ndcg_at_5']:.4f}")
        best = max(r["dev_ndcg_at_5"] for r in res.values())
        sel = max((r for r in res.values() if r["dev_ndcg_at_5"] == best), key=lambda r: r["tau"])  # larger tau
        print(f"selected tau = {sel['tau']} (dev nDCG@5 {best:.4f})")
        meta["systems"] = {"C_sel": {"selected_tau": sel["tau"], "fused_k": s["fused"]["k"], "sweep": res}}
    elif args.stage == "rerank":
        r = config["reranker"]
        vr = load_scores(f"{r['output_dir']}/scores.npz", task.queries, task.page_ids)
        sweep = {n: dev_metrics(rerank_top_n(base[r["pool_first"]], vr, n, task.page_ids)) for n in r["n_grid"]}
        nd = {n: m["ndcg_at_5"] for n, m in sweep.items()}
        n_sel = min(n for n, v in nd.items() if v == max(nd.values()))  # ties: smaller N
        for n, v in nd.items():
            print(f"  B_vr N={n:<3} dev nDCG@5 {v:.4f}{'  <- selected' if n == n_sel else ''}")
        meta["systems"] = {"B_vr": {"first": r["pool_first"], "selected_n": n_sel,
                                    "dev_ndcg_at_5_by_n": {str(n): v for n, v in nd.items()},
                                    "selected_dev_metrics": sweep[n_sel]}}
    elif args.stage == "ocr":
        a_vlm = load_scores(f"{config['ocr']['bm25_output_dir']}/scores.npz", task.queries, task.page_ids)
        meta["A_vlm_dev_ndcg_at_5"] = dev_metrics(a_vlm)["ndcg_at_5"]
        meta["A_dev_ndcg_at_5"] = dev_metrics(base["A"])["ndcg_at_5"]
        print(f"A (BM25, Tesseract) dev {meta['A_dev_ndcg_at_5']:.4f}; A_vlm dev {meta['A_vlm_dev_ndcg_at_5']:.4f}")
        meta["systems"] = {"C_vlm": {"inputs": ["A_vlm", "B"], **k_sweep("C_vlm", a_vlm)}}
    else:
        q3e = load_scores(f"{config['embedding']['output_dir']}/scores.npz", task.queries, task.page_ids)
        meta["Q3E_dev_ndcg_at_5"] = dev_metrics(q3e)["ndcg_at_5"]
        print(f"Q3E (Qwen3-VL-Embedding-2B) dev {meta['Q3E_dev_ndcg_at_5']:.4f}")
        meta["systems"] = {"E_ens": {"inputs": ["Q3E", "B"], **k_sweep("E_ens", q3e)}}

    path = out_dir / f"dev_selection_{args.stage}.json"
    path.write_text(json.dumps(meta, indent=2, default=float))
    print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
