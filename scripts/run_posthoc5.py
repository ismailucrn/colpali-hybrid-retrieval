"""POST-HOC set 5, CPU stages (pre-registered 2026-11-16 in docs/experiment_log.md, `265d3f9`).

Stages:
    python scripts/run_posthoc5.py --stage bm25     # A = BM25 on the authors' Tesseract text, nine tasks
    python scripts/run_posthoc5.py --stage fusion   # frozen C / C_w vs B per task, Holm over 9; headroom; Spearman
    python scripts/run_posthoc5.py --stage router   # leave-one-task-out router B vs C over the nine tasks + DocVQA

Nothing is tuned on these tasks. Rows sharing an image filename are one page, max over rows (official evaluator).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.bm25 import BM25Index, Tokenizer  # noqa: E402
from src.data import load_task_columns, max_over_rows  # noqa: E402
from src.evaluation import evaluate, scores_to_run  # noqa: E402
from src.fusion import ranks_from_scores, rrf, weighted_rrf  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402
from src.stats import bootstrap_ci, holm, paired_bootstrap_ci, paired_permutation_test  # noqa: E402

CONFIG = "configs/posthoc5_vidore.yaml"


def per_query(task, scores: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    agg, pq = evaluate(task.qrels, scores_to_run(scores, task.queries, task.page_ids))
    return agg, np.array([pq[q]["ndcg_at_5"] for q in task.queries]), np.array([pq[q]["gold_rank"] for q in task.queries])


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    return {"delta": float(a.mean() - b.mean()), "delta_ci95": paired_bootstrap_ci(a, b), "p": paired_permutation_test(a, b),
            "wins": int((a > b).sum()), "ties": int((a == b).sum()), "losses": int((a < b).sum())}


def fuse(spec: dict, S: dict, page_ids: list[str]) -> np.ndarray:
    mats = [S[x] for x in spec["inputs"]]
    if spec["kind"] == "rrf":
        return rrf(mats, page_ids, k=spec["k"])
    return weighted_rrf(mats, spec["weights"], page_ids, k=spec["k"])


def spearman_exact(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman rho with an exact two-sided permutation p (all n! orderings; n = 9 -> 362,880)."""
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n_ge = n = 0
    for perm in itertools.permutations(ry):
        r = np.corrcoef(rx, perm)[0, 1]
        n += 1
        n_ge += abs(r) >= abs(rho) - 1e-12
    return rho, n_ge / n


def stage_bm25(cfg: dict) -> None:
    tok = Tokenizer(cfg["bm25"]["tokenizer"])
    for name, tk in cfg["tasks"].items():
        t0 = time.perf_counter()
        task = load_task_columns(tk["base"], tk["base_rev"], "test", tk["text"], tk["text_rev"])
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        index = BM25Index(task.row_texts, tok)
        index_s = time.perf_counter() - t0
        rows = index.score_matrix(task.queries)
        scores = max_over_rows(rows, task.row_page, len(task.page_ids))
        lat = []
        for q in task.queries[:100]:
            t0 = time.perf_counter()
            max_over_rows(np.asarray(index.score(q))[None, :], task.row_page, len(task.page_ids))
            lat.append(time.perf_counter() - t0)
        out = REPO_ROOT / cfg["bm25"]["output_dir"].format(task=name)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "scores.npz", scores=scores.astype(np.float32), queries=np.array(task.queries, dtype=str),
                            page_ids=np.array(task.page_ids, dtype=str))
        agg, _, _ = per_query(task, scores)
        rec = collect_metadata(cfg, CONFIG, "cpu", cfg["seed"])
        rec.update({"label": "POST-HOC set 5 (pre-registered 2026-11-16): A = BM25, official tokenizer", "task": name,
                    "dataset": {"base": tk["base"], "base_rev": tk["base_rev"], "text": tk["text"], "text_rev": tk["text_rev"]},
                    "n_rows": len(task.row_page), "n_pages": len(task.page_ids), "n_queries": len(task.queries),
                    "n_empty_rows": sum(not x.strip() for x in task.row_texts), "metrics": agg,
                    "timings": {"load_s": load_s, "index_rows_s": index_s},
                    "latency": {"n": len(lat), "median_ms": float(np.median(lat) * 1000),
                                "note": "one query vs all rows + max-merge, CPU (Apple M5)"}})
        (out / "metrics.json").write_text(json.dumps(rec, indent=2))
        print(f"{name:<13} rows {len(task.row_page):>5} pages {len(task.page_ids):>5} queries {len(task.queries):>5}  "
              f"empty {rec['n_empty_rows']:>3}  BM25 nDCG@5 {agg['ndcg_at_5']:.4f}")


def load_task_scores(cfg: dict, name: str):
    tk = cfg["tasks"][name]
    task = load_task_columns(tk["base"], tk["base_rev"], "test")
    S = {"A": load_scores(cfg["bm25"]["output_dir"].format(task=name) + "/scores.npz", task.queries, task.page_ids),
         "B": load_scores(cfg["colpali"]["output_dir"].format(task=name) + "/scores.npz", task.queries, task.page_ids)}
    return task, S


def stage_fusion(cfg: dict) -> None:
    out_dir = REPO_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    res, p_c, p_cw = {}, {}, {}
    for name, tk in cfg["tasks"].items():
        task, S = load_task_scores(cfg, name)
        S["C"] = fuse(cfg["fusion"]["C"], S, task.page_ids)
        S["C_w"] = fuse(cfg["fusion"]["C_w"], S, task.page_ids)
        m, nd, rk = {}, {}, {}
        for s in ("A", "B", "C", "C_w"):
            m[s], nd[s], rk[s] = per_query(task, S[s])
        sweep = {str(k): float(per_query(task, rrf([S["A"], S["B"]], task.page_ids, k=k))[0]["ndcg_at_5"])
                 for k in cfg["fusion"]["k_grid_descriptive"]}
        r = {
            "group": tk["group"], "n_queries": len(task.queries), "n_pages": len(task.page_ids),
            "metrics": {s: m[s] for s in m},
            "ndcg_at_5_ci95": {s: bootstrap_ci(nd[s]) for s in nd},
            "C_vs_B": compare(nd["C"], nd["B"]), "C_w_vs_B": compare(nd["C_w"], nd["B"]),
            "headroom_A_only_top5": float(((rk["A"] <= 5) & (rk["B"] > 5)).mean()),
            "B_only_top5": float(((rk["B"] <= 5) & (rk["A"] > 5)).mean()),
            "B_miss_top5": float((rk["B"] > 5).mean()),
            "oracle_B_or_C_ndcg_at_5": float(np.maximum(nd["B"], nd["C"]).mean()),
            "card_B_ndcg_at_5": tk["card"], "B_minus_card_pts": 100 * (m["B"]["ndcg_at_5"] - tk["card"]),
            "C_k_sweep_descriptive": sweep,
        }
        r["B_within_2pt_of_card"] = abs(r["B_minus_card_pts"]) <= 2
        res[name], p_c[name], p_cw[name] = r, r["C_vs_B"]["p"], r["C_w_vs_B"]["p"]
        with open(out_dir / f"per_query_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query"] + [f"{s}_{x}" for s in ("A", "B", "C", "C_w") for x in ("ndcg_at_5", "gold_rank")])
            for i, q in enumerate(task.queries):
                w.writerow([q] + [v for s in ("A", "B", "C", "C_w") for v in (round(float(nd[s][i]), 6), int(rk[s][i]))])
    for name, ph in holm(p_c).items():
        res[name]["C_vs_B"]["p_holm"] = ph
    for name, ph in holm(p_cw).items():
        res[name]["C_w_vs_B"]["p_holm"] = ph
    names = list(res)
    rho, p_rho = spearman_exact([res[n]["headroom_A_only_top5"] for n in names], [res[n]["C_vs_B"]["delta"] for n in names])
    group_mean = {g: float(np.mean([res[n]["C_vs_B"]["delta"] for n in names if res[n]["group"] == g])) for g in ("text", "visual")}
    summary = {"spearman_headroom_vs_delta_C": {"rho": rho, "p_exact_two_sided": p_rho, "n_tasks": len(names)},
               "mean_delta_C_by_group": group_mean}
    rec = collect_metadata(cfg, CONFIG, "cpu", cfg["seed"])
    rec.update({"label": "POST-HOC set 5 (pre-registered 2026-11-16): frozen DocVQA fusion settings on nine ViDoRe v1 tasks",
                "tasks": res, "summary": summary})
    (out_dir / "fusion.json").write_text(json.dumps(rec, indent=2, default=float))
    print(f"{'task':<13} {'grp':<6} {'n':>5} {'A':>6} {'B':>6} {'card':>6} {'C':>6} {'dC':>7} {'pH':>7} {'C_w':>6} {'dCw':>7} {'pH':>6} {'head':>5}")
    for n, r in res.items():
        m = r["metrics"]
        print(f"{n:<13} {r['group']:<6} {r['n_queries']:>5} {m['A']['ndcg_at_5']:.4f} {m['B']['ndcg_at_5']:.4f} {r['card_B_ndcg_at_5']:.4f} "
              f"{m['C']['ndcg_at_5']:.4f} {r['C_vs_B']['delta']:+.4f} {r['C_vs_B']['p_holm']:.4f} {m['C_w']['ndcg_at_5']:.4f} "
              f"{r['C_w_vs_B']['delta']:+.4f} {r['C_w_vs_B']['p_holm']:.3f} {r['headroom_A_only_top5']:.3f}")
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- router

def features(queries: list[str], A: np.ndarray, B: np.ndarray, page_ids: list[str], tok: Tokenizer) -> np.ndarray:
    def rel_margin(S):
        top = -np.sort(-S, axis=1)[:, :2]
        return np.where(top[:, 0] > 0, (top[:, 0] - top[:, 1]) / np.where(top[:, 0] > 0, top[:, 0], 1), 0.0)

    def top_z(S):
        sd = S.std(axis=1)
        return np.where(sd > 0, (S.max(axis=1) - S.mean(axis=1)) / np.where(sd > 0, sd, 1), 0.0)

    rA, rB = ranks_from_scores(A, page_ids), ranks_from_scores(B, page_ids)
    a_top1 = np.argmin(rA, axis=1)
    rank_in_b = rB[np.arange(len(queries)), a_top1]
    overlap = ((rA <= 5) & (rB <= 5)).sum(axis=1) / 5
    qlen = np.array([len(tok(q)) for q in queries])
    digit = np.array([np.mean([any(c.isdigit() for c in w) for w in q.split()]) if q.split() else 0.0 for q in queries])
    return np.column_stack([rel_margin(B), rel_margin(A), top_z(B), top_z(A), np.log(rank_in_b), overlap, np.log1p(qlen), digit])


def stage_router(cfg: dict) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tok = Tokenizer("official")
    data = {}
    names = list(cfg["tasks"]) + ["docvqa"]
    for name in names:
        if name == "docvqa":
            d = cfg["docvqa"]
            task = load_task_columns(d["base"], d["base_rev"], "test")
            S = {k: load_scores(d[k], task.queries, task.page_ids) for k in ("A", "B")}
        else:
            task, S = load_task_scores(cfg, name)
        C = fuse(cfg["fusion"]["C"], S, task.page_ids)
        t0 = time.perf_counter()
        X = features(task.queries, S["A"], S["B"], task.page_ids, tok)
        feat_ms = (time.perf_counter() - t0) * 1000 / len(task.queries)
        _, nb, _ = per_query(task, S["B"])
        _, nc, _ = per_query(task, C)
        data[name] = {"X": X, "nb": nb, "nc": nc, "queries": task.queries, "feat_ms_per_query": feat_ms}
    res, pvals = {}, {}
    out_dir = REPO_ROOT / cfg["output_dir"]
    for held in names:
        Xs, ys, ws = [], [], []
        for n in names:
            if n == held:
                continue
            d = data[n]
            keep = d["nc"] != d["nb"]
            Xs.append(d["X"][keep]); ys.append((d["nc"][keep] > d["nb"][keep]).astype(int))
            ws.append(np.full(keep.sum(), 1.0 / max(keep.sum(), 1)))
        X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(X), y, sample_weight=w)
        d = data[held]
        route = clf.predict_proba(scaler.transform(d["X"]))[:, 1] >= cfg["router"]["threshold"]
        nr = np.where(route, d["nc"], d["nb"])
        r = {"n_queries": len(nr), "routed_to_C": int(route.sum()), "ndcg_at_5": {"B": float(d["nb"].mean()), "C": float(d["nc"].mean()),
             "R": float(nr.mean()), "oracle": float(np.maximum(d["nb"], d["nc"]).mean())},
             "R_ci95": bootstrap_ci(nr), "R_vs_B": compare(nr, d["nb"]), "R_vs_C": compare(nr, d["nc"]),
             "train_rows": int(len(y)), "train_share_C_better": float(y.mean()),
             "coef_standardised": dict(zip(cfg["router"]["features"], map(float, clf.coef_[0]))),
             "feature_ms_per_query": d["feat_ms_per_query"],
             "note": "reused test split (fifth+ use)" if held == "docvqa" else "fresh task"}
        res[held], pvals[held] = r, r["R_vs_B"]["p"]
        with open(out_dir / f"router_{held}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["query", "route_to_C", "B_ndcg_at_5", "C_ndcg_at_5"])
            for q, rt, b, c in zip(d["queries"], route, d["nb"], d["nc"]):
                wr.writerow([q, int(rt), round(float(b), 6), round(float(c), 6)])
    for n, ph in holm(pvals).items():
        res[n]["R_vs_B"]["p_holm"] = ph
    macro = {s: float(np.mean([res[n]["ndcg_at_5"][s] for n in names])) for s in ("B", "C", "R", "oracle")}
    rec = collect_metadata(cfg, CONFIG, "cpu", cfg["seed"])
    rec.update({"label": "POST-HOC set 5 (pre-registered 2026-11-16): leave-one-task-out router B vs C",
                "folds": res, "macro_ndcg_at_5": macro, "macro_delta_R_minus_B": macro["R"] - macro["B"]})
    (out_dir / "router.json").write_text(json.dumps(rec, indent=2, default=float))
    print(f"{'held-out':<13} {'n':>5} {'->C':>5} {'B':>6} {'C':>6} {'R':>6} {'orac':>6} {'dR':>7} {'pH':>6}")
    for n, r in res.items():
        x = r["ndcg_at_5"]
        print(f"{n:<13} {r['n_queries']:>5} {r['routed_to_C']:>5} {x['B']:.4f} {x['C']:.4f} {x['R']:.4f} {x['oracle']:.4f} "
              f"{r['R_vs_B']['delta']:+.4f} {r['R_vs_B']['p_holm']:.3f}")
    print("macro:", json.dumps(macro), f"delta R-B {rec['macro_delta_R_minus_B']:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["bm25", "fusion", "router"], required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / CONFIG).read_text())
    {"bm25": stage_bm25, "fusion": stage_fusion, "router": stage_router}[args.stage](cfg)


if __name__ == "__main__":
    main()
