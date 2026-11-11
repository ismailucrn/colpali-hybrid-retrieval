"""The single test-split evaluation of all frozen systems: metrics, bootstrap CIs, paired tests, per-type
breakdown, win/tie/loss counts and a resource table. Run once; do not change any setting afterwards.

Writes to <output_dir>/: metrics.json, per_query.csv (every system's nDCG@5 and gold rank per test query),
summary.md (the tables for the report).

Usage:
    python scripts/run_test_evaluation.py --config configs/test_evaluation.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run, subset  # noqa: E402
from src.fusion import convex_combination, rrf, weighted_rrf  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402
from src.stats import bootstrap_ci, holm, paired_bootstrap_ci, paired_permutation_test  # noqa: E402

REPORT_METRICS = ["ndcg_at_5", "recall_at_5", "recall_at_10", "ndcg_at_10", "mrr_at_10"]
TAGS = ["exact", "general", "visual"]


DEFAULT_COMPONENTS = {"A": "results/bm25_docvqa/metrics.json", "B": "results/colpali_v1.3_docvqa/metrics.json",
                      "BGE": "results/bge_m3_docvqa/metrics.json"}


def check_frozen_settings(systems: dict, mode: str = "main") -> None:
    """Fail loudly if the config differs from what was selected on dev."""
    if mode == "posthoc":
        cc = json.loads((REPO_ROOT / "results/posthoc_extensions/dev_selection_cc.json").read_text())["systems"]
        plain = json.loads((REPO_ROOT / "results/fusion_docvqa/dev_selection.json").read_text())["systems"]
        for sid in ("C_cc", "D_cc"):
            if systems[sid]["alpha"] != cc[sid]["selected_alpha"]:
                raise SystemExit(f"{sid}: config alpha {systems[sid]['alpha']} != dev selection {cc[sid]['selected_alpha']}")
        for sid in ("C", "C_code"):  # C_code reuses C's frozen k (no new tuning)
            if systems[sid]["k"] != plain["C"]["selected_k"]:
                raise SystemExit(f"{sid}: k {systems[sid]['k']} != C's dev selection {plain['C']['selected_k']}")
        return
    plain = json.loads((REPO_ROOT / "results/fusion_docvqa/dev_selection.json").read_text())["systems"]
    weighted = json.loads((REPO_ROOT / "results/fusion_docvqa/dev_selection_weighted.json").read_text())["systems"]
    expected = {"C": (None, plain["C"]["selected_k"]), "D": (None, plain["D"]["selected_k"]),
                "C_w": (weighted["C_w"]["selected_w"], weighted["C_w"]["selected_k"]),
                "D_w": (weighted["D_w"]["selected_w"], weighted["D_w"]["selected_k"])}
    for sid, (w, k) in expected.items():
        cfg = systems[sid]
        got_w = cfg["weights"][0] if cfg["kind"] == "wrrf" else None
        if (got_w, cfg["k"]) != (w, k):
            raise SystemExit(f"{sid}: config (w={got_w}, k={cfg['k']}) != dev selection (w={w}, k={k})")


def build_scores(systems: dict, base: dict[str, np.ndarray], page_ids: list[str]) -> tuple[dict, dict]:
    scores, fusion_ms = {}, {}
    for sid, cfg in systems.items():
        if cfg["kind"] == "single":
            scores[sid] = base[cfg["scores"]]
            continue
        inputs = [base[i] for i in cfg["inputs"]]
        if cfg["kind"] == "rrf":
            fuse = lambda x, c=cfg: rrf(x, page_ids, k=c["k"])  # noqa: E731
        elif cfg["kind"] == "wrrf":
            fuse = lambda x, c=cfg: weighted_rrf(x, c["weights"], page_ids, k=c["k"])  # noqa: E731
        elif cfg["kind"] == "cc":  # inputs = [ColPali, text]
            fuse = lambda x, c=cfg: convex_combination(x[0], x[1], c["alpha"])  # noqa: E731
        else:
            raise SystemExit(f"{sid}: unknown kind {cfg['kind']}")
        scores[sid] = fuse(inputs)
        # Fusion cost for one query (ranking both 500-page lists + combining), median of 200 repeats.
        one = [x[:1] for x in inputs]
        times = []
        for _ in range(200):
            t = time.perf_counter()
            fuse(one)
            times.append(time.perf_counter() - t)
        fusion_ms[sid] = float(np.median(times) * 1000)
    return scores, fusion_ms


def resource_table(fusion_ms: dict, systems: dict, components: dict | None = None) -> dict:
    comp = {}
    for key, path in (components or DEFAULT_COMPONENTS).items():
        m = json.loads((REPO_ROOT / path).read_text())
        lat = m["latency"]["total"]["median_ms"] if "total" in m["latency"] else m["latency"]["median_ms"]
        mem = m["peak_memory"]
        comp[key] = {
            "device": m["hardware"]["device"] + (f" ({m['hardware'].get('gpu_name')})" if m["hardware"].get("gpu_name") else ""),
            "index_time_s": m["timings"]["index_pages_s"],
            "index_size_mb": m["index"].get("size_mb", m["index"].get("pickled_size_mb")),
            "query_latency_median_ms": lat,
            "peak_memory_mb": next(v for k, v in mem.items() if k.endswith("_mb")),
            "peak_memory_kind": next(k for k in mem if k.endswith("_mb")),
        }
    table = {}
    for sid, cfg in systems.items():
        parts = [cfg["scores"]] if cfg["kind"] == "single" else cfg["inputs"]
        seq = sum(comp[p]["query_latency_median_ms"] for p in parts) + fusion_ms.get(sid, 0.0)
        par = max(comp[p]["query_latency_median_ms"] for p in parts) + fusion_ms.get(sid, 0.0)
        table[sid] = {
            "components": parts,
            "index_time_s": sum(comp[p]["index_time_s"] for p in parts),
            "index_size_mb": sum(comp[p]["index_size_mb"] for p in parts),
            "query_latency_sequential_ms": seq,
            "query_latency_parallel_ms": par,
            "fusion_overhead_ms": fusion_ms.get(sid, 0.0),
        }
    return {"components": comp, "systems": table,
            "note": "Component latencies were measured on different devices (ColPali: L4 GPU; BGE-M3: Mac MPS; "
                    "BM25 and fusion: Mac CPU). Sequential = sum of components + fusion; parallel = slowest component + fusion."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/test_evaluation.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    dcfg, scfg, systems = config["data"], config["statistics"], config["systems"]
    check_frozen_settings(systems, config.get("frozen_check", "main"))

    task = load_task(dcfg["dataset"], dcfg["revision"], dcfg["split"])
    test = json.loads((REPO_ROOT / config["splits"]).read_text())["test"]
    tags = {r["query"]: r["tag"] for r in csv.DictReader(open(REPO_ROOT / config["tags"]))}
    base = {k: load_scores(p, task.queries, task.page_ids) for k, p in config["scores"].items()}
    scores, fusion_ms = build_scores(systems, base, task.page_ids)

    # Per-system test metrics and per-query values.
    agg, perq = {}, {}
    for sid, s in scores.items():
        run = scores_to_run(s, task.queries, task.page_ids)
        agg[sid], perq[sid] = evaluate(*subset(task.qrels, run, test))
    metric = scfg["metric"]
    vec = {sid: np.array([perq[sid][q][metric] for q in test]) for sid in scores}

    def ci(values):
        return bootstrap_ci(values, scfg["bootstrap_resamples"], scfg["alpha"], scfg["seed"])

    results = {sid: {"name": systems[sid]["name"], "role": systems[sid].get("role", ""),
                     "metrics": {m: agg[sid][m] for m in REPORT_METRICS},
                     f"{metric}_ci95": ci(vec[sid])} for sid in scores}

    # Paired comparisons.
    def compare(a: str, b: str, queries_idx=None) -> dict:
        x, y = vec[a], vec[b]
        if queries_idx is not None:
            x, y = x[queries_idx], y[queries_idx]
        d = x - y
        return {"delta": float(d.mean()),
                "delta_ci95": paired_bootstrap_ci(x, y, scfg["bootstrap_resamples"], scfg["alpha"], scfg["seed"]),
                "p_permutation": paired_permutation_test(x, y, scfg["permutations"], scfg["seed"]),
                "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()), "n": int(len(d))}

    bl = config["baseline"]
    vs_baseline = {sid: compare(sid, bl) for sid in config["compare_to_baseline"]}
    adjusted = holm({sid: r["p_permutation"] for sid, r in vs_baseline.items()})
    for sid in vs_baseline:
        vs_baseline[sid]["p_holm"] = adjusted[sid]
    extra = {f"{a} vs {b}": compare(a, b) for a, b in config["extra_pairs"]}

    # Per query type (exploratory: no multiple-comparison correction, small n for visual).
    by_type = {}
    for tag in TAGS:
        idx = np.array([i for i, q in enumerate(test) if tags[q] == tag])
        by_type[tag] = {"n": int(len(idx)),
                        "systems": {sid: {"mean": float(vec[sid][idx].mean()), "ci95": ci(vec[sid][idx])} for sid in scores},
                        "vs_baseline": {sid: compare(sid, bl, idx) for sid in config["compare_to_baseline"]}}

    # Post-hoc sensitivity: without duplicated (ambiguous) query texts.
    clean_idx = np.array([i for i, q in enumerate(test) if q not in task.duplicated_queries])
    posthoc = {"n": int(len(clean_idx)), "systems": {sid: float(vec[sid][clean_idx].mean()) for sid in scores},
               "vs_baseline": {sid: compare(sid, bl, clean_idx) for sid in config["compare_to_baseline"]}}

    resources = resource_table(fusion_ms, systems, config.get("component_metrics"))

    # Optional: top-5 headroom between two single systems (who finds the gold page that the other misses).
    headroom = {}
    for a, b in config.get("headroom_pairs", []):
        ra = {q: perq[a][q]["gold_rank"] for q in test}
        rb = {q: perq[b][q]["gold_rank"] for q in test}
        headroom[f"{a} vs {b}"] = {
            group: {"n": len(qs), f"{a}_hit": sum(ra[q] <= 5 for q in qs), f"{b}_hit": sum(rb[q] <= 5 for q in qs),
                    f"{a}_only": sum(ra[q] <= 5 < rb[q] for q in qs), f"{b}_only": sum(rb[q] <= 5 < ra[q] for q in qs)}
            for group, qs in {"all": test, **{t: [q for q in test if tags[q] == t] for t in TAGS}}.items()}

    # Outputs.
    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_query.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "tag", "duplicated", "gold_page"] +
                   [f"{sid}_{c}" for sid in scores for c in (metric, "gold_rank")])
        for q in test:
            w.writerow([q, tags[q], int(q in task.duplicated_queries), next(iter(task.qrels[q]))] +
                       [v for sid in scores for v in (round(perq[sid][q][metric], 6), perq[sid][q]["gold_rank"])])

    record = collect_metadata(config, args.config, "cpu", scfg["seed"])
    record.update({"n_test_queries": len(test), "results": results, "vs_baseline": vs_baseline, "pairs": extra,
                   "by_type": by_type, "posthoc_without_duplicated_queries": posthoc, "resources": resources})
    if headroom:
        record["headroom_at_5"] = headroom
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))

    lines = [f"# {config.get('title', 'Test-split results')} (n = {len(test)} queries), nDCG@5 unless stated\n",
             "| System | Role | nDCG@5 [95% CI] | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | Δ vs B [95% CI] | p (perm.) | p (Holm) | W/T/L vs B |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sid in scores:
        r, m = results[sid], results[sid]["metrics"]
        lo, hi = r[f"{metric}_ci95"]
        cmp_ = vs_baseline.get(sid)
        cells = (f"{cmp_['delta']:+.4f} [{cmp_['delta_ci95'][0]:+.4f}, {cmp_['delta_ci95'][1]:+.4f}] | {cmp_['p_permutation']:.4f} | "
                 f"{cmp_['p_holm']:.4f} | {cmp_['wins']}/{cmp_['ties']}/{cmp_['losses']}") if cmp_ else "– | – | – | –"
        lines.append(f"| {r['name']} | {r['role'] or 'baseline'} | {m[metric]:.4f} [{lo:.4f}, {hi:.4f}] | {m['recall_at_5']:.4f} | "
                     f"{m['recall_at_10']:.4f} | {m['ndcg_at_10']:.4f} | {m['mrr_at_10']:.4f} | {cells} |")
    lines += ["", "## " + config.get("pairs_title", "Pairs (P3: lexical signal vs any second retriever)"), "", "| Pair | Δ [95% CI] | p (perm.) | W/T/L |", "|---|---|---|---|"]
    for name, c in extra.items():
        lines.append(f"| {name} | {c['delta']:+.4f} [{c['delta_ci95'][0]:+.4f}, {c['delta_ci95'][1]:+.4f}] | {c['p_permutation']:.4f} | {c['wins']}/{c['ties']}/{c['losses']} |")
    lines += ["", "## By query type (exploratory, uncorrected)", "",
              "| System | " + " | ".join(f"{t} (n={by_type[t]['n']})" for t in TAGS) + " |", "|---|" + "---|" * len(TAGS)]
    for sid in scores:
        cells = []
        for t in TAGS:
            s = by_type[t]["systems"][sid]
            c = by_type[t]["vs_baseline"].get(sid)
            cells.append(f"{s['mean']:.3f}" + (f" (Δ {c['delta']:+.3f}, p={c['p_permutation']:.3f})" if c else ""))
        lines.append(f"| {systems[sid]['name']} | " + " | ".join(cells) + " |")
    lines += ["", f"## Post-hoc: without duplicated query texts (n = {posthoc['n']})", "",
              "| System | nDCG@5 | Δ vs B |", "|---|---|---|"]
    for sid in scores:
        c = posthoc["vs_baseline"].get(sid)
        lines.append(f"| {systems[sid]['name']} | {posthoc['systems'][sid]:.4f} | " + (f"{c['delta']:+.4f}" if c else "–") + " |")
    lines += ["", "## Resources (per system; components measured on different devices, see note)", "",
              "| System | Index time (s) | Index size (MB) | Query latency, sequential (ms) | parallel (ms) | Fusion overhead (ms) |",
              "|---|---|---|---|---|---|"]
    for sid in scores:
        r = resources["systems"][sid]
        lines.append(f"| {systems[sid]['name']} | {r['index_time_s']:.2f} | {r['index_size_mb']:.2f} | "
                     f"{r['query_latency_sequential_ms']:.2f} | {r['query_latency_parallel_ms']:.2f} | {r['fusion_overhead_ms']:.3f} |")
    lines += ["", resources["note"]]
    for name, h in headroom.items():
        lines += ["", f"## Top-5 headroom: {name}", "", "| Group | " + " | ".join(k for k in h["all"] if k != "n") + " |",
                  "|---|" + "---|" * (len(h["all"]) - 1)]
        for group, row in h.items():
            lines.append(f"| {group} (n={row['n']}) | " + " | ".join(str(v) for k, v in row.items() if k != "n") + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
