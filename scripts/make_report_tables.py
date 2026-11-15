"""Generate every number and table in the report from committed result files.

Nothing in report/main.tex is typed by hand: inline numbers are LaTeX macros from
report/generated/numbers.tex, and tables are \\input from report/generated/tab_*.tex.

Inputs (all committed):
  results/{colpali_v1.3_docvqa,bm25_docvqa,bge_m3_docvqa}/metrics.json   full-set runs
  results/fusion_docvqa/dev_selection.json, dev_selection_weighted.json  dev selection
  results/test_evaluation/metrics.json                                   single test run
  results/failure_analysis/analysis.json                                 failure analysis
Outputs: report/generated/numbers.tex, report/generated/tab_*.tex

Usage:
    python scripts/make_report_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "report/generated"

ORDER = ["A", "BGE", "B", "C", "D", "C_w", "D_w"]
LABEL = {
    "A": "A: BM25",
    "BGE": "BGE-M3 (dense text)",
    "B": "B: ColPali v1.3",
    "C": "C: RRF(BM25, ColPali), $k{=}1$",
    "D": "D: RRF(BGE-M3, ColPali), $k{=}1$",
    "C_w": "C$_w$: wRRF(BM25, ColPali), $w{=}0.95$, $k{=}20$",
    "D_w": "D$_w$: wRRF(BGE-M3, ColPali), $w{=}0.95$, $k{=}10$",
}
SHORT = {"A": "A", "BGE": "BGE-M3", "B": "B", "C": "C", "D": "D", "C_w": "C$_w$", "D_w": "D$_w$"}
ROLE = {"A": "text baseline", "BGE": "D's text part", "B": "reproduced baseline", "C": "proposed",
        "D": "control", "C_w": "ablation E", "D_w": "ablation E"}


def load(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def f3(x: float) -> str:
    return f"{x:.3f}"


def sgn(x: float, nd: int = 3) -> str:
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def ci(c: list[float]) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def sci(c: list[float]) -> str:
    return f"[{sgn(c[0])}, {sgn(c[1])}]"


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def table(name: str, spec: str, header: str, rows: list[str], note: str | None = None) -> None:
    lines = [r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{" + spec + "}", r"\toprule",
             header + r" \\", r"\midrule"]
    lines += [r + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        lines.append(r"\par\smallskip{\footnotesize " + note + "}")
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def posthoc2(N: dict) -> None:
    """POST-HOC set 2 (pre-registered 2026-11-15, commit 92c2163): text RAG stack, cross-encoder, cascade, OCR."""
    t = load("results/posthoc2/metrics.json")
    hy = load("results/posthoc2/dev_selection_hybrid.json")
    rr = load("results/posthoc2/dev_selection_rerank.json")
    ce = load("results/reranker_docvqa/metrics.json")
    oq = load("results/ocr_quality/ocr_quality.json")
    R, V, P = t["results"], t["vs_baseline"], t["pairs"]
    k_h = hy["systems"]["T_hyb"]["selected_k"]
    n_t, n_b = rr["systems"]["T_rr"]["selected_n"], rr["systems"]["B_ce"]["selected_n"]
    N.update({
        "kThyb": str(k_h), "devThyb": f3(hy["systems"]["T_hyb"]["dev_sweep"][str(k_h)]["ndcg_at_5"]),
        "nTrr": str(n_t), "nBce": str(n_b),
        "devTrr": f3(rr["systems"]["T_rr"]["dev_sweep"][str(n_t)]["ndcg_at_5"]),
        "devBce": f3(rr["systems"]["B_ce"]["dev_sweep"][str(n_b)]["ndcg_at_5"]),
        "ceMaxLen": str(ce["max_length"]["chosen_max_length"]),
        "ceFitPct": f"{100 * ce['max_length']['share_of_pairs_fitting'][str(ce['max_length']['chosen_max_length'])]:.1f}",
        "cePairs": f"{ce['pool']['n_pairs']:,}".replace(",", "{,}"),
        "cePoolMean": f"{ce['pool']['mean_pages_per_query']:.0f}",
        "ceScoreS": f"{ce['timings']['score_pool_s']:.0f}",
        "ceMemGB": f"{next(v for k, v in ce['peak_memory'].items() if k.endswith('_mb')) / 1024:.1f}",
    })
    for s, key in (("T_hyb", "Thyb"), ("T_rr", "Trr"), ("B_ce", "Bce")):
        N[f"test{key}"] = f3(R[s]["metrics"]["ndcg_at_5"])
        N[f"test{key}CI"] = ci(R[s]["ndcg_at_5_ci95"])
        N[f"delta{key}"] = sgn(V[s]["delta"])
        N[f"delta{key}CI"] = sci(V[s]["delta_ci95"])
        N[f"pHolm{key}"] = pval(V[s]["p_holm"])
        N[f"wtl{key}"] = "{wins}/{ties}/{losses}".format(**V[s])
    for pair, key in (("T_rr vs T_hyb", "TrrThyb"), ("T_hyb vs A", "ThybA"), ("T_rr vs A", "TrrA")):
        N[f"delta{key}"] = sgn(P[pair]["delta"])
        N[f"delta{key}CI"] = sci(P[pair]["delta_ci95"])
        N[f"p{key}"] = pval(P[pair]["p_permutation"])
    lat = t["resources"]["systems"]
    for s, key in (("T_rr", "Trr"), ("B_ce", "Bce")):
        N[f"lat{key}"] = f"{lat[s]['query_latency_sequential_ms']:.0f}"
    H = t["headroom_at_5"]["T_rr vs B"]["all"]
    N.update({"headTrrOnly": str(H["T_rr_only"]), "headTrrBOnly": str(H["B_only"]), "hitTrr": str(H["T_rr_hit"])})

    rows = []
    for s in ["A", "BGE", "T_hyb", "T_rr", "B", "B_ce"]:
        m = R[s]
        label = {"A": "A: BM25", "BGE": "BGE-M3 (dense text)",
                 "T_hyb": f"T$_\\text{{hyb}}$: RRF(BM25, BGE-M3), $k{{=}}{k_h}$",
                 "T_rr": f"T$_\\text{{rr}}$: T$_\\text{{hyb}}$ top-{n_t} reranked by cross-encoder",
                 "B": "B: ColPali v1.3",
                 "B_ce": f"B$_\\text{{ce}}$: ColPali top-$N$ reranked by cross-encoder, $N{{=}}{n_b}$"
                         + (" (no reranking)" if n_b <= 1 else "")}[s]
        tail = f"{sgn(V[s]['delta'])} {sci(V[s]['delta_ci95'])} & {pval(V[s]['p_holm'])}" if s in V else "-- & --"
        rows.append(f"{label} & {f3(m['metrics']['ndcg_at_5'])} {ci(m['ndcg_at_5_ci95'])} & "
                    f"{f3(m['metrics']['recall_at_5'])} & {f3(m['metrics']['mrr_at_10'])} & {tail}")
    table("tab_posthoc2", "llllll",
          r"System (post-hoc set 2) & nDCG@5 [95\% CI] & R@5 & MRR@10 & $\Delta$ vs B [95\% CI] & $p_\mathrm{Holm}$", rows,
          f"Test split, {t['n_test_queries']} queries, evaluated once with settings chosen on dev. Holm correction over "
          "T$_\\text{hyb}$, T$_\\text{rr}$ and B$_\\text{ce}$ vs B. Cross-encoder: bge-reranker-v2-m3 over the same OCR text, "
          f"max length {ce['max_length']['chosen_max_length']} tokens.")

    rows = []
    for n in t["config"]["cascade_n_values"]:
        m = R[f"AB_{n}"]["metrics"]
        rows.append(f"{n} & {f3(R['A']['metrics'][f'recall_at_{n}'])} & {f3(m['ndcg_at_5'])} & {f3(m['recall_at_5'])}")
    rows.append(f"500 (all pages) & 1.000 & {f3(R['B']['metrics']['ndcg_at_5'])} & {f3(R['B']['metrics']['recall_at_5'])}")
    table("tab_cascade", "rrrr", r"$N$ & BM25 Recall@$N$ (ceiling) & nDCG@5 & R@5", rows,
          "Test split; ColPali reorders BM25's top $N$ (descriptive, no selection). $N{=}500$ is ColPali alone.")

    rows = []
    names = {"A": "BM25", "BGE": "BGE-M3", "B": "ColPali"}
    for g in ("low", "mid", "high", "empty OCR"):
        r = oq["by_tercile"][g]
        rng = f"{r['proxy_range'][0]:.2f}--{r['proxy_range'][1]:.2f}" if r["proxy_range"] else "--"
        rows.append(f"{g} & {r['n']} & {rng} & " + " & ".join(f3(r[f'{s}_hit5']) for s in names))
    rows.append(r"\midrule Spearman $\rho$ [95\% CI] & 451 & & " + " & ".join(
        f"{sgn(oq['spearman_vs_ndcg5'][s]['spearman_rho'], 2)} [{sgn(oq['spearman_vs_ndcg5'][s]['ci95'][0], 2)}, "
        f"{sgn(oq['spearman_vs_ndcg5'][s]['ci95'][1], 2)}]" for s in names))
    table("tab_ocr", "lrllll", r"Gold-page OCR proxy & $n$ & range & BM25 Hit@5 & BGE-M3 Hit@5 & ColPali Hit@5", rows,
          "Full query set. Proxy: share of alphabetic OCR tokens found in the NLTK English word list; queries grouped "
          "by the proxy of their gold page. $\\rho$: Spearman correlation between the proxy and per-query nDCG@5.")
    N.update({"ocrMedian": f"{oq['pages']['median']:.2f}", "ocrPtenth": f"{oq['pages']['p10']:.2f}",
              "ocrPninety": f"{oq['pages']['p90']:.2f}"})
    ep = load("results/checks/bge_empty_pages.json")  # diagnostic, not pre-registered, dev only
    E = ep["dev_ndcg_at_5"]
    N.update({"emptyTopFive": str(ep["queries_with_empty_page_in_bge_top5"]),
              "emptyBGEdev": f3(E["BGE-M3"]["as_is"]), "emptyBGEdevFixed": f3(E["BGE-M3"]["empty_demoted"]),
              "emptyDdev": f3(E["D = RRF(BGE-M3, ColPali), k=1"]["as_is"]),
              "emptyDdevFixed": f3(E["D = RRF(BGE-M3, ColPali), k=1"]["empty_demoted"]),
              "emptyThybdev": f3(E["T_hyb = RRF(BM25, BGE-M3), k=5"]["as_is"]),
              "emptyThybdevFixed": f3(E["T_hyb = RRF(BM25, BGE-M3), k=5"]["empty_demoted"])})
    for s, key in (("A", "A"), ("BGE", "BGE"), ("B", "B")):
        c = oq["spearman_vs_ndcg5"][s]
        N[f"ocrRho{key}"] = sgn(c["spearman_rho"], 2)
        N[f"ocrRho{key}CI"] = f"[{sgn(c['ci95'][0], 2)}, {sgn(c['ci95'][1], 2)}]"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full = {s: load(f"results/{d}/metrics.json") for s, d in
            [("B", "colpali_v1.3_docvqa"), ("A", "bm25_docvqa"), ("BGE", "bge_m3_docvqa")]}
    dev = load("results/fusion_docvqa/dev_selection.json")
    devw = load("results/fusion_docvqa/dev_selection_weighted.json")
    test = load("results/test_evaluation/metrics.json")
    fa = load("results/failure_analysis/analysis.json")

    N: dict[str, str] = {}

    # ---- full set (reproduction) -------------------------------------------------------------
    pub = full["B"]["published"]
    mB, mA, mG = (full[s]["metrics"] for s in ("B", "A", "BGE"))
    N.update({
        "nQueries": str(full["B"]["task"]["n_queries"]), "nPages": str(full["B"]["task"]["n_pages"]),
        "nDup": str(full["B"]["task"]["n_duplicated_query_texts"]),
        "pubCard": f3(pub["ndcg_at_5_model_card_v1.3"]), "pubPaper": f3(pub["ndcg_at_5_paper_table2_v1.0"]),
        "fullB": f3(mB["ndcg_at_5"]), "fullA": f3(mA["ndcg_at_5"]), "fullBGE": f3(mG["ndcg_at_5"]),
        "gapCardPts": f"{100 * (mB['ndcg_at_5'] - pub['ndcg_at_5_model_card_v1.3']):+.2f}".replace("-", "$-$"),
        "pubBMtwentyfive": f3(full["A"]["published"]["ndcg_at_5_paper_table2_bm25_unstructured_ocr"]),
        "pubBGE": f3(full["BGE"]["published"]["ndcg_at_5_paper_table2_bge_m3_unstructured_ocr"]),
        "fullBnodup": f3(full["B"]["metrics_without_duplicated_queries_posthoc"]["ndcg_at_5"]),
    })
    rows = []
    for s, pubv, src in [("B", pub["ndcg_at_5_model_card_v1.3"], "model card v1.3"),
                         ("A", full["A"]["published"]["ndcg_at_5_paper_table2_bm25_unstructured_ocr"],
                          "paper Tab.~2, Unstructured+OCR"),
                         ("BGE", full["BGE"]["published"]["ndcg_at_5_paper_table2_bge_m3_unstructured_ocr"],
                          "paper Tab.~2, Unstructured+OCR")]:
        m = full[s]["metrics"]
        rows.append(f"{LABEL[s]} & {f3(pubv)} & {src} & {f3(m['ndcg_at_5'])} & {sgn(m['ndcg_at_5'] - pubv)} & "
                    f"{f3(m['recall_at_5'])} & {f3(m['recall_at_10'])} & {f3(m['ndcg_at_10'])} & {f3(m['mrr_at_10'])}")
    table("tab_reproduction", "lrlrrrrrr",
          r"System & \multicolumn{2}{c}{Published nDCG@5 (source)} & nDCG@5 & $\Delta$ & R@5 & R@10 & nDCG@10 & MRR@10",
          rows,
          f"Measured on the full set ({N['nQueries']} queries, {N['nPages']} pages), official protocol. "
          f"Published values are the authors' (left); measured values are ours (right). "
          f"Context: the paper's v1.0 checkpoint reports {N['pubPaper']} (Table~2). "
          "BM25 and BGE-M3 read different OCR text than the paper, so their $\\Delta$ is not a reproduction gap.")

    # ---- dev selection ----------------------------------------------------------------------
    bdev = devw["B_ColPali_dev"]
    dC, dD = dev["systems"]["C"]["dev_sweep"], dev["systems"]["D"]["dev_sweep"]
    cw, dw = devw["systems"]["C_w"], devw["systems"]["D_w"]
    N.update({
        "nDev": str(dev["dev_n_queries"]), "devB": f3(bdev), "devA": f3(dev["baselines_dev"]["C text component"]["ndcg_at_5"]),
        "devBGE": f3(dev["baselines_dev"]["D text component"]["ndcg_at_5"]),
        "devCkone": f3(dC["1"]["ndcg_at_5"]), "devDkone": f3(dD["1"]["ndcg_at_5"]),
        "devCksixty": f3(dC["60"]["ndcg_at_5"]), "devDksixty": f3(dD["60"]["ndcg_at_5"]),
        "devCw": f3(cw["dev_ndcg_at_5"]), "devDw": f3(dw["dev_ndcg_at_5"]),
        "devCwGainPts": f"{100 * (cw['dev_ndcg_at_5'] - bdev):+.2f}",
        "nWgrid": str(len(cw["dev_surface"])), "nKgrid": str(len(next(iter(cw["dev_surface"].values())))),
    })
    rows = [f"B: ColPali & -- & {f3(bdev)} & --",
            f"A: BM25 & -- & {N['devA']} & {sgn(float(N['devA']) - bdev)}",
            f"BGE-M3 & -- & {N['devBGE']} & {sgn(float(N['devBGE']) - bdev)}"]
    for s, sweep in (("C", dC), ("D", dD)):
        for k in ("1", "60"):
            v = sweep[k]["ndcg_at_5"]
            sel = " (selected)" if int(k) == dev["systems"][s]["selected_k"] else " (default)"
            rows.append(f"{SHORT[s]}: RRF, $k{{=}}{k}${sel} & equal & {f3(v)} & {sgn(v - bdev)}")
    for s, d in (("C_w", cw), ("D_w", dw)):
        rows.append(f"{SHORT[s]}: wRRF, $k{{=}}{d['selected_k']}$ (selected) & $w{{=}}{d['selected_w']}$ & "
                    f"{f3(d['dev_ndcg_at_5'])} & {sgn(d['dev_ndcg_at_5'] - bdev)}")
    table("tab_dev", "llrr", r"System (dev only) & ColPali weight & nDCG@5 & $\Delta$ vs B", rows,
          f"Dev split, {N['nDev']} queries. Equal-weight RRF: $k$ grid of 10 values, best $k$ at the lower edge. "
          f"Weighted RRF: {N['nWgrid']}$\\times${N['nKgrid']} grid, chosen after the equal-weight results were seen "
          "(dev-informed), so its dev score is optimistically biased.")

    # ---- single test run --------------------------------------------------------------------
    R, V = test["results"], test["vs_baseline"]
    N["nTest"] = str(test["n_test_queries"])
    for s in ORDER:
        key = s.replace("_", "")
        N[f"test{key}"] = f3(R[s]["metrics"]["ndcg_at_5"])
        N[f"test{key}CI"] = ci(R[s]["ndcg_at_5_ci95"])
        if s in V:
            v = V[s]
            N[f"delta{key}"] = sgn(v["delta"])
            N[f"delta{key}CI"] = sci(v["delta_ci95"])
            N[f"pHolm{key}"] = pval(v["p_holm"])
            N[f"p{key}"] = pval(v["p_permutation"])
            N[f"wtl{key}"] = f"{v['wins']}/{v['ties']}/{v['losses']}"
    rows = []
    for s in ORDER:
        m = R[s]["metrics"]
        if s in V:
            v = V[s]
            tail = f"{sgn(v['delta'])} {sci(v['delta_ci95'])} & {pval(v['p_holm'])} & {v['wins']}/{v['ties']}/{v['losses']}"
        else:
            tail = "-- & -- & --"
        rows.append(f"{SHORT[s]} & {ROLE[s]} & {f3(m['ndcg_at_5'])} {ci(R[s]['ndcg_at_5_ci95'])} & {f3(m['recall_at_5'])} & "
                    f"{f3(m['recall_at_10'])} & {f3(m['mrr_at_10'])} & {tail}")
    table("tab_test", "llllllll",
          r"System & Role & nDCG@5 [95\% CI] & R@5 & R@10 & MRR@10 & $\Delta$ vs B [95\% CI] & $p_\mathrm{Holm}$",
          [r.rsplit(" & ", 1)[0] for r in rows],
          f"Test split, {N['nTest']} queries, run once with settings frozen on dev. CI: percentile bootstrap over queries "
          "(10{,}000 resamples). $p$: two-sided paired sign-flip permutation test (10{,}000), Holm-corrected over the six "
          "comparisons with B.")
    rows_wtl = [f"{SHORT[s]} & {V[s]['wins']} & {V[s]['ties']} & {V[s]['losses']}" for s in ORDER if s in V]
    table("tab_wtl", "lrrr", r"vs B & better & same & worse", rows_wtl,
          "Per-query nDCG@5 comparison with ColPali on the test split.")

    P = test["pairs"]
    N.update({"deltaCvsD": sgn(P["C vs D"]["delta"]), "deltaCvsDCI": sci(P["C vs D"]["delta_ci95"]),
              "pCvsD": pval(P["C vs D"]["p_permutation"]), "wtlCvsD": "{wins}/{ties}/{losses}".format(**P["C vs D"]),
              "pCwvsDw": pval(P["C_w vs D_w"]["p_permutation"])})

    # by type
    BT = test["by_type"]
    tags = ["exact", "general", "visual"]
    for t in tags:
        N[f"n{t}"] = str(BT[t]["n"])
        for s in ("A", "B", "C", "D"):
            N[f"type{t}{s}"] = f3(BT[t]["systems"][s]["mean"])
    rows = []
    for s in ORDER:
        cells = []
        for t in tags:
            mean = BT[t]["systems"][s]["mean"]
            if s == "B":
                cells.append(f"{f3(mean)}")
            else:
                v = BT[t]["vs_baseline"][s]
                pp = v['p_permutation']
                cells.append(f"{f3(mean)} ({sgn(v['delta'])}, " + ("$p<0.001$" if pp < 0.001 else f"$p={pp:.3f}$") + ")")
        rows.append(f"{SHORT[s]} & " + " & ".join(cells))
    table("tab_type", "llll",
          " & ".join(["System"] + [f"{t} ($n$={BT[t]['n']})" for t in tags]),
          rows, "Test nDCG@5 by frozen query tag; in brackets $\\Delta$ vs B and the uncorrected permutation $p$ "
                "(exploratory).")

    # post hoc without duplicated texts
    PH = test["posthoc_without_duplicated_queries"]
    N.update({"nNoDup": str(PH["n"]), "noDupB": f3(PH["systems"]["B"]), "noDupC": f3(PH["systems"]["C"]),
              "noDupDeltaC": sgn(PH["vs_baseline"]["C"]["delta"]), "noDupDeltaCw": sgn(PH["vs_baseline"]["C_w"]["delta"])})

    # resources
    RS = test["resources"]
    comp = RS["components"]
    N.update({"peakGPU": f"{comp['B']['peak_memory_mb'] / 1024:.1f}",
              "latB": f"{comp['B']['query_latency_median_ms']:.1f}", "latA": f"{comp['A']['query_latency_median_ms']:.2f}",
              "latBGE": f"{comp['BGE']['query_latency_median_ms']:.1f}",
              "fusionOverhead": f"{RS['systems']['C']['fusion_overhead_ms']:.2f}",
              "idxB": f"{comp['B']['index_time_s']:.1f}", "sizeB": f"{comp['B']['index_size_mb']:.0f}"})
    rows = []
    for s in ORDER:
        r = RS["systems"][s]
        rows.append(f"{SHORT[s]} & {r['index_time_s']:.2f} & {r['index_size_mb']:.2f} & "
                    f"{r['query_latency_sequential_ms']:.2f} & {r['query_latency_parallel_ms']:.2f} & {r['fusion_overhead_ms']:.2f}")
    table("tab_resources", "lrrrrr",
          r"System & Index time (s) & Index size (MB) & Latency seq.\ (ms) & Latency par.\ (ms) & Fusion (ms)", rows,
          "Medians over single queries. Components were measured on different devices: ColPali on an NVIDIA L4 "
          f"(bf16, peak GPU memory {N['peakGPU']}~GiB), BGE-M3 on Apple M5 MPS (fp16), BM25 and fusion on the M5 CPU. "
          "Sequential = sum of components + fusion; parallel = slowest component + fusion.")

    # ---- failure analysis -------------------------------------------------------------------
    H = fa["headroom_at_5"]
    for t in ["all"] + tags:
        h = H[t]
        N[f"head{t}AnotB"] = str(h["A_hit_B_miss (max gain)"])
        N[f"head{t}BnotA"] = str(h["B_hit_A_miss (risk)"])
    N.update({"hitB": str(H["all"]["B_hit"]), "hitA": str(H["all"]["A_hit"]), "oracle": str(H["all"]["oracle_A_or_B_hit"]),
              "bothMiss": str(H["all"]["both_miss"])})
    rows = [f"{t} ({H[t]['n']}) & {H[t]['B_hit']} & {H[t]['A_hit']} & {H[t]['A_hit_B_miss (max gain)']} & "
            f"{H[t]['B_hit_A_miss (risk)']} & {H[t]['both_miss']} & {H[t]['oracle_A_or_B_hit']}" for t in ["all"] + tags]
    table("tab_headroom", "lrrrrrr",
          r"Tag ($n$) & B hit & A hit & A only & B only & neither & A or B", rows,
          "Test split; hit = gold page in the top 5. ``A only'' bounds what any fusion of A and B could gain at cut-off 5.")
    SO = fa["selection_oracle_ndcg_at_5"]["all"]
    N.update({"oracleNDCG": f3(SO["oracle_max_A_B_ndcg_at_5"]), "oracleGainPts": f"{100 * SO['gain']:+.1f}",
              "oracleAbetter": str(SO["queries_A_better_than_B"])})
    C = fa["C_vs_B"]
    N.update({"cWins": str(C["wins"]), "cLosses": str(C["losses"]),
              "cLossBMmissed": str(C["loss_mechanism"]["BM25 missed (gold > 5)"]),
              "cwChanged": str(fa["C_w_changed_queries"])})
    L = fa["lexical_overlap"]
    N.update({"ovAhit": f"{L['A_hit'][0]:.2f}", "ovAmiss": f"{L['A_miss'][0]:.2f}"})
    BO = fa["hit_rate_by_overlap"]
    N.update({"ovZeroN": str(BO["0"]["n"]), "ovZeroA": f"{100 * BO['0']['A_hit_rate']:.0f}",
              "ovZeroB": f"{100 * BO['0']['B_hit_rate']:.0f}", "ovOneN": str(BO["1"]["n"]),
              "ovOneA": f"{100 * BO['1']['A_hit_rate']:.0f}", "ovOneB": f"{100 * BO['1']['B_hit_rate']:.0f}"})

    # ---- post-hoc extensions (pre-registered 2026-11-11, after the main test run) ---------------------------
    ph = load("results/posthoc_extensions/metrics.json")
    phd = load("results/posthoc_extensions/dev_selection_cc.json")
    acode = load("results/bm25_codes_docvqa/metrics.json")
    PR, PV, PP = ph["results"], ph["vs_baseline"], ph["pairs"]
    N.update({
        "fullAcode": f3(acode["metrics"]["ndcg_at_5"]), "devAcode": f3(acode["metrics_dev"]["ndcg_at_5"]),
        "testAcode": f3(PR["A_code"]["metrics"]["ndcg_at_5"]),
        "deltaAcodeA": sgn(PP["A_code vs A"]["delta"]), "deltaAcodeACI": sci(PP["A_code vs A"]["delta_ci95"]),
        "pAcodeA": pval(PP["A_code vs A"]["p_permutation"]),
        "wtlAcodeA": "{wins}/{ties}/{losses}".format(**PP["A_code vs A"]),
        "typeexactAcode": f3(ph["by_type"]["exact"]["systems"]["A_code"]["mean"]),
        "testCcode": f3(PR["C_code"]["metrics"]["ndcg_at_5"]), "deltaCcode": sgn(PV["C_code"]["delta"]),
        "deltaCcodeCI": sci(PV["C_code"]["delta_ci95"]), "pHolmCcode": pval(PV["C_code"]["p_holm"]),
        "deltaCcodevsC": sgn(PP["C_code vs C"]["delta"]), "pCcodevsC": pval(PP["C_code vs C"]["p_permutation"]),
        "alphaCcc": str(phd["systems"]["C_cc"]["selected_alpha"]), "alphaDcc": str(phd["systems"]["D_cc"]["selected_alpha"]),
    })
    for s in ("C_cc", "D_cc"):
        curve = {float(a): v for a, v in phd["systems"][s]["dev_curve"].items()}
        a_best = max((a for a in curve if a < 1.0), key=lambda a: curve[a])
        key = s.replace("_", "")
        N[f"devBestBelowOne{key}"] = f3(curve[a_best])
        N[f"devBestBelowOneAlpha{key}"] = f"{a_best:g}"
    HA = ph["headroom_at_5"]["A_code vs B"]["all"]
    N.update({"headAcodeOnly": str(HA["A_code_only"]), "headAcodeBOnly": str(HA["B_only"]),
              "hitAcode": str(HA["A_code_hit"])})
    rows = []
    for s in ["A", "A_code", "B", "C", "C_code", "C_cc", "D_cc"]:
        m = PR[s]
        label = {"A": "A: BM25, official tokenizer", "A_code": "A$_\\text{code}$: BM25, code-preserving tokenizer",
                 "B": "B: ColPali v1.3", "C": "C: RRF(A, B), $k{=}1$ (main run)",
                 "C_code": "C$_\\text{code}$: RRF(A$_\\text{code}$, B), $k{=}1$",
                 "C_cc": f"C$_\\text{{cc}}$: CC(BM25, B), $\\alpha{{=}}{phd['systems']['C_cc']['selected_alpha']}$",
                 "D_cc": f"D$_\\text{{cc}}$: CC(BGE-M3, B), $\\alpha{{=}}{phd['systems']['D_cc']['selected_alpha']}$"}[s]
        if s in PV:
            tail = f"{sgn(PV[s]['delta'])} {sci(PV[s]['delta_ci95'])} & {pval(PV[s]['p_holm'])}"
        else:
            tail = "-- & --"
        rows.append(f"{label} & {f3(m['metrics']['ndcg_at_5'])} {ci(m['ndcg_at_5_ci95'])} & {tail}")
    table("tab_posthoc", "llll", r"System (post-hoc) & nDCG@5 [95\% CI] & $\Delta$ vs B [95\% CI] & $p_\mathrm{Holm}$", rows,
          f"Test split, {ph['n_test_queries']} queries; post-hoc analyses pre-registered after the main test run. "
          "Holm correction over C$_\\text{code}$, C$_\\text{cc}$, D$_\\text{cc}$ vs B. "
          "$\\alpha$ was selected on dev; $\\alpha{=}1$ means ColPali alone.")

    posthoc2(N)

    lines = ["% generated by scripts/make_report_tables.py -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in N.items()]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(N)} macros and {len(list(OUT.glob('tab_*.tex')))} tables to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
