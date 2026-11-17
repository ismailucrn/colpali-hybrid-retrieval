"""Regenerate every report figure from committed result files (no model, no dataset download).

Inputs:  results/test_evaluation/metrics.json, results/fusion_docvqa/dev_selection*.json,
         results/failure_analysis/analysis.json, results/posthoc*/metrics.json, results/posthoc4/fresh_task_infovqa.json,
         results/posthoc5/fusion.json
Outputs: results/figures/fig*.png (300 dpi) and .pdf

Colour follows the entity in every figure: ColPali = blue, BM25 family (A, C, C_w) = orange,
BGE-M3 family (BGE-M3, D, D_w) = aqua; marker shape separates single systems / RRF / weighted RRF.
Palette: dataviz reference palette, first three categorical slots (validated all-pairs, light mode).

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
GRAY_DARK, GRAY_LIGHT = "#7d7c78", "#d9d8d3"

FAMILY = {"A": ORANGE, "C": ORANGE, "C_w": ORANGE, "B": BLUE, "BGE": AQUA, "D": AQUA, "D_w": AQUA}
MARKER = {"A": "o", "B": "o", "BGE": "o", "C": "s", "D": "s", "C_w": "D", "D_w": "D"}
SHORT = {"A": "A: BM25", "BGE": "BGE-M3", "B": "B: ColPali", "C": "C: RRF(BM25, ColPali)",
         "D": "D: RRF(BGE-M3, ColPali)", "C_w": "C_w: wRRF(BM25, ColPali)", "D_w": "D_w: wRRF(BGE-M3, ColPali)"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 0.8, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "text.color": INK,
    })


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote results/figures/{name}.png/.pdf")


def load(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def fig1_test_results(test: dict) -> None:
    """Test nDCG@5 with 95% bootstrap CI per system; ColPali's value as a reference line."""
    order = ["A", "BGE", "C", "D", "C_w", "D_w", "B"]
    res = test["results"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b = res["B"]["metrics"]["ndcg_at_5"]
    ax.axvline(b, color=BLUE, lw=1, ls="--", zorder=1)  # ColPali reference; its value is labelled on its own row
    for y, sid in enumerate(order):
        m = res[sid]["metrics"]["ndcg_at_5"]
        lo, hi = res[sid]["ndcg_at_5_ci95"]
        ax.plot([lo, hi], [y, y], color=FAMILY[sid], lw=2, solid_capstyle="round", zorder=2)
        ax.plot(m, y, MARKER[sid], color=FAMILY[sid], ms=7, mec=SURFACE, mew=1.5, zorder=3)
        cmp_ = test["vs_baseline"].get(sid)
        note = f"{m:.3f}" + (f"   Δ {cmp_['delta']:+.3f}, p_Holm {cmp_['p_holm']:.3g}" if cmp_ else "")
        ax.text(hi + 0.01, y, note, va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(order)), [SHORT[s] for s in order])
    ax.set_xlim(0.15, 0.85)
    ax.set_xlabel("nDCG@5 on the test split (n = 225), 95% bootstrap CI")
    ax.set_title("Test results: no fusion beats ColPali alone", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig1_test_ndcg5")


def fig2_k_sweep(dev: dict) -> None:
    """Dev nDCG@5 vs RRF k for C and D (ablation E, dev only)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    b = dev["baselines_dev"]["B: ColPali"]["ndcg_at_5"]
    ax.axhline(b, color=BLUE, lw=1.5, ls="--")
    ax.text(1, b + 0.003, f"B: ColPali {b:.3f}", color=INK2, fontsize=8, va="bottom")
    for sid, color in (("C", ORANGE), ("D", AQUA)):
        sweep = dev["systems"][sid]["dev_sweep"]
        ks = sorted(int(k) for k in sweep)
        ys = [sweep[str(k)]["ndcg_at_5"] for k in ks]
        ax.plot(ks, ys, "-" + MARKER[sid], color=color, lw=2, ms=6, mec=SURFACE, mew=1.2)
        ax.text(510, ys[-1], f" {SHORT[sid]}", color=INK2, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3000)
    ax.set_xticks([1, 5, 10, 20, 60, 100, 500], ["1", "5", "10", "20", "60", "100", "500"])
    ax.set_xlabel("RRF k (log scale)")
    ax.set_ylabel("dev nDCG@5 (n = 226)")
    ax.set_title("Equal-weight RRF stays below ColPali at every k (dev)", loc="left")
    save(fig, "fig2_dev_k_sweep")


def fig3_weighted_heatmap(devw: dict) -> None:
    """Dev Δ nDCG@5 vs ColPali over the weighted-RRF grid, for C_w and D_w."""
    b = devw["B_ColPali_dev"]
    cmap = LinearSegmentedColormap.from_list("div", ["#c0392b", "#e88a7d", "#f0efec", "#86b6ef", "#1c5cab"])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, sid in zip(axes, ("C_w", "D_w")):
        surf = devw["systems"][sid]["dev_surface"]
        ws = list(surf)
        ks = list(surf[ws[0]])
        z = np.array([[surf[w][k] - b for k in ks] for w in ws])
        im = ax.imshow(z, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-0.12, vmax=0.012), aspect="auto")
        ax.set_xticks(range(len(ks)), ks, rotation=45)
        ax.set_yticks(range(len(ws)), [w.split("=")[1] for w in ws])
        ax.set_xlabel("RRF k")
        ax.grid(False)
        sel_w, sel_k = devw["systems"][sid]["selected_w"], devw["systems"][sid]["selected_k"]
        i, j = ws.index(f"w={sel_w}"), ks.index(str(sel_k))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.5))
        ax.set_title(f"{SHORT[sid]}\nselected w={sel_w}, k={sel_k}: Δ {z[i, j]:+.4f}", loc="left", fontsize=9)
    axes[0].set_ylabel("weight on ColPali (w)")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("dev nDCG@5 − ColPali")
    fig.suptitle("Weighted RRF on dev: only near-ColPali weights avoid harm (ablation E)", x=0.08, ha="left",
                 fontweight="bold", fontsize=10, y=1.12)
    save(fig, "fig3_dev_weighted_rrf")


def fig4_by_type(test: dict) -> None:
    """Per query type: absolute nDCG@5 of A and B (left) and Δ vs B for C, D, C_w (right)."""
    tags = ["exact", "general", "visual"]
    bt = test["by_type"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.2), gridspec_kw={"width_ratios": [1, 1.15]})
    x = np.arange(len(tags))
    for off, sid in ((-0.19, "A"), (0.19, "B")):
        vals = [bt[t]["systems"][sid]["mean"] for t in tags]
        ax1.bar(x + off, vals, width=0.36, color=FAMILY[sid], edgecolor=SURFACE, linewidth=2, label=SHORT[sid])
        for xi, v in zip(x + off, vals):
            ax1.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax1.set_xticks(x, [f"{t}\n(n={bt[t]['n']})" for t in tags])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("test nDCG@5")
    ax1.set_title("ColPali leads on every type, most on exact", loc="left")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="x", visible=False)

    ax2.axhline(0, color=BLUE, lw=1.2, ls="--")
    for off, sid in ((-0.22, "C"), (0.0, "D"), (0.22, "C_w")):
        for xi, t in zip(x + off, tags):
            c = bt[t]["vs_baseline"][sid]
            lo, hi = c["delta_ci95"]
            ax2.plot([xi, xi], [lo, hi], color=FAMILY[sid], lw=2, solid_capstyle="round")
            hollow = sid == "C_w"
            ax2.plot(xi, c["delta"], MARKER[sid], color=FAMILY[sid], ms=6.5, mew=1.6 if hollow else 1.2,
                     mec=FAMILY[sid] if hollow else SURFACE, mfc=SURFACE if hollow else FAMILY[sid], zorder=3,
                     label=SHORT[sid] if t == "exact" else None)
    ax2.set_xticks(x, [f"{t}\n(n={bt[t]['n']})" for t in tags])
    ax2.set_ylabel("Δ test nDCG@5 vs ColPali (95% CI)")
    ax2.set_title("Fusion never helps; equal-weight RRF hurts", loc="left")
    ax2.legend(loc="lower left", fontsize=7.5)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig4_test_by_type")


def fig5_headroom(fa: dict) -> None:
    """Headroom at top-5: who finds the gold page, per query type."""
    groups = ["exact", "general", "visual", "all"]
    h = fa["headroom_at_5"]
    parts = [("both hit", GRAY_DARK, lambda g: h[g]["B_hit"] - h[g]["B_hit_A_miss (risk)"]),
             ("only ColPali hits", BLUE, lambda g: h[g]["B_hit_A_miss (risk)"]),
             ("only BM25 hits", ORANGE, lambda g: h[g]["A_hit_B_miss (max gain)"]),
             ("both miss", GRAY_LIGHT, lambda g: h[g]["both_miss"])]
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for y, g in enumerate(groups):
        left, n = 0.0, h[g]["n"]
        for label, color, f in parts:
            share = f(g) / n
            ax.barh(y, share, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.62,
                    label=label if y == 0 else None)
            if f(g) and (share > 0.06 or label == "only BM25 hits"):
                txt_color = SURFACE if color in (GRAY_DARK, BLUE) else INK
                ax.text(left + share / 2 if share > 0.06 else left + share + 0.005, y, str(f(g)),
                        ha="center" if share > 0.06 else "left", va="center", fontsize=7.5, color=txt_color)
            left += share
    ax.set_yticks(range(len(groups)), [f"{g} (n={h[g]['n']})" for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of test queries (gold page in the top 5)")
    ax.set_title("BM25 finds almost nothing ColPali misses (3 of 225 queries)", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "fig5_headroom")


def fig6_overlap(fa: dict) -> None:
    """Hit rate at top-5 by lexical overlap between query and gold-page OCR text."""
    bo = fa["hit_rate_by_overlap"]
    buckets = list(bo)
    x = np.arange(len(buckets))
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for off, key, sid in ((-0.19, "A_hit_rate", "A"), (0.19, "B_hit_rate", "B")):
        vals = [bo[b][key] for b in buckets]
        ax.bar(x + off, vals, width=0.36, color=FAMILY[sid], edgecolor=SURFACE, linewidth=2, label=SHORT[sid])
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [f"{b}\n(n={bo[b]['n']})" for b in buckets])
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("share of the query's BM25 tokens found in the gold page's OCR text")
    ax.set_ylabel("hit rate (gold page in top 5)")
    ax.set_title("BM25 needs the query words in the OCR text; ColPali does not", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "fig6_overlap")


# ---- overview figures added 2026-11-17 (post-hoc sets 1-5; short talk) -----------------------------------

ATTEMPTS = [  # (metrics file, system id, label, second signal); order = chronology
    ("test_evaluation", "C", "C: RRF(BM25, ColPali)", "text"),
    ("test_evaluation", "D", "D: RRF(BGE-M3, ColPali)  [control]", "text"),
    ("test_evaluation", "C_w", "C_w: weighted RRF (BM25)", "text"),
    ("test_evaluation", "D_w", "D_w: weighted RRF (BGE-M3)", "text"),
    ("posthoc_extensions", "C_code", "Set 1: C with code-preserving BM25", "text"),
    ("posthoc_extensions", "C_cc", "Set 1: score-level fusion (BM25)", "text"),
    ("posthoc_extensions", "D_cc", "Set 1: score-level fusion (BGE-M3)", "text"),
    ("posthoc2", "T_rr", "Set 2: text RAG stack (hybrid + reranker)", "text"),
    ("posthoc2", "B_ce", "Set 2: ColPali + cross-encoder rerank", "text"),
    ("posthoc3", "C_sel", "Set 3: selective fusion", "text"),
    ("posthoc3", "C_vlm", "Set 3: RRF(BM25 on Qwen3.5 OCR, ColPali)", "text"),
    ("posthoc3", "E_ens", "Set 3: RRF(Qwen3-VL-Emb-2B, ColPali)", "visual"),
    ("posthoc3", "B_vr", "Set 3: ColPali + Qwen3-VL-Reranker-2B", "visual"),
    ("posthoc4", "F_q8", "Set 4: RRF(Qwen3-VL-Emb-8B, ColPali)", "visual"),
    ("posthoc4", "B_vr8", "Set 4: ColPali + Qwen3-VL-Reranker-8B", "visual"),
    ("posthoc4", "F_cq", "Set 4: RRF(ColPali, ColQwen2.5)", "visual"),
    ("posthoc4", "F_tm", "Set 4: RRF(ColPali, Tomoro)", "visual"),
    ("posthoc4", "CQ", "Set 4: ColQwen2.5 alone (replaces ColPali)", "replace"),
    ("posthoc4", "TM", "Set 4: Tomoro ColQwen3-4B alone (replaces ColPali)", "replace"),
]
KIND_COLOR = {"text": ORANGE, "visual": AQUA, "replace": GRAY_DARK}


def fig7_all_attempts() -> None:
    """Every attempt to beat ColPali on the DocVQA test split: paired delta nDCG@5 with 95% CI (one row each)."""
    from matplotlib.lines import Line2D

    files = {f: load(f"results/{f}/metrics.json") for f in {a[0] for a in ATTEMPTS}}
    fresh = load("results/posthoc4/fresh_task_infovqa.json")
    rows = [(lbl, files[f]["vs_baseline"][sid]["delta"], files[f]["vs_baseline"][sid]["delta_ci95"],
             files[f]["vs_baseline"][sid]["p_holm"], kind) for f, sid, lbl, kind in ATTEMPTS]
    rows.append(("InfoVQA (fresh task): Tomoro alone, frozen", fresh["delta"], fresh["delta_ci95"],
                 fresh["p_permutation"], "replace"))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.axvline(0, color=BLUE, lw=1.2, ls="--", zorder=1)
    n = len(rows)
    for i, (lbl, d, (lo, hi), p, kind) in enumerate(rows):
        y = n - 1 - i
        c = KIND_COLOR[kind]
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(d, y, "o", color=c if p < 0.05 else SURFACE, mec=c, mew=1.6, ms=6.5, zorder=3)
        ax.text(0.205, y, f"{100 * d:+.1f}", va="center", ha="right", fontsize=7.5, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.set_yticks(range(n), [r[0] for r in reversed(rows)], fontsize=7.8)
    ax.set_xlim(-0.2, 0.21)
    ax.set_xlabel("Δ nDCG@5 vs ColPali v1.3 (paired, 95% bootstrap CI)")
    ax.set_title("Every attempt to improve on ColPali (DocVQA test, n = 225; bottom row: InfoVQA)", loc="left")
    handles = [Line2D([], [], color=KIND_COLOR[k], marker="o", lw=2, label=t) for k, t in
               (("text", "adds a text signal"), ("visual", "adds a second visual signal"), ("replace", "replaces ColPali"))]
    handles.append(Line2D([], [], color=MUTED, marker="o", mfc=SURFACE, lw=0, label="hollow = not significant (p ≥ 0.05)"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.42), fontsize=7.5)
    ax.text(0.205, n - 0.35, "Δ pts", ha="right", fontsize=7.5, color=MUTED)
    save(fig, "fig7_all_attempts")


TASK_NAMES = {"infovqa": "InfoVQA", "arxivqa": "ArxivQA", "tabfquad": "TabFQuAD", "tatdqa": "TAT-DQA",
              "shiftproject": "ShiftProject", "synth_energy": "Synth. energy", "synth_ai": "Synth. AI",
              "synth_gov": "Synth. gov.", "synth_health": "Synth. health"}


def fig8_vidore_tasks() -> None:
    """Post-hoc set 5: frozen C vs ColPali on nine ViDoRe v1 tasks; strength gap vs fusion delta (post-hoc)."""
    from matplotlib.lines import Line2D

    fu = load("results/posthoc5/fusion.json")["tasks"]
    gcol = {"visual": AQUA, "text": ORANGE, "none": GRAY_DARK}
    order = sorted(fu, key=lambda t: fu[t]["C_vs_B"]["delta"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    a1.axvline(0, color=BLUE, lw=1.2, ls="--")
    for y, t in enumerate(order):
        r = fu[t]["C_vs_B"]
        c = gcol[fu[t]["group"]]
        a1.plot(r["delta_ci95"], [y, y], color=c, lw=2, solid_capstyle="round")
        a1.plot(r["delta"], y, "o", color=c if r["p_holm"] < 0.05 else SURFACE, mec=c, mew=1.6, ms=6.5)
    a1.set_yticks(range(len(order)), [f"{TASK_NAMES[t]} (n={fu[t]['n_queries']})" for t in order], fontsize=7.8)
    a1.set_xlabel("Δ nDCG@5, C − ColPali (95% CI)")
    a1.set_title("Frozen DocVQA fusion on nine ViDoRe v1 tasks", loc="left")
    a1.legend(handles=[Line2D([], [], color=gcol[g], marker="o", lw=2, label=l) for g, l in
                       (("visual", "visual task"), ("text", "text-dense task"), ("none", "French tables"))]
              + [Line2D([], [], color=MUTED, marker="o", mfc=SURFACE, lw=0, label="hollow = n.s. after Holm")],
              loc="lower right", fontsize=7)
    offsets = {"synth_gov": (4, 5), "tatdqa": (6, -3), "synth_health": (-12, -13)}
    for t in fu:
        gap = fu[t]["metrics"]["B"]["ndcg_at_5"] - fu[t]["metrics"]["A"]["ndcg_at_5"]
        d = fu[t]["C_vs_B"]["delta"]
        a2.plot(gap, d, "o", color=gcol[fu[t]["group"]], ms=7, mec=SURFACE, mew=1.2)
        a2.annotate(TASK_NAMES[t], (gap, d), xytext=offsets.get(t, (4, 3)), textcoords="offset points",
                    fontsize=7, color=INK2)
    a2.axhline(0, color=BLUE, lw=1.2, ls="--")
    a2.set_xlabel("ColPali − BM25 (nDCG@5): how much weaker BM25 is")
    a2.set_ylabel("Δ nDCG@5, C − ColPali")
    a2.set_title("Post-hoc, descriptive: the gap predicts the loss", loc="left")
    fig.tight_layout()
    save(fig, "fig8_vidore_tasks")


def fig9_talk_summary() -> None:
    """Short-talk summary: DocVQA test nDCG@5 of the main systems, grouped by what the system adds."""
    t = load("results/test_evaluation/metrics.json")["results"]
    p2 = load("results/posthoc2/metrics.json")["results"]
    p4 = load("results/posthoc4/metrics.json")["results"]
    rows = [("BM25 (OCR text)", t["A"], ORANGE), ("Text RAG stack\n(hybrid + reranker)", p2["T_rr"], ORANGE),
            ("ColPali + BM25\n(RRF)", t["C"], ORANGE), ("ColPali v1.3", t["B"], BLUE),
            ("ColPali + Tomoro\n(RRF)", p4["F_tm"], AQUA), ("Tomoro ColQwen3-4B\n(newer model)", p4["TM"], GRAY_DARK)]
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    b = t["B"]["metrics"]["ndcg_at_5"]
    for x, (lbl, r, c) in enumerate(rows):
        v = r["metrics"]["ndcg_at_5"]
        lo, hi = r["ndcg_at_5_ci95"]
        ax.bar(x, v, color=c, width=0.62, edgecolor=SURFACE, linewidth=2)
        ax.plot([x, x], [lo, hi], color=INK2, lw=1.2)
        ax.text(x, hi + 0.015, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.axhline(b, color=BLUE, lw=1, ls="--")
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=7.6)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("nDCG@5 (95% CI)")
    ax.set_title("DocVQA test (225 queries): text signals do not help ColPali; a newer visual model does", loc="left")
    save(fig, "fig9_talk_summary")


def main() -> None:
    style()
    test = load("results/test_evaluation/metrics.json")
    fig1_test_results(test)
    fig2_k_sweep(load("results/fusion_docvqa/dev_selection.json"))
    fig3_weighted_heatmap(load("results/fusion_docvqa/dev_selection_weighted.json"))
    fig4_by_type(test)
    fa = load("results/failure_analysis/analysis.json")
    fig5_headroom(fa)
    fig6_overlap(fa)
    fig7_all_attempts()
    fig8_vidore_tasks()
    fig9_talk_summary()


if __name__ == "__main__":
    sys.exit(main())
