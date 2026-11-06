"""Failure analysis on the TEST split (descriptive, after the single test run; nothing here changes a system).

Questions:
  1. Headroom: how often does BM25 find the gold page when ColPali misses it (an upper bound on what
     fusion could gain), and how often is it the other way round (what fusion can lose)?
  2. Rank transitions: where do the gold pages move when ColPali is fused with BM25 (C, C_w)?
  3. Mechanism: does lexical overlap between query and gold-page OCR text explain when BM25 succeeds?
  4. Concrete examples: every query where C or C_w differs from B, with ranks from each system.

Writes results/failure_analysis/: analysis.json, changed_queries.csv, examples.md.

Usage:
    python scripts/failure_analysis.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.bm25 import Tokenizer  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, ranking, scores_to_run, subset  # noqa: E402
from src.fusion import rrf, weighted_rrf  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402

TAGS = ["exact", "general", "visual"]
BUCKETS = [(1, 1, "1"), (2, 5, "2-5"), (6, 10, "6-10"), (11, 10**9, ">10")]


def bucket(rank: int) -> str:
    return next(name for lo, hi, name in BUCKETS if lo <= rank <= hi)


def main() -> None:
    task = load_task("vidore/docvqa_test_subsampled", "49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6", "test",
                     text_dataset="vidore/docvqa_test_subsampled_tesseract",
                     text_revision="9a183a7237d678eaaf68dc0f640a42f46016802f")
    test = json.loads((REPO_ROOT / "results/splits/docvqa_split.json").read_text())["test"]
    tags = {r["query"]: r["tag"] for r in csv.DictReader(open(REPO_ROOT / "results/splits/docvqa_query_tags.csv"))}
    page_text = dict(zip(task.page_ids, task.page_texts))

    A = load_scores("results/bm25_docvqa/scores.npz", task.queries, task.page_ids)
    B = load_scores("results/colpali_v1.3_docvqa/scores.npz", task.queries, task.page_ids)
    BGE = load_scores("results/bge_m3_docvqa/scores.npz", task.queries, task.page_ids)
    systems = {"A": A, "B": B, "BGE": BGE,
               "C": rrf([A, B], task.page_ids, k=1), "D": rrf([BGE, B], task.page_ids, k=1),
               "C_w": weighted_rrf([B, A], [0.95, 0.05], task.page_ids, k=20)}
    runs = {s: scores_to_run(m, task.queries, task.page_ids) for s, m in systems.items()}
    perq = {s: evaluate(*subset(task.qrels, runs[s], test))[1] for s in systems}
    rank = {s: {q: perq[s][q]["gold_rank"] for q in test} for s in systems}
    ndcg = {s: {q: perq[s][q]["ndcg_at_5"] for q in test} for s in systems}

    tok = Tokenizer()

    def overlap(q: str) -> float | None:
        """Fraction of the query's BM25 tokens that occur in the gold page's OCR tokens."""
        qt = set(tok(q))
        if not qt:
            return None
        gold = next(iter(task.qrels[q]))
        return len(qt & set(tok(page_text[gold]))) / len(qt)

    ov = {q: overlap(q) for q in test}

    # 1. Headroom at cut-off 5 (success = gold page in the top 5).
    def hit(s, q):
        return rank[s][q] <= 5

    headroom = {}
    for group, qs in {"all": test, **{t: [q for q in test if tags[q] == t] for t in TAGS}}.items():
        headroom[group] = {
            "n": len(qs),
            "B_hit": sum(hit("B", q) for q in qs),
            "A_hit": sum(hit("A", q) for q in qs),
            "A_hit_B_miss (max gain)": sum(hit("A", q) and not hit("B", q) for q in qs),
            "B_hit_A_miss (risk)": sum(hit("B", q) and not hit("A", q) for q in qs),
            "both_miss": sum(not hit("A", q) and not hit("B", q) for q in qs),
            "oracle_A_or_B_hit": sum(hit("A", q) or hit("B", q) for q in qs),
        }

    # 2. Rank transitions B -> C and B -> C_w.
    transitions = {}
    for s in ("C", "C_w", "D"):
        t = Counter((bucket(rank["B"][q]), bucket(rank[s][q])) for q in test)
        transitions[f"B->{s}"] = {f"{a} -> {b}": n for (a, b), n in sorted(t.items())}

    # 3. Lexical overlap vs BM25 / ColPali success.
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return (float(np.mean(xs)), len(xs)) if xs else (None, 0)

    overlap_stats = {
        "A_hit": mean([ov[q] for q in test if hit("A", q)]),
        "A_miss": mean([ov[q] for q in test if not hit("A", q)]),
        "B_hit_A_miss": mean([ov[q] for q in test if hit("B", q) and not hit("A", q)]),
        "A_hit_B_miss": mean([ov[q] for q in test if hit("A", q) and not hit("B", q)]),
        "queries_with_no_bm25_tokens": sum(ov[q] is None for q in test),
        "gold_pages_with_empty_ocr": sum(not page_text[next(iter(task.qrels[q]))].strip() for q in test),
    }
    by_overlap = {}
    for lo, hi, name in [(0.0, 0.0, "0"), (1e-9, 0.5, "(0, 0.5]"), (0.5 + 1e-9, 0.999, "(0.5, 1)"), (1.0, 1.0, "1")]:
        qs = [q for q in test if ov[q] is not None and lo <= ov[q] <= hi]
        by_overlap[name] = {"n": len(qs), "A_hit_rate": float(np.mean([hit("A", q) for q in qs])) if qs else None,
                            "B_hit_rate": float(np.mean([hit("B", q) for q in qs])) if qs else None}

    # 4. Changed queries (C or C_w differs from B on nDCG@5).
    out = REPO_ROOT / "results/failure_analysis"
    out.mkdir(parents=True, exist_ok=True)
    changed = [q for q in test if ndcg["C"][q] != ndcg["B"][q] or ndcg["C_w"][q] != ndcg["B"][q]]
    with open(out / "changed_queries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "tag", "duplicated", "gold_page", "overlap", "rank_A", "rank_BGE", "rank_B", "rank_C",
                    "rank_C_w", "ndcg5_B", "ndcg5_C", "ndcg5_C_w", "top1_A", "top1_B", "top1_C"])
        for q in changed:
            w.writerow([q, tags[q], int(q in task.duplicated_queries), next(iter(task.qrels[q])),
                        None if ov[q] is None else round(ov[q], 3), rank["A"][q], rank["BGE"][q], rank["B"][q],
                        rank["C"][q], rank["C_w"][q], round(ndcg["B"][q], 4), round(ndcg["C"][q], 4),
                        round(ndcg["C_w"][q], 4), ranking(runs["A"][q])[0], ranking(runs["B"][q])[0],
                        ranking(runs["C"][q])[0]])

    def snippet(pid: str, n: int = 160) -> str:
        t = " ".join(page_text[pid].split())
        return (t[:n] + "…") if len(t) > n else (t or "(empty OCR)")

    def example(q: str) -> str:
        gold = next(iter(task.qrels[q]))
        top_a = ranking(runs["A"][q])[0]
        return (f"- **{q}** — tag `{tags[q]}`{' (duplicated text)' if q in task.duplicated_queries else ''}; "
                f"gold `{gold}`; gold rank A={rank['A'][q]}, B={rank['B'][q]}, C={rank['C'][q]}, C_w={rank['C_w'][q]}; "
                f"query–gold OCR overlap {ov[q] if ov[q] is None else round(ov[q], 2)}.\n"
                f"  - gold-page OCR: “{snippet(gold)}”\n"
                + (f"  - BM25 top-1 `{top_a}` OCR: “{snippet(top_a)}”\n" if top_a != gold else ""))

    c_wins = sorted([q for q in test if ndcg["C"][q] > ndcg["B"][q]], key=lambda q: ndcg["B"][q] - ndcg["C"][q])
    c_losses = sorted([q for q in test if ndcg["C"][q] < ndcg["B"][q]], key=lambda q: ndcg["C"][q] - ndcg["B"][q])
    cw_changed = [q for q in test if ndcg["C_w"][q] != ndcg["B"][q]]
    loss_types = Counter(tags[q] for q in c_losses)
    loss_mech = Counter("BM25 also hit (top 5)" if hit("A", q) else "BM25 missed (gold > 5)" for q in c_losses)

    md = ["# Failure analysis — test split (post-hoc, descriptive)\n",
          f"C = RRF(BM25, ColPali) k=1 vs B = ColPali: {len(c_wins)} wins, {len(c_losses)} losses (nDCG@5).",
          f"C losses by tag: {dict(loss_types)}; by mechanism: {dict(loss_mech)}.\n",
          "## Favourable: queries where C beats ColPali (all)\n", *[example(q) for q in c_wins],
          "\n## Unfavourable: the 10 largest losses of C vs ColPali\n", *[example(q) for q in c_losses[:10]],
          "\n## Every query where C_w (weighted, w=0.95, k=20) differs from ColPali\n", *[example(q) for q in cw_changed]]
    (out / "examples.md").write_text("\n".join(md) + "\n")

    record = collect_metadata({}, "none", "cpu", 0)
    record.update({"n_test": len(test), "headroom_at_5": headroom, "rank_transitions": transitions,
                   "lexical_overlap": overlap_stats, "hit_rate_by_overlap": by_overlap,
                   "C_vs_B": {"wins": len(c_wins), "losses": len(c_losses), "loss_tags": dict(loss_types),
                              "loss_mechanism": dict(loss_mech)},
                   "C_w_changed_queries": len(cw_changed)})
    (out / "analysis.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))

    print(json.dumps({k: record[k] for k in ("headroom_at_5", "rank_transitions", "lexical_overlap",
                                              "hit_rate_by_overlap", "C_vs_B", "C_w_changed_queries")}, indent=1))
    print(f"wrote {out}/{{analysis.json, changed_queries.csv, examples.md}}")


if __name__ == "__main__":
    main()
