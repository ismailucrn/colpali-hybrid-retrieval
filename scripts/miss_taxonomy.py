"""Taxonomy of ColPali's misses (descriptive; plan and flag definitions fixed in docs/experiment_log.md, 2026-11-16,
commit fe3533b, before any count was computed).

No new system and no selection: it reads the ranks of B (ColPali), A (BM25), BGE-M3 and T_rr (post-hoc set 2, frozen
settings: T_hyb = RRF(A, BGE-M3) k=5, cross-encoder rerank of the top 50), all already evaluated. Dev and test are
reported separately.

A miss = gold page not in ColPali's top 5. Flags (not exclusive), computed for misses and for hits:
  duplicate_text    query text occurs on more than one dataset row (one arbitrary gold page under the official protocol)
  alt_page_in_top5  a page of another row with the same query text is in ColPali's top 5
  gold_ocr_short    gold-page OCR has fewer than 20 whitespace tokens
  generic_query     every BM25 query token occurs in the OCR of >= 25 pages (sensitivity: 10, 50), or no BM25 token
  no_overlap        no BM25 query token occurs in the gold page's OCR
  near_miss         gold rank 6-10
  text_finds_it     A, BGE-M3 or T_rr has the gold page in its top 5
Primary category (priority): duplicate_text > gold_ocr_short > generic_query > no_overlap > other.

Writes results/miss_taxonomy/: taxonomy.json, misses.csv (both splits), examples.md (test misses).

Usage:
    python scripts/miss_taxonomy.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.bm25 import Tokenizer  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, ranking, scores_to_run, subset  # noqa: E402
from src.fusion import rrf  # noqa: E402
from src.rerank import rerank_top_n  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402

SHORT_OCR = 20
GENERIC_DF = 25
GENERIC_DF_SENSITIVITY = [10, 25, 50]
CATEGORIES = ["duplicate_text", "gold_ocr_short", "generic_query", "no_overlap", "other"]
FLAGS = ["duplicate_text", "alt_page_in_top5", "gold_ocr_short", "generic_query", "no_overlap", "near_miss", "text_finds_it"]


def main() -> None:
    from datasets import load_dataset

    revision = "49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6"
    task = load_task("vidore/docvqa_test_subsampled", revision, "test",
                     text_dataset="vidore/docvqa_test_subsampled_tesseract",
                     text_revision="9a183a7237d678eaaf68dc0f640a42f46016802f")
    rows = load_dataset("vidore/docvqa_test_subsampled", split="test", revision=revision).select_columns(
        ["query", "image_filename"])
    pages_of_text = defaultdict(list)  # query text -> every page it occurs on
    for q, pid in zip(rows["query"], rows["image_filename"]):
        if q is not None:
            pages_of_text[q].append(pid)

    split = json.loads((REPO_ROOT / "results/splits/docvqa_split.json").read_text())
    tags = {r["query"]: r["tag"] for r in csv.DictReader(open(REPO_ROOT / "results/splits/docvqa_query_tags.csv"))}
    page_text = dict(zip(task.page_ids, task.page_texts))

    A = load_scores("results/bm25_docvqa/scores.npz", task.queries, task.page_ids)
    B = load_scores("results/colpali_v1.3_docvqa/scores.npz", task.queries, task.page_ids)
    BGE = load_scores("results/bge_m3_docvqa/scores.npz", task.queries, task.page_ids)
    CE = load_scores("results/reranker_docvqa/ce_scores.npz", task.queries, task.page_ids)
    T_rr = rerank_top_n(rrf([A, BGE], task.page_ids, k=5), CE, 50, task.page_ids)
    systems = {"A": A, "B": B, "BGE": BGE, "T_rr": T_rr}
    runs = {s: scores_to_run(m, task.queries, task.page_ids) for s, m in systems.items()}
    perq = {s: evaluate(task.qrels, runs[s])[1] for s in systems}
    rank = {s: {q: perq[s][q]["gold_rank"] for q in task.queries} for s in systems}

    tok = Tokenizer()
    page_tokens = {pid: set(tok(t)) for pid, t in page_text.items()}
    df = Counter(t for toks in page_tokens.values() for t in toks)

    def flags(q: str, generic_df: int = GENERIC_DF) -> dict:
        gold = next(iter(task.qrels[q]))
        qt = set(tok(q))
        top5_b = ranking(runs["B"][q])[:5]
        others = [p for p in pages_of_text[q] if p != gold]
        return {
            "duplicate_text": q in task.duplicated_queries,
            "alt_page_in_top5": any(p in top5_b for p in others),
            "gold_ocr_short": len(page_text[gold].split()) < SHORT_OCR,
            "generic_query": (not qt) or min(df[t] for t in qt) >= generic_df,
            "no_overlap": not (qt & page_tokens[gold]),
            "near_miss": 6 <= rank["B"][q] <= 10,
            "text_finds_it": any(rank[s][q] <= 5 for s in ("A", "BGE", "T_rr")),
        }

    def primary(f: dict) -> str:
        return next((c for c in CATEGORIES[:-1] if f[c]), "other")

    out = REPO_ROOT / "results/miss_taxonomy"
    out.mkdir(parents=True, exist_ok=True)
    record = collect_metadata({}, "none", "cpu", 0)
    record["label"] = "descriptive, plan fixed in fe3533b before counting; no new system, no selection"
    record["thresholds"] = {"short_ocr_tokens": SHORT_OCR, "generic_min_df_pages": GENERIC_DF,
                            "generic_sensitivity": GENERIC_DF_SENSITIVITY}
    csv_rows = []
    for name in ("dev", "test"):
        qs = split[name]
        miss = [q for q in qs if rank["B"][q] > 5]
        hit = [q for q in qs if rank["B"][q] <= 5]
        fl = {q: flags(q) for q in qs}
        cat = {q: primary(fl[q]) for q in miss}
        res = {"n": len(qs), "B_hits": len(hit), "B_misses": len(miss),
               "flag_rate": {f: {"misses": sum(fl[q][f] for q in miss), "hits": sum(fl[q][f] for q in hit)}
                             for f in FLAGS},
               "primary_category": {}, "generic_sensitivity": {}}
        for c in CATEGORIES:
            cq = [q for q in miss if cat[q] == c]
            res["primary_category"][c] = {
                "n": len(cq), "by_tag": dict(Counter(tags[q] for q in cq)),
                "text_finds_it": sum(fl[q]["text_finds_it"] for q in cq),
                "near_miss": sum(fl[q]["near_miss"] for q in cq),
                "alt_page_in_top5": sum(fl[q]["alt_page_in_top5"] for q in cq)}
        for d in GENERIC_DF_SENSITIVITY:
            res["generic_sensitivity"][str(d)] = {"misses": sum(flags(q, d)["generic_query"] for q in miss),
                                                  "hits": sum(flags(q, d)["generic_query"] for q in hit)}
        res["misses_by_tag"] = dict(Counter(tags[q] for q in miss))
        res["gold_rank_of_misses"] = {"6-10": sum(6 <= rank["B"][q] <= 10 for q in miss),
                                      "11-50": sum(11 <= rank["B"][q] <= 50 for q in miss),
                                      ">50": sum(rank["B"][q] > 50 for q in miss)}
        record[name] = res
        for q in miss:
            csv_rows.append([name, q, tags[q], next(iter(task.qrels[q])), cat[q], rank["B"][q], rank["A"][q],
                             rank["BGE"][q], rank["T_rr"][q], *[int(fl[q][f]) for f in FLAGS]])
        if name == "test":
            test_miss, test_cat, test_fl = miss, cat, fl

    with open(out / "misses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "query", "tag", "gold_page", "primary_category", "rank_B", "rank_A", "rank_BGE", "rank_T_rr",
                    *FLAGS])
        w.writerows(csv_rows)

    def snippet(pid: str, n: int = 140) -> str:
        t = " ".join(page_text[pid].split())
        return (t[:n] + "…") if len(t) > n else (t or "(empty OCR)")

    md = ["# ColPali's test misses by primary category (descriptive)\n",
          "Plan and flag definitions: `docs/experiment_log.md`, 2026-11-16 (`fe3533b`). Ranks are gold-page ranks.\n"]
    for c in CATEGORIES:
        cq = sorted([q for q in test_miss if test_cat[q] == c], key=lambda q: rank["B"][q])
        md.append(f"\n## {c} ({len(cq)})\n")
        for q in cq:
            gold = next(iter(task.qrels[q]))
            top1 = ranking(runs["B"][q])[0]
            extra = [f for f in ("alt_page_in_top5", "near_miss", "text_finds_it") if test_fl[q][f]]
            md.append(f"- **{q}** — `{tags[q]}`; rank B={rank['B'][q]}, A={rank['A'][q]}, BGE={rank['BGE'][q]}, "
                      f"T_rr={rank['T_rr'][q]}{'; ' + ', '.join(extra) if extra else ''}\n"
                      f"  - gold `{gold}`: “{snippet(gold)}”\n"
                      f"  - ColPali top-1 `{top1}`: “{snippet(top1)}”")
    (out / "examples.md").write_text("\n".join(md) + "\n")
    (out / "taxonomy.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(json.dumps({k: record[k] for k in ("dev", "test")}, indent=1))
    print(f"wrote {out.relative_to(REPO_ROOT)}/{{taxonomy.json, misses.csv, examples.md}}")


if __name__ == "__main__":
    main()
