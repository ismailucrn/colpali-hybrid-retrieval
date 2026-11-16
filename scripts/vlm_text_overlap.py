"""Post-hoc set 3, H3 (descriptive, pre-registered in e70f2a7): does a cleaner page text change the miss taxonomy?

Recomputes the `no_overlap` flag of scripts/miss_taxonomy.py (no BM25 query token occurs in the gold page's text) with
the Qwen3.5-4B transcription instead of the Tesseract OCR, for ColPali's misses and hits on dev and test. ColPali's ranks
are unchanged (it reads the image), so this only asks whether the residual misses are an OCR artefact. No system is
evaluated here.

Writes results/miss_taxonomy/vlm_text_overlap.json.

Usage:
    python scripts/vlm_text_overlap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.bm25 import Tokenizer  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402


def main() -> None:
    task = load_task("vidore/docvqa_test_subsampled", "49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6", "test",
                     text_dataset="vidore/docvqa_test_subsampled_tesseract",
                     text_revision="9a183a7237d678eaaf68dc0f640a42f46016802f")
    vlm = json.loads((REPO_ROOT / "results/vlm_ocr_docvqa/page_texts.json").read_text())
    texts = {"tesseract": dict(zip(task.page_ids, task.page_texts)), "qwen3.5": vlm}
    split = json.loads((REPO_ROOT / "results/splits/docvqa_split.json").read_text())
    B = load_scores("results/colpali_v1.3_docvqa/scores.npz", task.queries, task.page_ids)
    rank = {q: r["gold_rank"] for q, r in evaluate(task.qrels, scores_to_run(B, task.queries, task.page_ids))[1].items()}
    tok = Tokenizer()

    record = collect_metadata({}, "none", "cpu", 0)
    record["label"] = "descriptive (post-hoc set 3, H3); ColPali ranks unchanged; no system evaluated"
    for name in ("dev", "test"):
        miss = [q for q in split[name] if rank[q] > 5]
        hit = [q for q in split[name] if rank[q] <= 5]
        res = {"n_misses": len(miss), "n_hits": len(hit)}
        for src, pages in texts.items():
            tokens = {pid: set(tok(t)) for pid, t in pages.items()}

            def no_overlap(q: str) -> bool:
                return not (set(tok(q)) & tokens[next(iter(task.qrels[q]))])

            res[src] = {"no_overlap_misses": sum(map(no_overlap, miss)), "no_overlap_hits": sum(map(no_overlap, hit))}
        both = [q for q in miss if not (set(tok(q)) & set(tok(texts["tesseract"][next(iter(task.qrels[q]))])))]
        res["misses_no_overlap_tesseract_but_overlap_qwen"] = sum(
            bool(set(tok(q)) & set(tok(vlm[next(iter(task.qrels[q]))]))) for q in both)
        record[name] = res
        print(name, json.dumps(res))
    out = REPO_ROOT / "results/miss_taxonomy/vlm_text_overlap.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
