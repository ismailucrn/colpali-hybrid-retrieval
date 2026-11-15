"""POST-HOC set 2 (descriptive): does retrieval success depend on the OCR quality of the gold page?

OCR-quality proxy per page (pre-registered 2026-11-15, commit 92c2163): share of alphabetic OCR tokens
(lowercased, regex [a-z]+, length >= 2) that are in the NLTK `words` English word list. Pages with empty OCR text
get no proxy and are reported separately.

Full query set (451; only systems without tuned parameters: A = BM25, BGE-M3, B = ColPali). Queries are grouped
by their gold page's proxy into terciles; per group Hit@5 and nDCG@5 per system. Spearman rho between the proxy and
per-query nDCG@5 per system, with a percentile bootstrap 95 % CI over queries.

Writes to results/ocr_quality/: ocr_quality.json, page_quality.csv, summary.md.

Usage:
    python scripts/ocr_quality.py --config configs/posthoc2_docvqa.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import nltk
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.bm25 import NLTK_DIR  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402

WORD = re.compile(r"[a-z]+")
SYSTEMS = {"A": "BM25", "BGE": "BGE-M3", "B": "ColPali"}


def english_words() -> set[str]:
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    try:
        nltk.data.find("corpora/words")
    except LookupError:
        nltk.download("words", download_dir=str(NLTK_DIR), quiet=True)
    from nltk.corpus import words

    return {w.lower() for w in words.words()}


def page_proxy(text: str, vocab: set[str]) -> tuple[float | None, int]:
    tokens = [t for t in WORD.findall(text.lower()) if len(t) >= 2]
    if not tokens:
        return None, 0
    return sum(t in vocab for t in tokens) / len(tokens), len(tokens)


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks; tied values get the average of the positions they occupy."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2.0)[inverse]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = average_ranks(x), average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        vals.append(spearman(x[idx], y[idx]))
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/posthoc2_docvqa.yaml")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    dcfg = config["data"]

    task = load_task(dcfg["dataset"], dcfg["revision"], dcfg["split"],
                     text_dataset=dcfg["text_dataset"], text_revision=dcfg["text_revision"])
    vocab = english_words()
    proxy, n_tok = {}, {}
    for pid, text in zip(task.page_ids, task.page_texts):
        proxy[pid], n_tok[pid] = page_proxy(text, vocab)
    valid = np.array([v for v in proxy.values() if v is not None])
    empty = [pid for pid, v in proxy.items() if v is None]

    perq = {}
    for sid in SYSTEMS:
        s = load_scores(config["scores"][sid], task.queries, task.page_ids)
        perq[sid] = evaluate(task.qrels, scores_to_run(s, task.queries, task.page_ids))[1]

    gold = {q: next(iter(task.qrels[q])) for q in task.queries}
    with_proxy = [q for q in task.queries if proxy[gold[q]] is not None]
    no_proxy = [q for q in task.queries if proxy[gold[q]] is None]
    x = np.array([proxy[gold[q]] for q in with_proxy])
    cuts = np.quantile(x, [1 / 3, 2 / 3])
    group_of = {q: ("low" if proxy[gold[q]] <= cuts[0] else "mid" if proxy[gold[q]] <= cuts[1] else "high") for q in with_proxy}
    groups = {g: [q for q in with_proxy if group_of[q] == g] for g in ("low", "mid", "high")}
    groups["empty OCR"] = no_proxy

    table = {}
    for g, qs in groups.items():
        row = {"n": len(qs), "proxy_range": None}
        if qs and g != "empty OCR":
            vals = [proxy[gold[q]] for q in qs]
            row["proxy_range"] = [float(min(vals)), float(max(vals))]
        for sid in SYSTEMS:
            row[f"{sid}_hit5"] = float(np.mean([perq[sid][q]["gold_rank"] <= 5 for q in qs])) if qs else None
            row[f"{sid}_ndcg5"] = float(np.mean([perq[sid][q]["ndcg_at_5"] for q in qs])) if qs else None
        table[g] = row

    corr = {}
    for sid in SYSTEMS:
        y = np.array([perq[sid][q]["ndcg_at_5"] for q in with_proxy])
        corr[sid] = {"spearman_rho": spearman(x, y), "ci95": bootstrap_spearman(x, y, args.bootstrap, config["seed"])}

    out = REPO_ROOT / "results/ocr_quality"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "page_quality.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page_id", "alpha_tokens", "dictionary_share"])
        for pid in task.page_ids:
            w.writerow([pid, n_tok[pid], "" if proxy[pid] is None else round(proxy[pid], 6)])

    record = collect_metadata(config, args.config, "cpu", config["seed"])
    record.update({
        "label": "POST-HOC set 2 (descriptive): OCR-quality proxy vs retrieval success, full query set",
        "proxy": "share of alphabetic OCR tokens ([a-z]+, len >= 2, lowercased) in the NLTK words list",
        "vocabulary_size": len(vocab),
        "pages": {"n": len(task.page_ids), "empty_ocr": empty, "median": float(np.median(valid)),
                  "p10": float(np.percentile(valid, 10)), "p90": float(np.percentile(valid, 90)),
                  "min": float(valid.min()), "max": float(valid.max())},
        "queries": {"n": len(task.queries), "with_proxy": len(with_proxy), "gold_page_empty_ocr": len(no_proxy),
                    "tercile_cuts": [float(c) for c in cuts]},
        "by_tercile": table,
        "spearman_vs_ndcg5": corr,
        "bootstrap_resamples": args.bootstrap,
    })
    (out / "ocr_quality.json").write_text(json.dumps(record, indent=2))

    lines = ["# OCR quality vs retrieval (POST-HOC set 2, descriptive, full set n = 451)", "",
             f"Proxy: share of alphabetic OCR tokens found in the NLTK English word list ({len(vocab)} words).",
             f"Pages: median {np.median(valid):.3f}, 10th–90th percentile {np.percentile(valid, 10):.3f}–"
             f"{np.percentile(valid, 90):.3f}; {len(empty)} pages with empty OCR.", "",
             "| Gold-page proxy group | n | proxy range | " + " | ".join(f"{n} Hit@5" for n in SYSTEMS.values()) + " | "
             + " | ".join(f"{n} nDCG@5" for n in SYSTEMS.values()) + " |",
             "|---|---|---|" + "---|" * (2 * len(SYSTEMS))]
    for g, r in table.items():
        rng = f"{r['proxy_range'][0]:.3f}–{r['proxy_range'][1]:.3f}" if r["proxy_range"] else "–"
        cells = [f"{r[f'{s}_hit5']:.3f}" if r[f"{s}_hit5"] is not None else "–" for s in SYSTEMS] + \
                [f"{r[f'{s}_ndcg5']:.3f}" if r[f"{s}_ndcg5"] is not None else "–" for s in SYSTEMS]
        lines.append(f"| {g} | {r['n']} | {rng} | " + " | ".join(cells) + " |")
    lines += ["", "| System | Spearman ρ (proxy vs nDCG@5) | 95% CI |", "|---|---|---|"]
    for sid, c in corr.items():
        lines.append(f"| {SYSTEMS[sid]} | {c['spearman_rho']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
