# Experiment log

Chronological record of every run and decision. One entry per run or decision; never edit past entries — add a dated correction instead.

## Entry template

```
### YYYY-MM-DD — <short title>
- Type: run | decision | fix
- Commit: <git hash>
- Config: configs/<file>.yaml
- Command: <exact command>
- Environment: <hardware, CUDA, Python, key package versions>
- Checkpoint: <HF name @ revision>
- Dataset: <HF name @ revision>, split, #queries, #pages
- Seed: <seed>
- Output: results/<path>
- Result (measured): <numbers, or "n/a">
- Notes: <what happened, surprises, what to do next>
```

## Entries

### 2026-11-03 — Repository setup
- Type: decision
- Commit: see `git log` for this date
- Notes: created repository skeleton, README, MIT license, docs templates and `CLAUDE.md` (project context and protocol). Compressed the timeline to the remaining ~2 weeks: ablation E reduced to RRF `k` sensitivity; control D kept. No experiments run yet.

### 2026-11-03 — Smoke test written; library and data checks
- Type: decision
- Notes:
  - Checked the official `colpali-engine` README, PyPI and the HF Hub API. Latest `colpali-engine` is 0.3.18 and is deprecated in favour of Sentence Transformers v6; kept for reproducibility as the authors' reference code.
  - The paper's checkpoint `vidore/colpali` is listed as *not supported* by current `colpali-engine` (v1.1–v1.3 are). Reproduction checkpoint still to decide.
  - `vidore/colpali-v1.3` and its base model are not gated. ViDoRe v1 `_tesseract` datasets provide OCR page text (`text_description`).
  - Wrote `scripts/smoke_test.py`, `configs/smoke_test.yaml`, `notebooks/smoke_test_colab.ipynb`. Tested everything except model loading/forward on the Mac (Python 3.11, torch 2.13.0, transformers 5.22.1, colpali-engine 0.3.18): 8 queries / 8 pages streamed from `vidore/docvqa_test_subsampled` (sha `49bf8f1`), processor output shapes (image: 1031 tokens at 448×448), scoring, metadata. Model sha `vidore/colpali-v1.3` = `b5c6dd6`.
  - Next: full smoke test on Colab T4.

### 2026-11-03 — Smoke test on Colab (first GPU run)
- Type: run
- Commit: `a1c77ff` (clean)
- Config: `configs/smoke_test.yaml`
- Command: `python scripts/smoke_test.py --config configs/smoke_test.yaml` via `notebooks/smoke_test_colab.ipynb`
- Environment: Colab Pro, NVIDIA L4 (22.5 GB), CUDA 13.0, Python 3.13.16, torch 2.11.0+cu130, transformers 5.22.0, peft 0.20.0, colpali-engine 0.3.18 (full freeze: `results/smoke_test/pip_freeze_colab.txt`)
- Checkpoint: `vidore/colpali-v1.3` @ `b5c6dd6`, bf16
- Dataset: `vidore/docvqa_test_subsampled` @ `49bf8f1`, first 8 rows of `test` (8 queries, 8 unique pages)
- Seed: 0
- Output: `results/smoke_test/20261103_054838_cuda.json`
- Result (measured, sanity check only — not an evaluation): 7/8 queries rank their own page first. Page embedding 1031×128; query embeddings 23–24×128. Encoding 0.49 s/page and 0.017 s/query (batch 4); model load 15.6 s. Peak GPU memory 16.2 GB allocated / 16.9 GB reserved.
- Notes:
  - Engineering fixes needed before it ran: Colab's preinstalled torchao 0.10 breaks peft 0.20 on import → uninstalled in the notebook; results dir now created before `pip freeze`.
  - Picked L4 instead of T4 (user has Colab Pro): native bf16, so no fp16 deviation.
  - The miss is query 4, "What is the table number?": its scores are nearly flat across all 8 pages (15.31–15.56). *Interpretation:* the query is generic and could match any page with a table; a candidate example for the failure analysis.
  - *Hypothesis (untested):* the 16.2 GB peak is far above the ~6 GB of bf16 weights, probably because the PaliGemma forward also computes LM-head logits over the full vocabulary (and the processor passes `labels`). This could OOM on a 15 GB T4 at batch size 4. Check before the full indexing run (e.g. batch size, or dropping `labels`, verifying embeddings are unchanged).

### 2026-11-03 — Task chosen: ViDoRe v1 DocVQA; published scores and protocol checked
- Type: decision
- Notes:
  - Chose `vidore/docvqa_test_subsampled`: mixed query types already visible in the smoke test (numeric, names/codes, chart axes); OCR text available in the `_tesseract` variant; 500 pages / 451 queries is manageable.
  - **Published nDCG@5 (DocVQA):** paper arXiv 2407.01449v6 Table 2 — ColPali 54.4, BM25 (Unstructured + OCR) 36.8, BGE-M3 (Unstructured + OCR) 25.7. Model-card `results.json`: v1.0 0.5443, v1.1 0.5499, v1.2 0.5717, v1.3 0.5845. The v1.0 model card matches the paper (54.4).
  - **Measured data stats:** 500 rows, 500 unique pages, 451 unique queries, 0 null queries; 22 query texts appear on several different pages (e.g. "What is the page number?" ×12). `_tesseract` text has the same row order; 4 pages have empty text; median 134 words/page.
  - **Official protocol** read from `vidore-benchmark` (commit `de1d88c`, `ViDoReEvaluatorQA`): queries deduplicated; one relevant page per query, taken from the query's last occurrence; metrics via MTEB `RetrievalEvaluator`. We will reproduce this exactly for the published-vs-measured comparison.
  - Open: reproduction checkpoint (v1.3 vs v1.0).

### 2026-11-03 — System B reproduction: ColPali v1.3 on DocVQA (full set)
- Type: run (registry rows 7–10)
- Commit: `0a716fa`; config `configs/colpali_v1.3_docvqa.yaml`; notebook `notebooks/colpali_docvqa_colab.ipynb`
- Environment: Colab Pro, NVIDIA L4, bf16, Python 3.13.16, torch 2.11.0+cu130, transformers 5.22.0, colpali-engine 0.3.18
- Checkpoint: `vidore/colpali-v1.3` @ `b5c6dd6` on base `vidore/colpaligemma-3b-pt-448-base` @ `30ab955`
- Dataset: `vidore/docvqa_test_subsampled` @ `49bf8f1` — 451 queries, 500 pages; official protocol
- Output: `results/colpali_v1.3_docvqa/` (`scores.npz`, `metrics.json`, `per_query.csv`, console logs)
- **Result (measured):** nDCG@5 **0.5752**; Recall@5 0.6497; Recall@10 0.6896; nDCG@10 0.5882; MRR@10 0.5559. Post-hoc, without the 22 duplicated query texts (429 queries): nDCG@5 0.6011.
- **Published:** model card v1.3 nDCG@5 0.5845; paper Table 2 (v1.0) 54.4.
- **Gap:** −0.93 nDCG@5 points vs the model card — inside the proposed ±2-point tolerance (tolerance not yet confirmed by the user).
  - *Interpretation / hypotheses (untested):* same checkpoint and protocol, but different library versions (transformers 5.22 vs whatever produced the model card), hardware and bf16 kernels; bf16 scores have many ties, so tiny numeric changes can reorder near-tied pages. The model card number is self-reported and its exact environment is unknown. A check, if time allows: rerun with older transformers/colpali-engine versions.
  - Our number is above the paper's v1.0 value (54.4), consistent with v1.3 being a later, stronger checkpoint (model cards: 54.4 → 58.5).
- **Resources (measured):** indexing 111.7 s for 500 pages (0.22 s/page, batch 4); single-query latency median 51.3 ms (encode 44.9 + MaxSim vs 500 pages 6.4), p95 53.3 ms; index 126 MB (515,500 × 128 bf16 vectors); peak GPU memory 16.19 GB allocated (same as the smoke test, so it does not grow with corpus size — consistent with the LM-head-logits hypothesis, still untested).
- Verification: `scores.npz` re-evaluated locally on CPU with `scripts/evaluate_scores.py` → identical metrics.

### 2026-11-03 — Query tags and dev/test split frozen
- Type: decision + run (registry row 11)
- Notes:
  - Tags are rule-based on query text only (`src/query_types.py`, rules in `docs/query_tagging.md`): visual 35, exact 186, general 230.
  - Stratified 50/50 split, seed 0: dev 226, test 225. Committed before any run of A, C, D, E.
  - Changed plan vs the draft: tags are fully automatic instead of manual, for reproducibility and to rule out outcome-dependent tagging.
  - *Limitation:* only ~17 visual queries per split.
