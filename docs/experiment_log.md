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

### 2026-11-04 — System A: BM25 over Tesseract OCR text (full set)
- Type: run (registry rows 12–13)
- Implementation: identical to vidore-benchmark `BM25Retriever` (NLTK `word_tokenize`, alphanumeric tokens only, lowercase, NLTK English stopwords; `rank_bm25.BM25Okapi` defaults k1=1.5, b=0.75 — no tuning). Text: `vidore/docvqa_test_subsampled_tesseract` @ `9a183a7`, `text_description`.
- **Result (measured):** full set nDCG@5 **0.3907**; Recall@5 0.4501; Recall@10 0.4900; nDCG@10 0.4036; MRR@10 0.3764. Dev nDCG@5 0.3995. Post-hoc without duplicated queries 0.4093. Test-split metrics not computed (touched once, at the end).
- **Published:** paper Table 2, BM25 on Unstructured + OCR **36.8**; Unstructured text-only 34.1.
- *Interpretation:* +2.3 points over the paper's BM25, but the text differs (Tesseract dump vs Unstructured layout-aware chunks), so this is a deviation, not a reproduction of 36.8. Same order of magnitude; ColPali (0.5752) is far ahead, as in the paper.
- **Resources (measured, Mac M5 CPU):** index build 0.35 s for 500 pages; single-query latency median 0.12 ms; index 0.63 MB pickled (56,470 tokens, 14,573 terms); peak process RSS 922 MB (mostly the dataset/Python, not BM25).
- **Observation relevant to the hypothesis:** the official preprocessing removes some lexical anchors — non-alphanumeric tokens (`Log-in`, `B&W`, `2012-2013`) and stopwords such as "no" are dropped; 2 queries become empty (all-zero scores). So some `exact`-tagged queries cannot benefit from BM25 by construction. We keep the official preprocessing (comparability with the paper) and will report this in the failure analysis; a different tokenizer would be a separate, unplanned change.
- 4 pages have empty OCR text → BM25 can never retrieve them.

### 2026-11-04 — Pre-registration: control D uses BGE-M3 (before any BGE-M3 or fusion result)
- Type: decision (confirmed by the user)
- Dense text retriever for control D: `BAAI/bge-m3` @ `5617a9f`, dense vectors only, implemented like vidore-benchmark `BGEM3Retriever` (FlagEmbedding, fp16, query max_length 512, dot-product scoring), over the same `vidore/docvqa_test_subsampled_tesseract` text as BM25 (`9a183a7`).
- Why this model: it is the paper's own dense text baseline (Table 2: 25.7 nDCG@5 on DocVQA with Unstructured + OCR), so the control is not chosen to make C look good or bad.
- Fusion (C and D) will use the same RRF formula and the same `k` search on dev; neither BM25 nor BGE-M3 gets any tuning of its own.
- No BGE-M3 scores and no fusion scores exist at the time of this commit.

### 2026-11-04 — Dense text component of control D: BGE-M3 (full set)
- Type: run (registry rows 14–15)
- Setup: `BAAI/bge-m3` @ `5617a9f`, FlagEmbedding 1.4.2 `BGEM3FlagModel`, dense vectors, fp16 on Apple MPS (Mac M5), query max_length 512, **passage max_length 8192** (explicit: FlagEmbedding's default changed from 8192 in 1.2.x to 512 in 1.4.x; 8192 keeps whole pages, matching the text BM25 sees), batch size 4, dot product. Same `_tesseract` text as BM25 (`9a183a7`).
- **Result (measured):** full set nDCG@5 **0.2486**; Recall@5 0.2927; Recall@10 0.4058; nDCG@10 0.2850; MRR@10 0.2495. Dev 0.2351. Post-hoc without duplicated queries 0.2604.
- **Published:** paper Table 2, BGE-M3 on Unstructured + OCR **25.7**.
- *Interpretation:* within ~1 point of the paper despite different OCR text and hardware; like the paper, the dense text retriever is clearly below BM25 on DocVQA (0.2486 vs 0.3907), plausibly because noisy OCR hurts a semantic encoder more than term matching *(hypothesis)*.
- **Resources (measured, Mac M5 MPS):** indexing 21.1 s for 500 pages (0.04 s/page); single-query latency median 29.0 ms (encode on MPS + numpy dot product); index 1.95 MB (500 × 1024 float32); peak MPS driver memory 2.14 GB (sampled).
- Deviation: run on Apple MPS (fp16), while ColPali ran on an NVIDIA L4. Latencies of the components are therefore not directly comparable across devices; this is noted for the resource comparison.

### 2026-11-04 — Dev selection for C and D: equal-weight RRF hurts ColPali on dev
- Type: run (registry rows 16–17), **dev split only — test not touched**
- Grid k ∈ {1, 5, 10, 20, 40, 60, 80, 100, 200, 500}, same for C and D; rule: max dev nDCG@5, ties → closest to 60.
- **Measured (dev, 226 queries, nDCG@5):** B ColPali 0.5804; A BM25 0.3995; BGE-M3 0.2351. C = RRF(BM25, ColPali): 0.5335 at k=1 (best), 0.4669 at k=60. D = RRF(BGE-M3, ColPali): 0.5263 at k=1, 0.4219 at k=60. **Selected k = 1 for both** — at the lower edge of the grid; scores fall monotonically as k grows.
- **By query type (dev, nDCG@5):**

  | system | all (226) | exact (93) | general (115) | visual (18) |
  |---|---|---|---|---|
  | A BM25 | 0.3995 | 0.6437 | 0.2559 | 0.0556 |
  | BGE-M3 | 0.2351 | 0.4046 | 0.1181 | 0.1073 |
  | B ColPali | 0.5804 | 0.8283 | 0.4105 | 0.3850 |
  | C k=1 | 0.5335 | 0.7858 | 0.3737 | 0.2513 |
  | D k=1 | 0.5263 | 0.7762 | 0.3575 | 0.3138 |

- *Interpretation:*
  - Pre-registered P1 (C > B) is **not supported on dev**: equal-weight RRF with a much weaker second retriever pulls good ColPali rankings down. Smaller k helps because it lets each system's top ranks dominate, which favours ColPali when it is right.
  - P2's premise is **contradicted on dev**: ColPali is already strongest on `exact` queries (0.83 vs BM25 0.64), so there is little lexical headroom; BM25 is nearly useless on `visual` queries (0.06), and fusion hurts there most — the "least help on visual" half of P2 holds, as harm.
  - P3 (C > D): on dev C is slightly above D overall (0.5335 vs 0.5263) and on exact/general, below on visual; both below B, so the comparison is between two harmful fusions.
- k = 1 is now frozen for the test run of C and D. Any further variant (e.g. weighting) is a dev-informed change and must be labelled as such.

### 2026-11-04 — Pre-registration of ablation E: weighted RRF (dev-informed, before running it)
- Type: decision (user approved), **made after seeing the dev results above** — labelled dev-informed everywhere.
- Method: `fused = w / (k + rank_ColPali) + (1 − w) / (k + rank_text)`; systems C_w (BM25 text) and D_w (BGE-M3 text), same grid and rule for both.
- Grid: w ∈ {0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0} (w = 0.5 is plain RRF, w = 1.0 is ColPali alone, so "no fusion" is a possible outcome); k ∈ {1, 5, 10, 20, 40, 60, 80, 100, 200, 500}. 90 configurations per system.
- Rule: max dev nDCG@5; ties → w closest to 0.5, then k closest to 60.
- Rationale (stated before running): equal weights let a much weaker retriever override ColPali; a small text weight could still break ties or rescue near-misses where BM25 is confident (e.g. `exact` queries) without overriding ColPali elsewhere. *Hypothesis:* the best dev w is high (≥ 0.8); any gain over B is small.
- Risk: 90 configurations on 226 dev queries → the selected dev score is optimistically biased; only the single test run counts.
- The pre-registered C and D (equal weight, k = 1) remain the primary result; C_w/D_w are reported alongside, clearly as an ablation.

### 2026-11-04 — Ablation E on dev: weighted RRF
- Type: run (registry row 18), dev only; grid and rule committed beforehand (`b558ceb`).
- **Measured (dev nDCG@5; B ColPali = 0.5804):**
  - C_w = wRRF(BM25, ColPali): best **0.5851** at w = 0.95, k = 20 (+0.47 pt over B). Near the top, w = 0.85–0.95 with k = 5–20 gives 0.580–0.585; w = 0.5 (plain RRF) gives ≤ 0.5335.
  - D_w = wRRF(BGE-M3, ColPali): best 0.5804 — equal to B; no configuration beats ColPali alone. Selected by the tie rule: w = 0.95, k = 10.
- *Interpretation:* the text retrievers only help (if at all) when they are allowed to reorder ColPali's ranking marginally; any weight that lets them override ColPali hurts. BM25 shows a small dev gain that BGE-M3 does not — consistent with P3's direction, but +0.47 pt after a 90-config search is within what selection bias alone could produce *(hypothesis; the test run and paired test will tell)*.
- **Frozen for the single test run:** C k=1; D k=1; C_w (w=0.95, k=20); D_w (w=0.95, k=10); plus A, B and BGE-M3 alone.

### 2026-11-04 — Test-evaluation plan (committed before the single test run)
- Script `scripts/run_test_evaluation.py`, config `configs/test_evaluation.yaml`; it refuses to run if the fusion settings differ from the dev selection files.
- Statistics, decided before seeing test: per-system percentile bootstrap 95% CI of mean nDCG@5 (10,000 resamples over queries, seed 0); paired comparisons vs B with paired bootstrap CI of Δ and a two-sided sign-flip permutation test (10,000, seed 0); Holm correction over the six comparisons vs B; P3 pairs C vs D and C_w vs D_w reported separately; per-type breakdown exploratory and uncorrected; post-hoc analysis without the 22 duplicated query texts.
- Resources: per-component measurements reused from each system's run (different devices — stated in the table), fusion overhead measured per query; sequential and parallel latency reported.
- Dry run on dev (registry row 19) reproduced all dev numbers; on dev, C_w vs B is +0.0047 with p = 0.25 — even on the data it was tuned on, the gain is not significant.
