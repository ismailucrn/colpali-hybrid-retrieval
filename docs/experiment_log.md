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

### 2026-11-04 — SINGLE TEST RUN: results and verdict on the pre-registered predictions
- Type: run (registry row 20). Commit `deec1d8`; config `configs/test_evaluation.yaml`; outputs `results/test_evaluation/` (`summary.md` has all tables). Test split n = 225 (exact 93, general 115, visual 17). Run once; nothing tuned afterwards.
- **Measured (test, nDCG@5 [95% bootstrap CI]):** A BM25 0.3818 [0.323, 0.441]; BGE-M3 0.2622 [0.209, 0.317]; **B ColPali 0.5699 [0.510, 0.628]**; C 0.5162 [0.457, 0.574]; D 0.5187 [0.460, 0.578]; C_w 0.5706 [0.511, 0.629]; D_w 0.5699 (identical ranking to B in the top 5 for all 225 queries).
- **Paired vs B (Δ nDCG@5 [95% CI], permutation p, Holm p, W/T/L):**
  - C: −0.054 [−0.076, −0.032], p 0.0001, Holm 0.0006, 9/167/49.
  - D: −0.051 [−0.072, −0.030], p 0.0001, Holm 0.0006, 4/171/50.
  - C_w: +0.001 [−0.002, +0.005], p 0.94, Holm 1.0, 2/220/3.
  - D_w: 0, 0/225/0.
- **P3 pairs:** C vs D −0.003 [−0.027, +0.021], p 0.84; C_w vs D_w +0.001, p 0.94.
- **Verdict on pre-registered predictions:**
  - **P1 (C > B): rejected.** Equal-weight RRF with BM25 is significantly *worse* than ColPali alone (−5.4 pt), confirming the dev finding on unseen queries.
  - **P2 (gain concentrated on exact, little on visual): not supported.** There is no gain anywhere; C loses on exact (−0.059), general (−0.046) and visual (−0.074, n = 17, not significant). ColPali is strongest on `exact` queries (0.833 vs BM25 0.617), so the premise that ColPali is weak on lexical-match queries does not hold on DocVQA.
  - **P3 (C > D): not supported.** C and D are indistinguishable (p 0.84); the harm comes from ensembling with a much weaker retriever, whatever its kind.
  - Ablation E (dev-informed weighting): the best weighted fusion is statistically indistinguishable from ColPali (+0.08 pt, p 0.94); the +0.47 pt seen on dev was selection noise.
- **Resources:** fusion adds ~0.13 ms per query (rank + combine) on top of the components; C costs +0.35 s indexing and +0.6 MB over B, D +21 s and +2 MB; with parallel retrieval the latency is set by ColPali (~51 ms). Caveat: components were measured on different devices.
- Post-hoc (without 22 duplicated queries, n = 212): same picture (B 0.5977; C −0.056; C_w +0.001).
- *Interpretation:* on DocVQA, ColPali's page-image embeddings already capture the lexical anchors that BM25 relies on (its OCR-free reading of the page text is better than Tesseract OCR + BM25), and rank fusion can only import BM25's errors. A lexical signal could still help where ColPali reads text poorly (dense small print, other languages) — an open question for other ViDoRe tasks, not tested here.

### 2026-11-06 — Failure analysis (test split, post-hoc, descriptive)
- Type: analysis (registry row 21); `scripts/failure_analysis.py` → `results/failure_analysis/` (`examples.md` has every favourable case and the ten largest unfavourable ones, with OCR snippets).
- **Headroom (measured, hit = gold page in top 5, n = 225):** ColPali 147 hits, BM25 100. BM25 hits while ColPali misses on only **3** queries (0 of 93 `exact`, 3 of 115 `general`, 0 of 17 `visual`); ColPali hits while BM25 misses on **50**. Oracle "either system" = 150 → even a perfect selector could add at most 3 hits (+1.3 pt Recall@5). *Interpretation:* this is why no fusion can win much — BM25's successes are almost a subset of ColPali's.
- **Where C loses (measured):** 49 losses = 26 general, 19 exact, 4 visual; in **42 of 49** BM25 had missed the gold page (rank > 5), so equal-weight RRF pulled BM25's wrong top pages above ColPali's correct one. Typical transition: ColPali rank 1 → C rank 2–5 (20 queries) and rank 2–5 → 6–10 (10).
- **Mechanism — lexical overlap** (share of the query's BM25 tokens found in the gold page's OCR tokens): mean 0.73 when BM25 hits vs 0.18 when it misses. With zero overlap (75 queries) BM25 hits 0 % and ColPali still 23 %; with full overlap (43 queries) BM25 91 %, ColPali 95 %. *Interpretation:* BM25 only works when the query's words survive OCR and the official preprocessing; ColPali reads the page image and matches even when the OCR text does not contain the words.
- **Concrete unfavourable examples (C vs B):**
  - "What is No. 2-26 of subheading H?" (`exact`): ColPali rank 1, BM25 rank 342 — "No" is a stopword and "2-26" is not alphanumeric, so nothing of the anchor reaches BM25; C drops to rank 3.
  - "What is the VO. NO.?" (`exact`): OCR reads "YO. NO.73-170"; ColPali rank 1, BM25 137, C 3.
  - "What is the cost of postage (see the postal stamp at top left)?" (`visual`): the gold page has **empty** OCR text; ColPali rank 3, BM25 252, C 7.
  - "What Tobacco brand of GPI is shown in the picture?" (`visual`): ColPali 1, BM25 31, C 3.
- **Favourable examples (C vs B, all 9):** mostly small corrections from ColPali rank 2 to rank 1 when BM25 is also right ("Who is from Indiana?", "what is the project number?", "When is the remittance due?"); two larger rescues: "What is the name of the laboratory?" (ColPali 14 → C 4) and "How many bullet points are there?" (ColPali 136 → C 4, BM25 2).
- C_w (weighted) changes only 5 queries relative to ColPali (2 better, 3 worse) — with w = 0.95 BM25 can only break near-ties.
- *Interpretation for the report:* the pre-registered hypothesis assumed ColPali is weak on lexical-match queries; on DocVQA the opposite holds — ColPali is strongest there, and the lexical signal is weakest exactly where OCR and tokenisation damage the anchors (codes with hyphens, "No.", OCR misreads, empty OCR). Open question (untested): a better text channel (layout-aware OCR, a tokenizer that keeps codes) might raise BM25's headroom above 3 queries.

### 2026-11-11 — Audit before submission; selection-oracle bound; two report claims corrected
- Type: analysis + fix (registry rows 24–27)
- **Fresh-clone reproduction (measured):** cloning the repository and running the CPU pipeline in a new environment reproduces every quality metric, split, dev selection, test result and failure-analysis number exactly. All six figures are pixel-identical. Only timings and run metadata differ, e.g. BM25 latency 0.12 → 0.13 ms. The check is now a committed script: `scripts/checks/reproduce_cpu_pipeline.sh`.
- **Fix:** `src/runinfo.py` imported torch at module level, so the "CPU-only" scripts needed torch. The import is now lazy, and `requirements-cpu.txt` lists the torch-free environment. Tests pass without torch.
- **Selection oracle (post-hoc, descriptive, test split; added to `scripts/failure_analysis.py`):**
  - Taking the better of BM25 and ColPali per query gives nDCG@5 **0.5878** vs ColPali 0.5699 (+1.8 pt).
  - BM25 has the higher nDCG@5 on 10 queries: 2 exact, 8 general, 0 visual.
  - *Interpretation:* the earlier headroom counts (3 extra top-5 hits) bound only *which* pages a per-query selection could bring into the top 5; nDCG@5 can also rise by reordering inside the top 5. Neither number bounds every possible fusion function. The report previously said the headroom "bounds any fusion"; that was too strong and is corrected.
- **Correction (literature):** the report and matrix said DocVQA's *train* split is in ColPali's training data.
  - Checked against the paper (§4.2) and the `vidore/colpali_train_set` card: DocVQA is among the academic training sources (App. A.3 / Table 3 count read only from an extracted summary), but **the split is not stated**.
  - The paper's contamination check covers multi-page PDFs only.
  - Wording corrected; test-page overlap with the training set was not checked.
- Verified and filled in: Bruch et al. venue (ACM TOIS 42(1), 2024, Crossref); DocVQA licensing (ViDoRe copy tagged MIT on HF, `_tesseract` copy none, original DocVQA pages none; images from the UCSF Industry Documents Library); the authors' hardware statements (L4 for latency/indexing, 8 GPUs of unstated type for training). ColPali and slide-study venues could not be verified (dblp/OpenReview blocked, no Crossref record) and are cited as arXiv.

### 2026-11-11 — Pre-registration of two POST-HOC extensions (written before running either)
- Type: decision (approved by the user 2026-11-11). Both extensions come **after** the single main test run of 2026-11-04. They are labelled post-hoc and exploratory everywhere, and they change nothing in the main result or its verdicts.

**1. Score-level convex combination (C_cc, D_cc)**
- Why: our own literature (Bruch et al., ACM TOIS 2024) finds tuned convex combination better than RRF for lexical + dense text retrieval. The report lists "rank fusion only" as a limitation.
- Method: `alpha·minmax(ColPali) + (1 − alpha)·minmax(text)`, min-max per query over all 500 pages.
- Selection: on dev, alpha ∈ {0, 0.05, …, 1.0}, maximise nDCG@5; ties go to the alpha closest to 0.5.
- Matched conditions: same grid and rule for BM25 (C_cc) and BGE-M3 (D_cc); alpha = 1 (ColPali alone) is allowed.
- Code and config: `src/fusion.py`, `scripts/run_cc_fusion_dev.py`, `configs/fusion_cc_docvqa.yaml`.
- *Hypothesis (before running):*
  - the selected alpha is high (≥ 0.8);
  - on test, |Δ C_cc vs B| < 1 nDCG@5 point and not significant;
  - C_cc ≈ D_cc.
  - The per-query selection oracle (+1.8 pt) suggests little room.
  - **What would change our conclusion:** C_cc significantly > B. That would mean the negative result is specific to rank fusion.

**2. Code-preserving BM25 tokenizer (A_code, C_code)**
- Why: the failure analysis attributes BM25 misses partly to the official preprocessing, which drops "no" and non-alphanumeric codes such as "2-26".
- Method (`src/bm25.py`, mode `codes`):
  - lowercase, then split into alphanumeric runs joined by `- . / &`;
  - keep each joined code whole *and* as its parts;
  - NLTK stopwords minus "no".
- Same BM25 parameters and the same OCR text as System A; no tuning.
- C_code = RRF(A_code, ColPali) with C's frozen k = 1 (no new selection).
- Config: `configs/bm25_codes_docvqa.yaml`.
- *Hypothesis (before running):*
  - A_code ≥ A on test nDCG@5, with the gain mostly on `exact` queries;
  - the number of test queries where BM25 has the gold page in its top 5 and ColPali does not ("A only") rises from 3, but stays small (< 10);
  - C_code is still significantly below B, because equal-weight RRF still imports BM25's misses.
  - **What would change our conclusion:** a large rise in "A only", i.e. if the tokenizer were the main bottleneck.

**Test protocol**
- `configs/posthoc_test_evaluation.yaml` → `scripts/run_test_evaluation.py` (frozen_check `posthoc`: refuses alpha/k that differ from the committed dev selections).
- Test is evaluated once for these systems.
- Holm correction over C_cc, D_cc and C_code vs B.
- Pairs: C_cc vs D_cc, A_code vs A, C_code vs C.
- Top-5 headroom for A and A_code vs B.

### 2026-11-11 — Post-hoc extensions on dev (test not touched)
- Type: run (registry rows 33–34). Grid and rule as pre-registered in `cb332b7`.
- **A_code** (code-preserving tokenizer, measured):
  - full set nDCG@5 **0.4020**, vs A's 0.3907;
  - dev 0.3989, vs A's 0.3995;
  - no query is left without tokens (A: 2).
  - Index: 18,719 terms, vs 14,573 for A.
- **Convex combination on dev** (measured, B = 0.5804):
  - Every alpha < 1 is below ColPali, for both text retrievers.
  - Best alpha < 1: C_cc 0.5753 (alpha 0.95), D_cc 0.5759 (alpha 0.85).
  - The rule selects **alpha = 1.0 (ColPali alone) for both**.
  - The pre-registered hypothesis (alpha ≥ 0.8) holds in its strongest form: dev concludes "no fusion".
- **Frozen for the post-hoc test run:** C_cc alpha = 1.0, D_cc alpha = 1.0 (both therefore identical to B on test by construction); C_code = RRF(A_code, B), k = 1.

### 2026-11-11 — POST-HOC test evaluation (run once; exploratory)
- Type: run (registry row 35). Config `configs/posthoc_test_evaluation.yaml` (frozen from dev in `8c4ad0c`); outputs `results/posthoc_extensions/` (`metrics.json`, `per_query.csv`, `summary.md`, console log).
- **Measured (test, n = 225, nDCG@5):**
  - A_code 0.4052 vs A 0.3818: Δ +0.023 [+0.004, +0.046], p 0.027 (uncorrected pair), W/T/L 14/206/5. The gain is on `exact` queries: 0.617 → 0.667.
  - C_code = RRF(A_code, B): 0.5253, Δ vs B −0.045 [−0.066, −0.024], p_Holm 0.0003. C_code vs C: +0.009, p 0.09.
  - C_cc = D_cc = B (alpha = 1 selected on dev), Δ 0 by construction.
- **Headroom (top-5, test):** A_code finds the gold page where ColPali misses it on **3** queries, the same as A. Among exact queries the count is still 0. A_code's extra hits (103 vs 100) fall on queries ColPali already gets right ("B only" drops from 50 to 47).
- **Verdict on the post-hoc hypotheses:**
  - CC: as expected, no gain. Dev already rejects every alpha < 1.
  - Tokenizer: A_code is better than A, as expected. The expected rise in "A only" did **not** happen; C_code is still significantly below B, as expected.
- *Interpretation:* better lexical matching improves BM25 on its own, but only on queries ColPali already solves. This strengthens the main conclusion: on DocVQA the lexical signal is largely redundant with ColPali, whatever the tokenizer and whatever the fusion function (rank or score). Caveat: post-hoc, single task.

### 2026-11-15 — Pre-registration of POST-HOC set 2: text RAG retrieval stack, cross-encoder reranking, OCR quality (written before running any of it)
- Type: decision (approved by the user 2026-11-15). Requested by the user on 2026-11-15, **after** the main test run (2026-11-04) and post-hoc set 1 (2026-11-11). Everything below is labelled *post-hoc set 2, exploratory* in every output. It changes nothing in the main result or its verdicts.
- Why:
  - A common text-RAG retrieval stack is hybrid lexical + dense retrieval followed by a cross-encoder reranker. The ColPali paper compares ColPali with single text retrievers; this stack is a stronger text baseline.
  - A cross-encoder reads the query and the page text together, so it is a much stronger text signal than BM25 or BGE-M3. Reranking ColPali's candidates with it tests whether our negative result only holds for *weak* text retrievers.
  - The failure analysis points to OCR damage; we have not measured OCR quality directly.
- Text input for every text system: the same `_tesseract` OCR text (`9a183a7`), one page = one passage. No chunking: the median page has 134 words.

**1. Text hybrid T_hyb = RRF(BM25, BGE-M3)**
- Scores of A and BGE-M3 as committed. k from the same grid as C/D ({1, 5, 10, 20, 40, 60, 80, 100, 200, 500}), same rule: max dev nDCG@5, ties → k closest to 60.

**2. Cross-encoder reranker**
- `BAAI/bge-reranker-v2-m3` @ `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (Apache-2.0, not gated), FlagEmbedding 1.4.2 `FlagReranker`, fp16 on Apple MPS, raw scores (`normalize=False`; ordering is the same).
- `max_length`: the smallest value in {512, 1024, 2048} for which at least 95 % of (query + page) pairs fit without truncation, taking the longest query. Decided from token counts only, before any reranker score exists.
- The reranker scores, once, a candidate pool per query: the union of the top 50 pages of T_hyb (selected k) and the top 50 of ColPali. Relevance labels are not used to build the pool.
- **T_rr** = T_hyb's top N reranked by the cross-encoder (pages below N keep T_hyb's order).
- **B_ce** = ColPali's top N reranked by the cross-encoder (pages below N keep ColPali's order).
- N ∈ {1, 5, 10, 20, 50}, selected on dev separately for T_rr and B_ce (max nDCG@5; ties → smaller N). N = 1 means no reranking, so "do not rerank" is a possible outcome.

**3. Cascade A→B (ColPali reranks BM25's candidates)**
- ColPali's order applied to BM25's top N; pages below N keep BM25's order. N ∈ {10, 20, 50, 100}.
- Descriptive cost/quality curve: no selection. Every N is reported on test, together with BM25's Recall@N (the ceiling of the cascade).

**4. OCR quality (descriptive)**
- Proxy per page: share of alphabetic OCR tokens (lowercased, regex `[a-z]+`, length ≥ 2) found in the NLTK `words` English word list. Pages with empty OCR text are reported separately.
- Full query set (451; no tuned systems involved): A, BGE-M3 and B. Queries are grouped by the proxy of their gold page into terciles. Per group: Hit@5 and nDCG@5 per system; Spearman ρ between the proxy and per-query nDCG@5 per system, with a bootstrap 95 % CI (10,000 resamples, seed 0).

**Hypotheses (before running)**
- T_hyb ≈ A: |Δ| < 2 nDCG@5 points, because BGE-M3 is much weaker than BM25 here (Bruch et al.: fusion gains shrink when the inputs differ in quality).
- T_rr > T_hyb on test, but T_rr significantly < B: the full text RAG stack does not reach ColPali on DocVQA.
- B_ce: dev selects a small N (≤ 5) or N = 1; on test |Δ B_ce vs B| < 1 point and not significant.
  - **What would change our conclusion:** B_ce significantly > B. The main negative result would then hold only for weak first-stage text retrievers, not for text signals in general.
- Cascade: nDCG@5 below B for every N; the gap shrinks as N grows and is bounded by BM25's Recall@N.
- OCR: ρ is clearly positive for A and BGE-M3, and much smaller for ColPali; text systems' Hit@5 rises across terciles more than ColPali's.
  - **What would contradict it:** ColPali equally sensitive to the proxy. The proxy would then measure page difficulty in general, not damage to the text channel.

**Test protocol**
- Frozen config `configs/posthoc2_test_evaluation.yaml` → `scripts/run_test_evaluation.py`; it refuses settings that differ from the committed dev selections. Test is evaluated once for these systems.
- Holm correction over T_hyb, T_rr and B_ce vs B. Extra pairs: T_rr vs T_hyb, T_hyb vs A, T_rr vs A. Top-5 headroom T_rr vs B.
- Resources: reranker latency per query at the selected N (median over 50 queries, MPS; the device differs from ColPali's L4) and the model's memory.

### 2026-11-15 — Post-hoc set 2: dev selection, reranker run, single test run, OCR analysis
- Type: runs (registry rows 39–44). Design and expectations pre-registered in `92c2163` (approved by the user).
- **T_hyb (dev):** k = 5, dev nDCG@5 0.4062 (BM25 alone 0.3995).
- **Cross-encoder** (`bge-reranker-v2-m3`, MPS fp16): max_length 1024 by the rule (97.4 % of pairs fit; 13 pages truncated). Pool 37,895 pairs (84 pages/query). 1,402 s in total. Rerank latency median: top 5 0.70 s, top 50 13.6 s.
- **Dev selection of N:**
  - T_rr: 0.4062 (N = 1) → 0.4747 (N = 50), monotone; selected N = 50.
  - B_ce: 0.5804 (N = 1) → 0.5508 (5) → 0.5234 (10) → 0.5025 (20) → 0.4725 (50); selected **N = 1 (no reranking)**. The cross-encoder over OCR text reorders ColPali's top pages for the worse, even inside the top 5.
- **Test (once, `68cf359`, n = 225, nDCG@5):**
  - T_hyb 0.385 vs A 0.382: Δ +0.003, p 0.79.
  - T_rr 0.438: vs T_hyb +0.053 [+0.014, +0.095], p 0.012; vs B −0.131 [−0.180, −0.085], p_Holm 0.0003.
  - B_ce = B by construction.
  - Cascade (ColPali reorders BM25's top N): 0.454 / 0.476 / 0.501 / 0.514 for N = 10 / 20 / 50 / 100; BM25 Recall@N 0.498 / 0.551 / 0.636 / 0.689.
  - Headroom: T_rr finds the gold page that ColPali misses on 6 queries (BM25: 3); ColPali finds it where T_rr misses on 43.
- **OCR proxy (full set, descriptive):** median page 0.731 (10th–90th 0.525–0.843). Spearman ρ with nDCG@5: BM25 +0.027 [−0.065, +0.120], BGE-M3 −0.019 [−0.108, +0.073], ColPali +0.008 [−0.087, +0.104].
- **Verdict on the pre-registered hypotheses:**
  - T_hyb ≈ A: held.
  - T_rr > T_hyb and T_rr significantly < B: held.
  - B_ce: dev selects N = 1: held (strongest form). The result that would have changed our conclusion (B_ce > B) did not occur.
  - Cascade below B for every N, gap shrinking with N: held.
  - OCR: **not supported**. No system's success correlates with the proxy. *Interpretation:* the proxy is crude (it penalises clean pages full of names, codes and numbers) and does not measure whether the *query's* words survive OCR, which the earlier overlap analysis does.
- *Interpretation:* a much stronger text signal (cross-encoder) improves the text pipeline by ~5 points but does not approach ColPali, and hurts when applied to ColPali's candidates. The main negative result is not an artefact of weak first-stage text retrievers.

### 2026-11-15 — Diagnostic (not pre-registered): empty-OCR pages dominate BGE-M3's top ranks
- Type: analysis (registry row 45), dev only; test not touched. Found while reading the OCR table: BGE-M3 hits all 4 queries whose gold page has empty OCR text.
- **Measured:** the 4 empty pages share the empty-string embedding; at least one is in BGE-M3's top 5 for 388 of 451 queries (each page for 363–368).
- On dev, moving them to the bottom of BGE-M3's ranking: BGE-M3 0.2351 → 0.3131; D (k = 1) 0.5263 → 0.5194; T_hyb (k = 5) 0.4062 → 0.4060.
- *Interpretation:* BGE-M3's stand-alone score is understated, and its closeness to the paper's 25.7 is partly coincidental. D and T_hyb, on which our conclusions rest, are essentially unchanged. Not corrected on test; reported as a limitation. `scripts/checks/bge_empty_pages.py`, `results/checks/bge_empty_pages.json`.

### 2026-11-16 — Plan (written before computing any count): taxonomy of ColPali's misses
- Type: descriptive analysis, user request ("why can't fusion beat ColPali?"). **No new system, no selection, no new test-set evaluation of any system:** it reads ranks of B, A, BGE-M3 and T_rr that are already frozen and evaluated. Run on dev and test separately.
- A **miss** = gold page not in ColPali's top 5 (gold rank > 5). The same flags are computed for ColPali's hits, so each flag's rate in misses can be compared with its rate in hits.
- Flags (not exclusive), fixed here before looking at counts:
  - F1 `duplicate_text`: the query text occurs on more than one dataset row (the official protocol keeps one arbitrary gold page). F1b `alt_page_in_top5`: a page of another row with the same query text is in ColPali's top 5.
  - F2 `gold_ocr_short`: the gold page's OCR text has fewer than 20 whitespace tokens (includes the 4 empty pages).
  - F3 `generic_query`: every BM25 query token (official tokenizer) occurs in the OCR of at least 25 pages (5 % of the corpus), or the query has no BM25 token. Sensitivity: 10 and 50 pages.
  - F4 `no_overlap`: no BM25 query token occurs in the gold page's OCR.
  - F5 `near_miss`: gold rank 6–10.
  - F6 `text_finds_it`: BM25, BGE-M3 or T_rr has the gold page in its top 5.
- Primary category for the count table, in this priority order: duplicate_text → gold_ocr_short → generic_query → no_overlap → `other` (specific query, gold page has text and shares a word with it, ColPali still misses).
- *Expectation (hypothesis):* duplicate and generic queries are over-represented among misses compared with hits; `text_finds_it` is rare in every category.

### 2026-11-16 — Results: taxonomy of ColPali's misses (descriptive)
- Type: analysis (registry row 48), `scripts/miss_taxonomy.py` → `results/miss_taxonomy/`. Flags fixed in `fe3533b` before counting.
- **Measured (misses vs hits, test n = 225: 78 misses / 147 hits; dev n = 226: 80 / 146):**
  - `no_overlap` (no query word in the gold page's OCR): test 58 vs 19, dev 52 vs 26. The clearest difference between misses and hits on both splits.
  - `duplicate_text`: test 10 vs 3, dev 9 vs 0. An alternative page with the same query text is in ColPali's top 5 for only 2 test misses.
  - `generic_query` (≥ 25-page df): test 12 vs 13, dev 10 vs 3. Not a consistent difference; sensitive to the threshold (test 33/40 at 10 pages, 5/3 at 50).
  - `gold_ocr_short`: test 7 vs 5, dev 5 vs 8. Not a driver.
  - `near_miss` (rank 6–10): test 6, dev 12. Most misses are far: test 42 of 78 have the gold page below rank 50.
  - `text_finds_it` (BM25, BGE-M3 or T_rr top 5): test 11 of 78 misses, dev 8 of 80. Most of them fall in `other`, 9 of 17 on test.
  - Primary category, test / dev: duplicate 10 / 9, short OCR 4 / 4, generic 8 / 9, no overlap 39 / 34, other 17 / 24.
- *Interpretation:* the typical miss is a question about **position or layout without a content word**, e.g. "What is the name of the company on the letter head?" or "What is the date of the letter?". The gold page's OCR is often readable but shares no word with the query, and many other pages (letters, memos) fit the question equally well. A text signal cannot help there, since there is nothing to match. These DocVQA questions were written by annotators looking at the page; as retrieval queries they are under-specified. The residual errors therefore look mostly like a property of the queries, not of the OCR. This predicts that cleaner OCR will help little, and that only a model which reads the page image against the question (a visual reranker) can address part of them.
- *Hypothesis expectation from the plan:* duplicates are over-represented among misses (held, but small). Generic queries over-represented (not consistent). Text rarely finds a miss (held).

### 2026-11-16 — Pre-registration: post-hoc set 3 (improvement attempts; written before any of these models was run)
- Type: plan. User request after the miss taxonomy: "try the improvement hypotheses anyway; Colab is fine for GPU; use the newest Qwen models if available". Exploratory, labeled post-hoc. **This is the fourth use of the test split.** It changes nothing in the main result. Every setting is chosen on dev; the test split is evaluated once, after the settings are frozen in a committed config.
- Models (Hugging Face revisions pinned now; all Apache-2.0, not gated):
  - `Qwen/Qwen3-VL-Reranker-2B` @ `4bd860ac4f15ad1897a214615cccc700f8f71818`: multimodal reranker, page **image** + question.
  - `Qwen/Qwen3-VL-Embedding-2B` @ `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`: single-vector visual retriever.
  - `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`: VLM transcription of the 500 pages. Non-thinking mode, greedy decoding, max 2,048 new tokens. Prompt: "Transcribe all text on this page in natural reading order as plain text. Output only the text." If Qwen3.5 cannot run on Colab with vLLM or transformers within one attempt, the pre-registered fallback is `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`, with the same prompt and decoding.
  - Caveat *(unverified)*: whether these models' training data includes DocVQA pages is not stated in the cards we read. ColPali's training set includes DocVQA's train split.
- Systems:
  - **H1 `B_vr`, visual reranker:** ColPali's top N pages reordered by Qwen3-VL-Reranker-2B (`src/rerank.py`). N ∈ {1, 5, 10, 20}, where N = 1 means no reranking. Selected by max dev nDCG@5, ties to the smaller N. Fixed instruction: "Given a question, retrieve the document page image that answers it." Scores are computed for ColPali's top 20 of every query. Reachable misses: gold rank 6–20 for 20 of 78 test misses and 24 of 80 dev misses.
  - **H2 `C_sel`, selective fusion:** use C = RRF(BM25, ColPali), k = 1 (the frozen main C), only when BM25 is confident; otherwise use B. Confidence g(q) = (s1 − s2) / s1 for BM25's two highest scores (0 if s1 = 0). Threshold τ ∈ dev quantiles of g at {0.5, 0.75, 0.9, 0.95, 0.99}, plus +∞ (never fuse, = B). Selected on dev, ties to the larger τ.
  - **H3 `C_vlm`, better OCR:** `A_vlm` = BM25 (official tokenizer) over the Qwen3.5 transcriptions; `C_vlm` = RRF(A_vlm, ColPali). k from the main grid {1, 5, 10, 20, 40, 60, 80, 100, 200, 500}, chosen on dev, ties closest to 60.
  - **H4 `E_ens`, stronger/second visual retriever (diagnostic):** `Q3E` = Qwen3-VL-Embedding-2B (page images; query instruction "Retrieve the document page image that answers the question."; cosine). `E_ens` = RRF(ColPali, Q3E), same k grid and rule.
- Test analysis: Holm over B_vr, C_sel, C_vlm, E_ens vs B (paired permutation, bootstrap CIs, as before). Descriptive: A_vlm vs A; Q3E alone; overlap of Q3E's and ColPali's misses; the taxonomy's `no_overlap` flag recomputed with the VLM text.
- Resources: reranker latency per query at the selected N; OCR seconds per page; Q3E indexing time; peak GPU memory (Colab L4).
- **Expectations (hypotheses, written before running):**
  - H1: dev selects N > 1 and B_vr > B on test. This is the only attempt we expect to improve, and the gain is bounded by the reachable misses plus reordering inside the top 5.
  - H2: C_sel ≈ B (n.s.); its upper bound is the +1.8-point selection oracle.
  - H3: A_vlm > A, but C_vlm ≤ B. The `no_overlap` share among ColPali's misses drops only a little, because the taxonomy points to the queries, not the OCR.
  - H4: Q3E misses more than half of ColPali's misses too, and E_ens is within ±2 points of B.
  - What would change our reading: C_vlm > B would mean OCR quality was the bottleneck for fusion. B_vr ≤ B would mean even reading the image cannot resolve the under-specified queries.

### 2026-11-16 — Post-hoc set 3: GPU runs and dev selection (test not touched)
- Type: runs (registry rows 50–59). Colab L4. Two failed attempts were fixed in code, not in settings: the transformers-5 `mm_token_type_ids` padding in the reranker helper, and a missing `pytrec_eval` import in the vLLM runtime.
- **H4 (Q3E):** dev nDCG@5 0.5028 (B 0.5804). E_ens = RRF(Q3E, B): k = 1, dev 0.5733.
- **H3 (Qwen3.5-4B transcription):** 500 pages, 0 empty, 11 pages hit the 2,048-token cap; 1,187 s on L4. A_vlm dev 0.4459 vs A 0.3995 (+4.6 pt). C_vlm: k = 1, dev 0.5578 (< B). Descriptive check: with the cleaner text, no query word on the gold page still holds for 53 of 78 test misses (58 with Tesseract) and 50 of 80 dev misses (52). For hits it falls from 19 to 11 and from 26 to 18.
- **H1 (Qwen3-VL-Reranker-2B on ColPali's top N):** dev 0.5804 / 0.6052 / 0.6231 / 0.6299 for N = 1 / 5 / 10 / 20 → N = 20. Cost: about 7.1 s per query at N = 20 on L4 (one pair per forward pass, as in the model's helper).
- **H2 (selective fusion):** τ = 0.590 (0.9 quantile), fuses on 23 dev queries; dev 0.5846.
- *Interpretation (dev only):* only the visual reranker moves dev clearly. This is the one attempt that reads the page **image** against the question. The text-side attempts (cleaner OCR, gating) stay at or below ColPali, as pre-registered.
- Frozen in `configs/posthoc3_test_evaluation.yaml`; the dry run on dev reproduces every selection.

### 2026-11-16 — Post-hoc set 3: single test run (fourth use of the test split)
- Type: run (registry rows 60–61), `e24575c`, n = 225. nDCG@5:

  | System | nDCG@5 | Δ vs B | 95% CI | p_Holm |
  |---|---|---|---|---|
  | B_vr (H1) | 0.581 | +0.011 | [−0.018, +0.039] | 0.92 |
  | C_sel (H2) | 0.568 | −0.002 | [−0.007, 0.000] | 1.0 |
  | C_vlm (H3) | 0.535 | −0.035 | [−0.057, −0.012] | 0.011 |
  | E_ens (H4) | 0.558 | −0.012 | [−0.031, +0.008] | 0.71 |

  B is 0.570.
- Descriptive pairs:
  - A_vlm 0.428 vs A 0.382: +0.046 [+0.019, +0.076], p 0.002. The cleaner text helps BM25 itself.
  - Q3E 0.471 vs B: −0.099, p 0.0001.
- Top-5 headroom: A_vlm finds the gold page where ColPali misses on 3 queries, the same as Tesseract BM25. Q3E does so on 6.
- **H1 dev vs test:**
  - Dev gain: +4.9 pt (dev CI of the paired Δ [+1.8, +8.4]).
  - Test gain: +1.1 pt, not significant.
  - Flow analysis (`results/posthoc3/rerank_flow.json`):
    - Dev: 11 ColPali misses fixed, 3 hits broken, rank-1 hits 115 → 128.
    - Test: 10 misses fixed, 8 hits broken, rank-1 hits 106 → 110.
  - *Interpretation:* the reranker does reach misses on both splits (10–11 of 20–24 reachable). On test it also pushes more correct pages out of the top 5 and promotes fewer to rank 1. The dev gain was partly split-specific. N was the largest grid value and the dev curve was still rising, so selection on dev is a small source of optimism. The rest is compatible with chance: the dev CI [+1.8, +8.4] and the test CI [−1.8, +3.9] overlap, while the test CI includes zero and excludes the dev point estimate.
- **Verdict on the pre-registered expectations:**
  - H1 (B_vr > B on test): **not supported**. The direction is positive but not significant, and the dev gain did not replicate.
  - H2 (C_sel ≈ B): held.
  - H3 (A_vlm > A, but C_vlm ≤ B): held. C_vlm is significantly below B. The `no_overlap` share among misses dropped only a little (58 → 53).
  - H4 (Q3E misses most of ColPali's misses; E_ens within ±2 pt): held. E_ens is −1.2 pt, and Q3E finds only 6 of ColPali's misses.
- *Interpretation:* better text and a second visual retriever do not help, as predicted. Reading the image against the question (H1) is the only direction with a positive sign. It costs about 7 s per query, and on this test split its gain is not distinguishable from zero. The main conclusion is unchanged: on this task we found no change that reliably beats ColPali alone.
