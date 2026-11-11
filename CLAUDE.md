# CLAUDE.md — YZM5110 Research Project (ColPali + Hybrid Retrieval)

## Communication

- Talk to the user in **Turkish**. Write code, comments, README, configs and the report in **English**.
- The user is a master's student (AI and Data Engineering). Explain *why*, not only *what*; they must be able to defend every step in a presentation.
- Keep answers **short and direct**; add depth only when asked.
- Keep four things strictly separate in every answer and document (the assignment requires this distinction):
  1. **Published** — what the authors report.
  2. **Interpretation** — our own reading or explanation of the literature or of a result.
  3. **Measured** — what we obtained ourselves (with config, commit, date).
  4. **Hypothesis** — what we expect but have not tested.

## Integrity rules for Claude

- **Never invent** citations, paper details, published scores, dataset statistics or API details. Mark anything not yet checked as *(unverified)*.
- Every number in the report or in an answer must come from a file in `results/` (measured) or from a cited source checked by the user (published).
- The user is responsible for reading sources and checking citations (assignment AI policy); point them to the exact place to verify.

## Assignment (YZM5110 Research Project)

- Individual project, **30%** of the course grade.
- **Deadline: 18 November 2026**, before the presentation session. Project started 23 October 2026.
- Deliverables:
  - Report: `<LastName>_YZM5110_Research.pdf` (last name: TBD), repository URL on the **first page**.
  - Report structure: (1) research question, scope, motivation, contributions; (2) literature search approach and critical synthesis; (3) technical explanation of the method; (4) reproduction target, implementation, deviations; (5) proposed improvement and rationale; (6) experimental design, datasets, splits, baselines, metrics, resources; (7) results, controls, uncertainty, failure analysis; (8) limitations, conclusions, open questions.
  - Public repository with: setup instructions, dependency versions, licensed data-access instructions, exact commands, configurations, saved outputs, scripts that produce every reported table/figure. **Pin the submitted version with a commit or tag** (e.g. `v1.0-submission`).
  - A defensible presentation (graded under "reproducibility and communication").
- Literature: analyze **at least 5 papers** in a comparison matrix with these fields: question and contribution; modalities and assumptions; mechanism and objective; data and protocol; baselines and evidence; resources and reproducibility; limitations and connections. Explain how papers were found and selected. The synthesis must end in a **justified research gap**.
- Reproduction: choose a testable claim, state what we expect and what would make us question the claim. Record source revision, dependencies, checkpoint, data version, preprocessing, seeds, config, commands. Smaller scale is allowed but every deviation must be described, together with which comparisons still hold. Keep published and measured scores separate; investigate discrepancies before blaming the method.
- Improvement: one targeted change, rationale stated **before** seeing results, evaluated against the reproduced baseline under matched conditions, with a **control or ablation**, task-quality metrics and a **resource metric** (latency, peak memory, ...). Negative results are acceptable if analyzed honestly. Report unfavorable examples as well as favorable ones. **Never tune on the test set.**
- Grading criteria: literature depth and synthesis; technical understanding; reproduction; targeted improvement; experimental rigor (fair controls, valid splits, honest uncertainty, failure analysis); reproducibility and communication. Reasoning and honesty count, not only whether the score improved.
- AI use is allowed but must be disclosed in the report (tools + specific contributions). Keep `docs/ai_usage.md` updated as we go.
- Integrity: no sharing of project-specific code, outputs, analysis, reports or slides with classmates.

## Decisions made so far

- **Research topic (for the instructor's topic table; submission status unconfirmed):** Multimodal Document Retrieval for RAG using Vision-Language Models
- **Main paper:** ColPali: Efficient Document Retrieval with Vision Language Models — https://arxiv.org/abs/2407.01449
  - Code: https://github.com/illuin-tech/colpali (package `colpali-engine`)
  - Benchmark: ViDoRe — https://huggingface.co/vidore
- **Why this counts as multimodal (≥ 2 modalities):** documents are page *images* (visual layout, figures, tables, rendered text) and queries are *text*; ColPali aligns the two in one embedding space. BM25 adds a third view: extracted page text.
- **No training / fine-tuning.** We use the authors' published checkpoint for inference only.
- **Scope is deliberately narrow:** one ViDoRe task, done well. No generation (no full RAG/LLM answering), no multi-task sweep.
- **Repository:** https://github.com/ismailucrn/colpali-hybrid-retrieval (public, MIT).

### What ColPali is (for explanations)

ColPali encodes each document page *image* with a PaliGemma-based vision-language model into many patch-level vectors, encodes the text query into token-level vectors, and scores query–page pairs with ColBERT-style late interaction (MaxSim: for each query token, take the max similarity over page vectors, then sum). It replaces the OCR → chunk → embed pipeline.

## Research plan

**Research question (hypothesis, not a result):** Does fusing lexical retrieval (BM25 over page text) with ColPali improve page retrieval over ColPali alone, and for which query types?

**Motivating hypothesis:** ColPali may be weaker on queries needing exact lexical matches (numbers, names, codes, rare terms) and on text-heavy pages; BM25 should help most there and least on chart/figure-centric queries.

**Research gap (to be backed by the literature matrix):** hybrid lexical + dense fusion is a known, cheap gain in text retrieval, while visual retrievers like ColPali are usually evaluated alone. Whether a lexical signal still adds anything once retrieval works on page images — and for which queries — is the open question we test.

### Systems

| ID | System | Role |
|---|---|---|
| A | BM25 over page text (OCR or dataset-provided text) | Text-only baseline |
| B | ColPali (published checkpoint) | Reproduced baseline |
| C | RRF(BM25, ColPali) | Proposed improvement |
| D | RRF(dense text retriever, ColPali) | **Control:** is the gain from the lexical signal, or just from ensembling any second retriever? |
| E | RRF `k` sensitivity (weighted score fusion only if time allows) | Ablation |

**Matched conditions for C vs D:** the dense text retriever reads **the same page text** as BM25, retrieves over the same corpus, uses the same RRF formula, and gets the same tuning budget on dev. The dense model is chosen and recorded **before** any results are seen.

### Pre-registered expectations (write before running; do not edit afterwards — add a dated note instead)

- **Reproduction claim:** ColPali's published nDCG@5 on the chosen task is reproducible with the published checkpoint.
  - Tolerance: fixed before the run (proposed: within ±2 nDCG@5 points).
  - A larger gap is investigated (checkpoint revision, precision, image resolution, scoring code, dataset revision) before any conclusion about the method.
- **Improvement predictions:**
  - P1: C > B on test nDCG@5 (paired test).
  - P2: the gain is concentrated in exact-match/numeric queries; little or none on visual/chart queries.
  - P3: C > D. If D ≈ C, the gain comes from ensembling, not from the lexical signal.
- **What weakens the hypothesis:** no gain for C; a gain spread evenly over query types or larger on visual queries; D ≥ C.

### Evaluation protocol

- Primary metric: **nDCG@5** (the paper's main metric — needed for published-vs-measured comparison). Also report Recall@5, Recall@10, nDCG@10, MRR.
- Resource metrics: end-to-end query latency per system (C and D include both retrievers + fusion), indexing time, index size on disk / in memory, peak GPU memory.
- **Two evaluation sets, with different purposes:**
  - **Reproduction (B vs published):** the **full** query set of the task, as in the paper. No tuning happens here.
  - **Fusion comparisons (A–E):** queries split into **dev** (choose `k`, weights) and **test** (touched once, at the end). Split is stratified by query type, seed fixed and recorded, split file committed.
  - The test split is small → confidence intervals will be wide; say so rather than over-claim.
- **Query-type tagging (frozen 3 Nov, before any fusion run):** automatic rule-based tags (visual > exact > general) from the query text only — `src/query_types.py`, `docs/query_tagging.md`, `results/splits/docvqa_query_tags.csv`. Never re-tag after seeing results.
- Uncertainty: per-query bootstrap confidence intervals and a paired test between systems.
- Failure analysis: per query type, where each system wins/loses, with concrete examples — favorable *and* unfavorable.
- Any `k` sensitivity curve shown on test is post-hoc and labeled as such; selection uses dev only.
- Report differences from the original paper (checkpoint version, OCR pipeline, hardware, precision, subset) in `docs/deviations.md`.

## Compute

- **Google Colab Pro, L4 GPU (22.5 GB, native bf16)** for ColPali; laptop for BM25, analysis and writing. Kaggle as backup. On a T4 the script falls back to fp16 (would be a deviation) and may run out of memory (smoke-test peak was 16.2 GB).
- Claude can drive Colab through the app's built-in browser (user signs in to Google; Claude never enters passwords). The Colab notebook prints outputs instead of downloading them, so they can be read from the page and committed.
- User's laptop: Apple M5, 16 GB unified memory, macOS. ColPali via MPS is plausible but **untested** — do not assume it works.
- Compute ColPali embeddings **once** and save them to disk (e.g. Google Drive) so sessions/disconnects don't force recomputation.
- The PaliGemma-based checkpoint may require accepting a license on Hugging Face and an HF token. **Never commit tokens**; use environment variables / Colab secrets.

## Things to verify (do not state these as facts until checked)

- **Checkpoint: `vidore/colpali-v1.3` (decided 3 November 2026).** Reproduction target = its model-card DocVQA nDCG@5 0.5845; the paper's v1.0 value 54.4 is reported as context only. Reason: v1.0 is marked *not supported* in colpali-engine 0.3.18 and making old versions work has an unknown time cost. Optional extra, only if time remains: v1.0 with an older colpali-engine.
- DocVQA dataset license (HF card shows none; DocVQA's original terms to check).
- Whether `vidore/docvqa_test_subsampled_tesseract` text is the same OCR the paper used for its "Unstructured + OCR" BM25 baseline (paper used Unstructured with Tesseract; likely not identical).
- Whether the topic was submitted to the instructor.

### Chosen task: ViDoRe v1 DocVQA (`vidore/docvqa_test_subsampled`), decided 3 November 2026

- **Published nDCG@5 on DocVQA** (verified 3 Nov):
  - Paper, arXiv 2407.01449v6, Table 2: ColPali **54.4**; BM25 on Unstructured text-only 34.1; BM25 on Unstructured + OCR **36.8**; BM25 on Unstructured + captioning 38.4; BGE-M3 on Unstructured + OCR 25.7.
  - Model cards (`results.json` on each HF repo, self-reported by the authors): v1.0 `vidore/colpali` 0.5443, v1.1 0.5499, v1.2 0.5717, **v1.3 0.5845**.
- **Data (measured 3 Nov):** 500 rows = 500 unique pages; 451 unique queries; 22 query texts occur more than once, each time on a *different* page (e.g. "What is the page number?" ×12, "What is the table number?" ×5). `_tesseract` variant: same row order, OCR text in `text_description`, 4 pages with empty text, median 134 words/page, noisy OCR.
- **Official protocol** (`vidore-benchmark`, `ViDoReEvaluatorQA`): queries are deduplicated; each unique query has exactly **one** relevant page — for duplicated query texts, the page of its **last** occurrence (dict overwrite). Metrics via MTEB's `RetrievalEvaluator` (pytrec_eval). Paper's BM25 baseline: `rank_bm25` BM25Okapi with NLTK tokenisation and stopwords.
  - *Interpretation:* the 22 duplicated generic queries are effectively ambiguous (one arbitrary gold page among several valid ones). We keep the official protocol for the published-vs-measured comparison and, as a labeled post-hoc sensitivity check, also report results without them.
- **Control D — confirmed by the user 2026-11-04, before any BGE-M3 or fusion result:** `BAAI/bge-m3` dense vectors (pinned sha `5617a9f`), implemented as in vidore-benchmark's `BGEM3Retriever` (FlagEmbedding `BGEM3FlagModel`, fp16, query max_length 512, dot product), over the **same** `_tesseract` OCR text as BM25. It is the paper's own dense text baseline (DocVQA 25.7 on Unstructured + OCR), so D is a recognised, not hand-picked, model.

### Verified (3 November 2026, from the official README, PyPI and the HF Hub API)

- `colpali-engine` latest is **0.3.18** (Python >=3.10,<3.15, transformers >=5.3). The package is **deprecated** in favour of Sentence Transformers v6 `MultiVectorEncoder`, but kept "for research, reproducibility"; we use it because it is the authors' reference implementation. API: `ColPali.from_pretrained`, `ColPaliProcessor.process_images / process_queries / score_multi_vector`.
- `vidore/colpali-v1.3` (LoRA adapter) and its base `vidore/colpaligemma-3b-pt-448-base` are **not gated** → no HF license acceptance or token strictly needed.
- ViDoRe v1 tasks ship OCR text: `vidore/<task>_tesseract` datasets have a `text_description` column (also `_captioning`, `_ocr_chunk` variants) → page text for BM25 does not need our own OCR.
- `vidore/docvqa_test_subsampled`: 500 (query, page) rows; `_beir` variant: 500 corpus pages, 451 queries, qrels.
- GPU environment: Google Colab (free T4) chosen first; user's Mac (Apple M5, 16 GB) is a possible MPS alternative, untested.

## Current status (as of 11 November 2026)

- **All experiments done and committed** (3–6 Nov): B reproduced (nDCG@5 0.5752 vs model card 0.5845, −0.94 pt; the exact gap is −0.935, an earlier note said −0.93), A 0.3907, BGE-M3 0.2486 (full set); single test run: fusion does not beat ColPali (C −5.4 pt, p_Holm 0.0006; C ≈ D; C_w +0.08 pt, n.s.); failure analysis and figures.
- **Literature matrix** (6 Nov) and **report draft** (7 Nov, `report/main.pdf`) written.
- **Audit 11 Nov:** a fresh clone reproduces every quality metric and figure exactly (`scripts/checks/reproduce_cpu_pipeline.sh`). Fixed: README rewritten; torch no longer needed by the CPU scripts (`requirements-cpu.txt`); two over-strong report claims corrected (DocVQA split in ColPali's training data is not stated; the headroom bound applies to per-query selection, not to every fusion — selection-oracle nDCG@5 added to the failure analysis); verifiable report TODOs resolved (Bruch et al. venue, DocVQA licensing, paper hardware).
- **Open:** report TODOs that need the user (name/ID, ±2 tolerance confirmation, literature cell checks, App. A.3 count, AI-use self-verification, tag hash); presentation; `v1.0-submission` tag after the final edits.

## Timeline (compressed, from 3 November)

- **3–6 Nov:** repo skeleton + docs; GPU environment; smoke test; choose the ViDoRe task; full-set ColPali reproduction with saved embeddings; write pre-registered tolerance.
- **7–10 Nov:** page text + BM25 (A); dense text retriever (D); RRF (C); query tagging + dev/test split committed; evaluation script; choose `k` on dev.
- **11–13 Nov:** single test-set run; resource measurements; bootstrap CIs and paired tests; failure analysis by query type.
- **14–18 Nov:** report, figures, literature matrix completion, repo cleanup, tag `v1.0-submission`, slides.
- Literature matrix runs in parallel: roughly one paper per day.

## Literature (matrix filled 6 Nov — `docs/literature_matrix.md`; the user must still read and confirm every cell)

- **Matrix (8):** ColPali (2407.01449); ColBERT (2004.12832); DSE (2406.11251); ViDoRe V2 (2505.17166); RRF, Cormack et al. SIGIR 2009; Bruch et al., fusion functions (2210.11934); IRPAPERS (2602.17687); slide-retrieval study (2509.15211).
- **Related work only:** BEIR (2104.08663; does *not* evaluate fusion); VisRAG (2410.10594); Lost in OCR Translation? (2505.05666). Dropped: RAG (2005.11401).
- **Note 2026-11-06 on the research gap:** the original wording ("visual retrievers are usually evaluated alone") is too broad. The targeted search found prior text + visual fusion (IRPAPERS helps with clean transcriptions; DSE App. A.2 "minimal improvement"; slide study no gain), none with significance tests, a second-retriever control or query-type analysis, and none on a ViDoRe task with the benchmark OCR. The refined gap is in `docs/literature_matrix.md` §5. The pre-registered predictions above are unchanged.
- Open verification items are listed at the end of the matrix file.

## Recording every run

- **Every execution gets a row in `docs/run_registry.md`** — GPU runs on Colab, local checks, tests, dry runs — **including failed runs**, with: UTC time, where (colab-L4 / mac), script or notebook, commit, status (ok / FAILED + cause / partial), output files and log path.
- **Save the console output** of every Colab run next to its results (`results/<run>/console_<YYYYMMDD_HHMM>[_FAILED].log`, copied from the notebook page) and of every local check (`results/checks/<timestamp>_<name>.log`). A result without a log does not count as reported.
- Scripts used for checks live in the repo (`scripts/checks/`), never only in a scratch directory.
- `docs/experiment_log.md` keeps the *reasoning* (decisions, interpretations); `docs/run_registry.md` keeps the *evidence* (what ran, when, with what outcome).

## Git workflow

- Remote: https://github.com/ismailucrn/colpali-hybrid-retrieval, branch `main`.
- **Commit and push at every important milestone**, without waiting to be asked: e.g. repo skeleton, working smoke test, reproduction result, each new system (A–E), split/tag files frozen, final test run, figures/tables, report drafts, submission tag. Small, focused commits; imperative English messages.
- **All commits are authored by the user's GitHub account only** (`ismailucrn`, email `89876113+ismailucrn@users.noreply.github.com`, set in the repo-local git config). Check `git config user.name` / `user.email` before committing.
  - **No `Co-Authored-By: Claude` trailer or any other AI attribution line in commit messages.** AI assistance is disclosed in `docs/ai_usage.md` and the report instead.
- Before each commit: review `git status` / the diff; never commit credentials, `.env`, data, embeddings or checkpoints.
- Never force-push or rewrite pushed history; that would break the traceability of reported results.

## Repository conventions

```
configs/      experiment configs (YAML); every run reads one
scripts/      entry points (smoke test, indexing, retrieval, evaluation, tables/figures)
src/          reusable code
notebooks/    Colab notebooks (thin wrappers around scripts/)
results/      small saved outputs: run files, metrics JSON/CSV, tables, figures
data/         local data cache — gitignored
docs/         experiment_log.md, run_registry.md, deviations.md, literature_matrix.md, ai_usage.md, query_tagging.md
```

- Every reported number must be reproducible by a committed script + config.
- Every run writes: config used, git commit, package versions, checkpoint name/revision, dataset name/revision, seed, hardware, timestamp.
- Do not commit: credentials, `.env`, large data, embeddings, model checkpoints. Commit small result files (metrics, run files) only.
- Pin dependencies (`requirements.txt` from `pip freeze` after the first working run).
- Provide a small fast test (a few pages/queries) in addition to the full experiment, and state which results need a GPU.

## Next immediate steps

1. ~~Repo skeleton, README, LICENSE, docs templates~~ — done 3 November.
2. ~~GPU environment~~ — Colab first (3 November).
3. ~~Smoke test~~ — passed on Colab L4 (3 November, 7/8 top-1, peak 16.2 GB); `requirements.txt` pinned. Open item: investigate the high peak memory before full indexing (see `docs/experiment_log.md`).
4. ~~Choose the ViDoRe task~~ — DocVQA (3 November).
5. ~~Checkpoint~~ — v1.3. `scripts/run_colpali.py` + `configs/colpali_v1.3_docvqa.yaml` + `notebooks/colpali_docvqa_colab.ipynb` written (3 November), dry-run with a fake model on CPU. Ran on Colab L4 (3 November): nDCG@5 0.5752; outputs committed.
6. ~~Dev/test split + query tagging~~ — frozen 3 November (`results/splits/`, `docs/query_tagging.md`): dev 226 / test 225; visual 35, exact 186, general 230.
7. ~~BM25 (A)~~ — 4 November: full-set nDCG@5 0.3907 (paper 36.8 with different OCR); `results/bm25_docvqa/`. Runs on the Mac CPU in seconds.
8. ~~BGE-M3 (D's text component)~~ — 4 November: full-set nDCG@5 0.2486 (paper 25.7); `results/bge_m3_docvqa/`. Ran on the Mac (MPS) in ~1 minute.
9. ~~Dev selection~~ — 4 November. Equal-weight RRF hurts on dev (B 0.5804; C 0.5335, D 0.5263 at k=1). Dev-informed ablation E (weighted RRF, pre-registered in `b558ceb`): C_w w=0.95/k=20 dev 0.5851; D_w w=0.95/k=10 dev 0.5804 (= B).
10. ~~Single test run~~ — 4 November (`results/test_evaluation/`). **Negative result:** B 0.5699; C 0.5162 (−5.4 pt, p_Holm 0.0006); D 0.5187; C_w +0.08 pt (p 0.94); C ≈ D. P1, P2, P3 not supported.
11. ~~Failure analysis~~ — 6 November (`results/failure_analysis/`): BM25 hits where ColPali misses on only 3/225 test queries (0 exact); 42/49 of C's losses come from BM25 misses; lexical overlap explains BM25 success.
12. ~~Figures~~ — 6 November: `scripts/make_figures.py` → `results/figures/fig1–6` (listed in README).
13. ~~Literature matrix~~ — 6 November (`docs/literature_matrix.md`): 8 papers + 3 related, search approach, refined research gap. User must read and confirm.
14. ~~Report draft~~ — 7 November: `report/main.tex` → `report/main.pdf` (19 pp.). Every number comes from `report/generated/` (`scripts/make_report_tables.py`). Red TODOs: last name/ID, ±2 tolerance confirmation, DocVQA license, paper hardware, venues, AI-use self-verification, tag hash.
15. Next: presentation; README cleanup (stale "TBD"/status sections); rename PDF to `<LastName>_YZM5110_Research.pdf`; tag `v1.0-submission`.

### Implementation notes

- `src/evaluation.py` reproduces MTEB 1.16's `RetrievalEvaluator` exactly for nDCG/Recall/P/MAP (cross-checked: max |diff| = 0 on 451×500 bf16-tied scores, 3 Nov). MRR uses trec_eval tie order (score desc, doc id desc), so it can differ slightly from MTEB's MRR on ties.
- We persist the **query × page score matrix** (`scores.npz`, ~1 MB) rather than the page embeddings (~126 MB): every downstream experiment (fusion, analysis) only needs ColPali's scores, and the matrix is small enough to commit. Embeddings can be regenerated with the committed script.
- Getting outputs out of Colab: text outputs can be read from the page; binary ones (`scores.npz`) are zipped and downloaded through the browser — **ask the user before each download**. Downloads Claude triggers can arrive late or not at all (seen 3 Nov); if nothing arrives within a minute, ask the user to click ▶ on the download cell. Downloads land in `~/Downloads`; verify contents (`diff`, sha256) before copying into `results/`.
- `scripts/evaluate_scores.py` re-evaluates any saved `scores.npz` on CPU — run it after every GPU run as an independent check.
