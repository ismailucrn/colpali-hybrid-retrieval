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
- **Query-type tagging (before any system is run):** tags (exact-match/numeric, visual/chart, general) are assigned from written rules in `docs/query_tagging.md`, using only the query (and its gold page if needed), never system outputs. The tag file is committed before results exist.
- Uncertainty: per-query bootstrap confidence intervals and a paired test between systems.
- Failure analysis: per query type, where each system wins/loses, with concrete examples — favorable *and* unfavorable.
- Any `k` sensitivity curve shown on test is post-hoc and labeled as such; selection uses dev only.
- Report differences from the original paper (checkpoint version, OCR pipeline, hardware, precision, subset) in `docs/deviations.md`.

## Compute

- Not yet confirmed. Plan: Google Colab or Kaggle GPU (~16 GB, e.g. T4/P100) for ColPali; laptop only for BM25, analysis and writing.
- Do **not** assume local ColPali inference on the user's laptop works.
- Compute ColPali embeddings **once** and save them to disk (e.g. Google Drive) so sessions/disconnects don't force recomputation.
- The PaliGemma-based checkpoint may require accepting a license on Hugging Face and an HF token. **Never commit tokens**; use environment variables / Colab secrets.

## Things to verify (do not state these as facts until checked)

- Which ColPali checkpoint version (v1.0 / v1.2 / v1.3 / ...) matches the paper's reported table, and its published nDCG@5 for the chosen task.
- Current `colpali-engine` API and versions — check the official README before writing code.
- Whether the chosen ViDoRe dataset already ships page text (otherwise we OCR ourselves, e.g. Tesseract — a documented deviation).
- Dataset size, license, and number of queries of the chosen task (determines how small the test split will be).
- The user's GPU environment and HF license/token status.
- Whether the topic was submitted to the instructor.

## Current status (as of 3 November 2026)

- GitHub repo connected; skeleton (folders, README, MIT LICENSE, docs templates) pushed on 3 November. The assignment PDF is kept local only (gitignored).
- No smoke test, no task chosen, nothing measured yet.
- About 2 weeks left → plan compressed: ablation E reduced to RRF `k` sensitivity; control D is kept (main source of rigor).

## Timeline (compressed, from 3 November)

- **3–6 Nov:** repo skeleton + docs; GPU environment; smoke test; choose the ViDoRe task; full-set ColPali reproduction with saved embeddings; write pre-registered tolerance.
- **7–10 Nov:** page text + BM25 (A); dense text retriever (D); RRF (C); query tagging + dev/test split committed; evaluation script; choose `k` on dev.
- **11–13 Nov:** single test-set run; resource measurements; bootstrap CIs and paired tests; failure analysis by query type.
- **14–18 Nov:** report, figures, literature matrix completion, repo cleanup, tag `v1.0-submission`, slides.
- Literature matrix runs in parallel: roughly one paper per day.

## Literature (candidate set for the 5+ paper matrix — verify details before citing)

- ColPali (2407.01449) — main paper.
- ColBERT (2004.12832) — late interaction / multi-vector retrieval origin.
- ViDoRe Benchmark V2 (2505.17166) — newer evaluation protocol.
- VisRAG (2410.10594) — vision-based RAG including generation (related work only, not implemented).
- DSE, Document Screenshot Embedding — single-vector visual document retrieval (competing approach).
- Cormack et al., 2009, Reciprocal Rank Fusion — method used in the improvement.
- A text-IR paper on lexical vs dense / hybrid retrieval (e.g. BEIR, 2104.08663 — verify) — evidence that BM25 + dense fusion helps in text retrieval; supports the research gap.
- Lewis et al., 2020, RAG (2005.11401) — background only.

Templates: `docs/literature_matrix.md`.

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
docs/         experiment_log.md, deviations.md, literature_matrix.md, ai_usage.md, query_tagging.md
```

- Every reported number must be reproducible by a committed script + config.
- Every run writes: config used, git commit, package versions, checkpoint name/revision, dataset name/revision, seed, hardware, timestamp.
- Do not commit: credentials, `.env`, large data, embeddings, model checkpoints. Commit small result files (metrics, run files) only.
- Pin dependencies (`requirements.txt` from `pip freeze` after the first working run).
- Provide a small fast test (a few pages/queries) in addition to the full experiment, and state which results need a GPU.

## Next immediate steps

1. ~~Repo skeleton, README, LICENSE, docs templates~~ — done 3 November.
2. Confirm the user's GPU environment (free Colab / Colab Pro / Kaggle / other) and HF access.
3. Write `scripts/smoke_test.py` + a Colab notebook: load ColPali, encode a few page images and queries, print scores, peak GPU memory and timing.
4. Choose the ViDoRe task (mixed text-heavy and visual pages, manageable size, ideally with page text) together with the user.
