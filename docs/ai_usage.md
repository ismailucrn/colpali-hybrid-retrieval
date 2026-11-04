# AI-use disclosure

The assignment allows disclosed AI assistance for search, coding and editing. This file records which tools were used and what each contributed, so the report's disclosure section can be written from it. The author remains responsible for reading sources, understanding the code, running experiments, checking citations and verifying every claim.

| Date | Tool | What it contributed | What the author did / verified |
|---|---|---|---|
| 2026-10-23 → 2026-11-03 | Claude (claude.ai chat) | Discussed topic options; helped scope the project and draft the research plan (systems A–E, metrics, protocol); drafted the first `CLAUDE.md`. | Chose the topic and main paper; decided scope. |
| 2026-11-03 | Claude Code (Claude Opus 5.5) | Reviewed `CLAUDE.md` against the assignment brief and revised it (evaluation split, pre-registered expectations, query tagging, control D conditions); set up the git remote, repository skeleton, README, license and docs templates. | Approved changes; *(to fill in)*. |
| 2026-11-03 | Claude Code (Claude Opus 5.5) | Checked `colpali-engine` README/PyPI and HF Hub metadata; wrote `scripts/smoke_test.py`, `src/runinfo.py`, `configs/smoke_test.yaml` and the Colab notebook; tested data loading/processor/scoring locally. | Runs the notebook on Colab; *(to fill in)*. |
| 2026-11-03 | Claude Code (Claude Opus 5.5), via the Claude app's built-in browser | Ran the smoke-test notebook on Colab (selected L4, ran all cells), diagnosed and fixed the torchao/peft conflict, copied the printed outputs into `results/smoke_test/`, pinned `requirements.txt`, logged the run. | Signed in to Google; *(to fill in)*. |
| 2026-11-03 | Claude Code (Claude Opus 5.5) | Verified published DocVQA scores (paper Table 2, model cards) and the official protocol in `vidore-benchmark`; wrote `src/data.py`, `src/evaluation.py` (cross-checked against MTEB 1.16), tests, `scripts/run_colpali.py`, its config and Colab notebook. | Chose DocVQA and checkpoint v1.3; *(to fill in)*. |
| 2026-11-03 | Claude Code (Claude Opus 5.5) | Wrote the rule-based query tagger (`src/query_types.py`) and tests, the stratified split script; ran both; froze `docs/query_tagging.md`. Also downloaded/verified System B outputs and wrote `scripts/evaluate_scores.py`. | Approved freezing; *(to fill in: spot-check of tags)*. |
| 2026-11-04 | Claude Code (Claude Opus 5.5) | Read the official BM25 baseline code; wrote `src/bm25.py`, `scripts/run_bm25.py`, config and tests; ran System A on the Mac; logged results. | *(to fill in)* |
