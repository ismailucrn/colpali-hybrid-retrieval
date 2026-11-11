# colpali-hybrid-retrieval

Reproducing ColPali on ViDoRe DocVQA and testing whether fusing it with BM25 helps (YZM5110 Multimodal Learning, research project).

> **Status (11 November 2026):** all experiments are done and their outputs are committed. The report draft is `report/main.pdf`. The submitted version will be pinned with the tag `v1.0-submission`.

## Summary

**Question.** Does fusing lexical retrieval (BM25 over page text) with ColPali improve page retrieval over ColPali alone, and for which query types? Is any gain due to the *lexical* signal, or to ensembling with any second retriever?

**Answers** (all measured by us; details and uncertainty in `results/test_evaluation/summary.md` and the report):

- **Reproduction.** ColPali v1.3 reaches **nDCG@5 0.5752** on the full DocVQA set. The authors' model card reports 0.5845, so the gap is within the ±2-point tolerance fixed before the run.
- **Fusion (test split, 225 queries, run once).** ColPali alone scores 0.570. Adding BM25 makes it significantly worse:

  | System | nDCG@5 | Δ vs ColPali | Holm-corrected p |
  |---|---|---|---|
  | RRF(BM25, ColPali) | 0.516 | −0.054 | 0.0006 |
  | RRF(BGE-M3, ColPali), the dense-text control | 0.519 | −0.051 | 0.0006 |
  | Dev-tuned weighted RRF | 0.571 | +0.001 | 0.94 |

- **Why it fails.** On only 3 of 225 test queries does BM25 find the gold page when ColPali misses it. BM25 works only when the query's words survive OCR (`results/failure_analysis/`).

## Systems

| ID | System | Role |
|---|---|---|
| A | BM25 over the benchmark's Tesseract OCR text | Text-only baseline |
| B | ColPali v1.3 (published checkpoint, inference only) | Reproduced baseline |
| C | RRF(BM25, ColPali), `k` chosen on dev | Proposed improvement |
| D | RRF(BGE-M3 dense vectors over the same OCR text, ColPali) | Control: lexical signal vs. any second retriever |
| E | RRF `k` sweep on dev, plus weighted RRF (C_w, D_w), dev-informed | Ablation |

- **Primary metric:** nDCG@5. We also report Recall@5/10, nDCG@10, MRR@10 and resource cost (indexing time, index size, query latency, peak memory).
- **Pre-registered predictions and protocol:** `CLAUDE.md` and `docs/experiment_log.md`.

## Repository layout

```
configs/    experiment configs (YAML); every run reads one
scripts/    entry points: one per system, the dev selection, the single test run, analysis, figures, report tables
  checks/   independent checks (MTEB cross-check, dry runs, fresh-clone reproduction)
src/        reusable code (data loading and official protocol, evaluation, BM25, fusion, statistics, query tags)
notebooks/  Colab notebooks (thin wrappers around scripts/)
results/    committed outputs: score matrices (scores.npz), metrics, per-query CSVs, splits, figures, console logs
report/     LaTeX source of the report; generated/ holds every number and table, written by a script
slides/     Beamer slides (same generated numbers) and speaker notes
docs/       experiment log, run registry, deviations, literature matrix, query tagging rules, AI-use disclosure
data/       local cache (gitignored)
```

## Setup

There are two environments, depending on what you want to run.

**CPU only: reproduce every table and figure from the committed score matrices.** No GPU, no torch, no model download.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
```

**Full: re-encode with the models.**

```bash
# install a torch build for your hardware first (tested: 2.11.0+cu130 on Colab L4, 2.13.0 on macOS/MPS)
pip install -r requirements.txt
```

- Versions are pinned in `requirements.txt`. The complete Colab environment is in `results/colpali_v1.3_docvqa/pip_freeze_colab.txt`.
- Every `metrics.json` records the package versions, git commit, hardware and dataset/model revisions of its run.
- On Colab, run `pip uninstall -y torchao` first: the preinstalled torchao 0.10 breaks peft 0.20.

### What needs which hardware

| Step | Hardware | Time |
|---|---|---|
| ColPali v1.3 encoding (System B) | GPU with bf16, about 16 GB (we used a Colab Pro NVIDIA L4) | about 2 min encoding, plus a ~6 GB model download |
| BGE-M3 encoding | Apple MPS or a GPU (CPU works, slower) | about 1 min on an M5 |
| Everything else | CPU | seconds; reads the committed `scores.npz` files |

## Data access

All data is downloaded automatically from the Hugging Face Hub at pinned revisions. No token is needed and no data is stored in this repository.

| Dataset | Revision | Used for | License (as stated, checked 2026-11-11) |
|---|---|---|---|
| `vidore/docvqa_test_subsampled` | `49bf8f1` | page images, queries, relevance labels | tagged MIT on its Hugging Face card |
| `vidore/docvqa_test_subsampled_tesseract` | `9a183a7` | OCR page text for BM25 and BGE-M3 | none stated |

The original DocVQA pages (docvqa.org) state no license. They say the images come from the UCSF Industry Documents Library, so please respect that source's terms.

| Model | Revision | License |
|---|---|---|
| `vidore/colpali-v1.3` (LoRA adapter) | `b5c6dd6` | see model card |
| `vidore/colpaligemma-3b-pt-448-base` | `30ab955` | see model card |
| `BAAI/bge-m3` | `5617a9f` | see model card |

The ColPali repositories are not gated.

## Reproducing the results

**Step 1 — GPU, once: ColPali scores.** These are already committed as `results/colpali_v1.3_docvqa/scores.npz`. To recompute them on Colab, open `notebooks/colpali_docvqa_colab.ipynb` on an L4 runtime and run all cells. Elsewhere:

```bash
python scripts/run_colpali.py --config configs/colpali_v1.3_docvqa.yaml
python scripts/evaluate_scores.py --scores results/colpali_v1.3_docvqa/scores.npz \
    --dataset vidore/docvqa_test_subsampled --revision 49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6   # CPU re-evaluation
```

**Step 2 — CPU.** Text systems, split, fusion, statistics, analysis, figures, report:

```bash
python scripts/run_bm25.py --config configs/bm25_docvqa.yaml                           # System A
python scripts/run_bge_m3.py --config configs/bge_m3_docvqa.yaml                       # D's text part (needs torch)
python scripts/make_split.py --seed 0 --dev-fraction 0.5                               # frozen tags + dev/test split
python scripts/run_fusion_dev.py --config configs/fusion_docvqa.yaml                   # RRF k on dev (C, D)
python scripts/run_weighted_fusion_dev.py --config configs/fusion_weighted_docvqa.yaml # ablation E on dev
python scripts/run_test_evaluation.py --config configs/test_evaluation.yaml            # the single test run
python scripts/failure_analysis.py                                                     # test, post-hoc, descriptive
python scripts/make_figures.py                                                         # results/figures/
python scripts/make_report_tables.py                                                   # report/generated/
(cd report && tectonic main.tex)                                                       # report/main.pdf
(cd slides && tectonic slides.tex)                                                     # slides/slides.pdf
```

- The test evaluation refuses to run if the fusion settings differ from the committed dev selections. Re-running it reproduces the committed numbers; it does not re-tune anything.
- The report is built with [tectonic](https://tectonic-typesetting.github.io/) (a self-contained LaTeX engine). Every number in the report comes from `report/generated/`, which `scripts/make_report_tables.py` writes from `results/`.

| Report item | Produced by | File |
|---|---|---|
| Table: published vs measured | `run_colpali.py`, `run_bm25.py`, `run_bge_m3.py` | `results/*/metrics.json` |
| Table: dev selection; report Figure 1 (k sweep, weighted grid) | `run_fusion_dev.py`, `run_weighted_fusion_dev.py` | `results/fusion_docvqa/` |
| Tables: test results, by query type, resources; report Figures 2–3 | `run_test_evaluation.py` | `results/test_evaluation/` |
| Table: headroom; report Figure 4; examples | `failure_analysis.py` | `results/failure_analysis/` |

Figure files (`results/figures/`, made by `make_figures.py`):

| File | Shows |
|---|---|
| `fig1_test_ndcg5` | test nDCG@5 with 95% CI, Δ vs ColPali (Holm p) |
| `fig2_dev_k_sweep` | dev nDCG@5 vs RRF k for C and D |
| `fig3_dev_weighted_rrf` | dev Δ vs ColPali over the weighted-RRF grid |
| `fig4_test_by_type` | test nDCG@5 by query type; Δ of fusions vs ColPali |
| `fig5_headroom` | who finds the gold page (top 5), per query type |
| `fig6_overlap` | hit rate vs query–OCR lexical overlap |

## Tests and checks

```bash
python -m pytest -q tests                              # 29 unit tests, CPU, under a second, no torch
bash scripts/checks/reproduce_cpu_pipeline.sh          # fresh clone, run Step 2, compare with committed results
```

Other checks:
- **Small fast end-to-end test:** `scripts/checks/dry_run_colpali.py` runs the ColPali pipeline on real data with a fake model on CPU.
- **Test-evaluation dry run:** `scripts/checks/dry_run_test_evaluation.py` runs the test evaluation on dev queries.
- **Metric cross-check:** `scripts/checks/check_metrics_against_mteb.py` compares our evaluator with MTEB 1.16 (max |diff| 0).

All logs are in `results/checks/`, and every run, including failed ones, is listed in `docs/run_registry.md`.

## Documentation

| File | Contents |
|---|---|
| `docs/experiment_log.md` | reasoning behind every decision; pre-registrations; interpretations |
| `docs/run_registry.md` | evidence: every execution with time, machine, commit, status and log |
| `docs/deviations.md` | differences from the original paper and which comparisons still hold |
| `docs/query_tagging.md` | frozen query-type rules and split |
| `docs/literature_matrix.md` | 8-paper comparison matrix, search approach, research gap |
| `docs/ai_usage.md` | AI-use disclosure |

## License

Code: MIT (see `LICENSE`). Datasets and model checkpoints keep their own licenses (see Data access).
