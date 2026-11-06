# colpali-hybrid-retrieval

Reproducing ColPali and evaluating hybrid BM25 + ColPali fusion for visual document retrieval (YZM5110 Research Project).

> **Status (3 November 2026):** repository skeleton only. No experiments have been run yet; no numbers in this repository are results until they appear under `results/` with their run metadata.

## Research question

Does fusing lexical retrieval (BM25 over page text) with ColPali improve page retrieval over ColPali alone, and for which query types? *(Hypothesis, not a result.)*

- Main paper: Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models* — https://arxiv.org/abs/2407.01449
- Reference implementation: https://github.com/illuin-tech/colpali (`colpali-engine`)
- Benchmark: ViDoRe — https://huggingface.co/vidore

## Systems

| ID | System | Role |
|---|---|---|
| A | BM25 over page text | Text-only baseline |
| B | ColPali (published checkpoint, inference only) | Reproduced baseline |
| C | RRF(BM25, ColPali) | Proposed improvement |
| D | RRF(dense text retriever, ColPali) | Control: lexical signal vs. ensembling |
| E | RRF `k` sensitivity | Ablation |

Primary metric: nDCG@5. Also Recall@5/10, nDCG@10, MRR, and resource metrics (query latency, indexing time, index size, peak GPU memory). See `CLAUDE.md` for the full protocol and pre-registered expectations.

## Repository layout

```
configs/    experiment configs (YAML); every run reads one
scripts/    entry points (smoke test, indexing, retrieval, evaluation, tables/figures)
src/        reusable code
notebooks/  Colab notebooks (thin wrappers around scripts/)
results/    small saved outputs: run files, metrics, tables, figures
data/       local data cache (gitignored)
docs/       experiment log, deviations, literature matrix, query tagging rules, AI-use disclosure
```

## Setup

*TBD after the first working run.* Dependencies will be pinned in `requirements.txt`.

- ColPali inference requires a GPU (planned: Google Colab / Kaggle, ~16 GB). BM25, fusion, evaluation and analysis run on CPU.
- The PaliGemma-based checkpoint may require accepting its license on Hugging Face and an HF token. Provide the token via the `HF_TOKEN` environment variable or Colab secrets; never commit it.

### Smoke test

Checks that the checkpoint loads, encodes 8 ViDoRe pages and queries, and records timing and peak memory (needs a GPU; ~6 GB of weights are downloaded).

- Colab: open `notebooks/smoke_test_colab.ipynb`, select a T4 GPU runtime, run all cells.
- Any machine with Python ≥3.10:

```bash
pip install -r requirements.txt
python scripts/smoke_test.py --config configs/smoke_test.yaml
```

Output: `results/smoke_test/<timestamp>_<device>.json` (scores, timings, peak memory, package versions, git commit, model and dataset revisions).

### System B — ColPali v1.3 on DocVQA (GPU)

Encodes all 500 pages and 451 queries of `vidore/docvqa_test_subsampled`, evaluates with the official ViDoRe protocol and records resource use.

- Colab: open `notebooks/colpali_docvqa_colab.ipynb` on an L4 GPU runtime and run all cells.
- Elsewhere: `python scripts/run_colpali.py --config configs/colpali_v1.3_docvqa.yaml`

Output: `results/colpali_v1.3_docvqa/` — `scores.npz` (query × page scores), `metrics.json`, `per_query.csv`.

### System A — BM25 on DocVQA (CPU, seconds)

```bash
python scripts/run_bm25.py --config configs/bm25_docvqa.yaml
```

Output: `results/bm25_docvqa/`. NLTK data is downloaded once into `data/nltk_data/` (gitignored).

### Control D's text component — BGE-M3 on DocVQA (Mac MPS / GPU, ~1 min; CPU works but slower)

```bash
python scripts/run_bge_m3.py --config configs/bge_m3_docvqa.yaml
```

Output: `results/bge_m3_docvqa/`.

### Fusion, test evaluation, failure analysis and figures (CPU, seconds — no GPU, no model download)

```bash
python scripts/run_fusion_dev.py --config configs/fusion_docvqa.yaml                  # RRF k on dev (C, D)
python scripts/run_weighted_fusion_dev.py --config configs/fusion_weighted_docvqa.yaml # ablation E on dev
python scripts/run_test_evaluation.py --config configs/test_evaluation.yaml            # the single test run
python scripts/failure_analysis.py                                                    # test, descriptive
python scripts/make_figures.py                                                        # results/figures/
```

These read only the committed `scores.npz` files and split/tag files, so every table and figure can be regenerated on a laptop.

| Figure | File | Shows |
|---|---|---|
| 1 | `results/figures/fig1_test_ndcg5` | test nDCG@5 with 95% CI, Δ vs ColPali (Holm p) |
| 2 | `results/figures/fig2_dev_k_sweep` | dev nDCG@5 vs RRF k for C and D (ablation E) |
| 3 | `results/figures/fig3_dev_weighted_rrf` | dev Δ vs ColPali over the weighted-RRF grid |
| 4 | `results/figures/fig4_test_by_type` | test nDCG@5 by query type; Δ of fusions vs ColPali |
| 5 | `results/figures/fig5_headroom` | who finds the gold page (top 5), per query type |
| 6 | `results/figures/fig6_overlap` | hit rate vs query–OCR lexical overlap |

### Fast tests (CPU, seconds)

```bash
python -m pytest -q tests
```

## Data access

*TBD once the ViDoRe task is chosen:* dataset name, revision, license and download command.

## Reproducing the results

*TBD.* Exact commands for each table and figure, plus a small fast test that runs on a few pages/queries.

## AI use

AI assistance is disclosed in [`docs/ai_usage.md`](docs/ai_usage.md).

## License

Code: MIT (see `LICENSE`). Datasets and model checkpoints keep their own licenses.
