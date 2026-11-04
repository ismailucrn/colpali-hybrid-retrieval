# Deviations from the original setup

Every difference between our setup and the ColPali paper (arXiv 2407.01449), and what it means for the comparison. Fill in as soon as a deviation is known, not at the end.

| # | Aspect | Paper (published) | Ours (measured setup) | Reason | Effect on comparability |
|---|---|---|---|---|---|
| 1 | Checkpoint | `vidore/colpali` (v1.0): DocVQA nDCG@5 54.4 (Table 2, arXiv v6) | `vidore/colpali-v1.3` @ `b5c6dd6` (model-card DocVQA nDCG@5 0.5845) | v1.0 is marked not supported in colpali-engine 0.3.18 | reproduction is checked against the v1.3 model-card score (authors' self-reported, not peer-reviewed); the paper's 54.4 is context only |
| 2 | Task / subset | ViDoRe v1, 10 tasks | DocVQA only (`vidore/docvqa_test_subsampled`, 500 pages, 451 unique queries) | scope: one task done well | no claim about the other tasks |
| 3 | Query set | full task query set (451 deduplicated queries) | full set for reproduction; dev/test split for fusion experiments | tuning requires a held-out dev set | reproduction comparable; fusion results are on a subset (wider CIs) |
| 4 | Hardware / precision | *(to verify)* | NVIDIA L4 (Colab Pro), bf16 | available GPU | precision matches the authors' README examples (bf16); hardware differs |
| 5 | Page text for BM25 | Unstructured (layout detection + Tesseract OCR); BM25 nDCG@5 36.8 | `vidore/docvqa_test_subsampled_tesseract` `text_description` *(whether identical to the paper's pipeline: unverified)* | released by the authors; no own OCR needed | measured 0.3907 vs 36.8: comparable in kind (same BM25 code and parameters as the benchmark), not in input text |
| 6 | Library versions | *(to verify)* | colpali-engine 0.3.18, transformers 5.22.0, torch 2.11.0 | latest release; the paper predates these | small numerical differences possible; to check against published scores |

## Which comparisons still hold

- **B vs published (v1.3 model card):** same checkpoint, dataset, query deduplication, qrels and metric implementation (MTEB-equivalent); differs in library versions, hardware and possibly bf16 kernels. Measured 0.5752 vs 0.5845 (−0.93 pt). Comparable, with the caveat that the model card's own environment is undocumented.
- **B vs paper (v1.0, 54.4):** different checkpoint — not a reproduction of the paper's number; context only.
- **All fusion comparisons (A–E):** run by us under identical conditions, so they do not depend on matching the published environment.
