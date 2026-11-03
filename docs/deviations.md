# Deviations from the original setup

Every difference between our setup and the ColPali paper (arXiv 2407.01449), and what it means for the comparison. Fill in as soon as a deviation is known, not at the end.

| # | Aspect | Paper (published) | Ours (measured setup) | Reason | Effect on comparability |
|---|---|---|---|---|---|
| 1 | Checkpoint | *(to verify)* | | | |
| 2 | Task / subset | *(to verify)* | | | |
| 3 | Query set | full task query set | full set for reproduction; dev/test split for fusion experiments | tuning requires a held-out dev set | reproduction comparable; fusion results are on a subset (wider CIs) |
| 4 | Hardware / precision | *(to verify)* | NVIDIA L4 (Colab Pro), bf16 | available GPU | precision matches the authors' README examples (bf16); hardware differs |
| 5 | Page text for BM25 | *(to verify: dataset text or paper's OCR pipeline)* | | | |
| 6 | Library versions | *(to verify)* | colpali-engine 0.3.18, transformers 5.22.0, torch 2.11.0 | latest release; the paper predates these | small numerical differences possible; to check against published scores |

## Which comparisons still hold

*To be written after the reproduction run.*
