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
