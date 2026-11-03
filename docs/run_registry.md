# Run registry

Every execution we did — GPU runs, local checks, **including failed runs** — one row each, in order. This is the evidence that a result was actually produced, and how. Rows are never deleted or edited; a correction is a new row or a dated note.

- **Where:** `colab-L4` = Google Colab Pro, NVIDIA L4, run through the Claude app's built-in browser; `mac` = the author's MacBook (Apple M5, 16 GB), Python 3.11 venv.
- **Log:** console output saved next to the results. For Colab runs it is copied from the notebook page (progress bars abridged).
- **Status:** `ok`, `FAILED` (with cause), or `partial`.
- Times in UTC.

| # | Date / time (UTC) | Where | What (script / notebook) | Commit | Status | Outputs and log | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-03 05:28 | mac | Smoke-test pieces without the model: data streaming, processor, scoring on random embeddings, metadata | `d164f1f` + uncommitted `scripts/smoke_test.py` | ok | none saved (interactive check) | Verified shapes: image 1031 tokens at 448×448. Recorded in `experiment_log.md`. |
| 2 | 2026-11-03 05:44 | colab-L4 | `notebooks/smoke_test_colab.ipynb` → `scripts/smoke_test.py` | `3e86e44` | FAILED | `results/smoke_test/console_20261103_0544_FAILED.log` | Colab's torchao 0.10 incompatible with peft 0.20 at model load. Fixed in `a1c77ff`. |
| 3 | 2026-11-03 05:47 | colab-L4 | same | `a1c77ff` | ok | `results/smoke_test/20261103_054838_cuda.json`, `pip_freeze_colab.txt`, `console_20261103_054838.log` | 7/8 top-1; peak 16.2 GB; 0.49 s/page. |
| 4 | 2026-11-03 ~06:00 | mac | `pytest tests` (6 tests) | `75d64d0` (before commit) | ok | `results/checks/20261103_060614_pytest.log` (re-run at 06:06) | |
| 5 | 2026-11-03 ~06:00 | mac | MTEB 1.16.5 cross-check of `src/evaluation.py` | `75d64d0` + `scripts/checks/` | ok | `results/checks/20261103_060614_mteb_crosscheck.log` | max \|diff\| = 0 for nDCG/Recall/P/MAP; MRR differs on ties (expected). |
| 6 | 2026-11-03 ~06:00 | mac | `scripts/checks/dry_run_colpali.py` (fake model, real data) | `75d64d0` + `scripts/checks/` | ok | `results/checks/20261103_060614_dry_run_colpali.log` | Near-chance metrics as expected with random embeddings; output files written. |
| 7 | 2026-11-03 06:05 | colab-L4 | `notebooks/colpali_docvqa_colab.ipynb` → `scripts/run_colpali.py` | `75d64d0` | FAILED | `results/colpali_v1.3_docvqa/console_20261103_0605_FAILED.log` | Pinned adapter revision was also applied to the base-model repo. Fixed in `0a716fa`. |
| 8 | 2026-11-03 06:08 | colab-L4 | `notebooks/colpali_docvqa_colab.ipynb` → `scripts/run_colpali.py` (System B, full DocVQA) | `0a716fa` (dirty only by an untracked file from run 7) | ok (outputs partly transferred) | `results/colpali_v1.3_docvqa/metrics.json`, `console_20261103_0608.log`; `scores.npz`, `per_query.csv`, `pip_freeze_colab.txt` pending download | **nDCG@5 = 0.5752** (model card 0.5845). Auto-download from the built-in browser did not arrive (3 attempts); metrics.json copied from the page, byte size verified (5828). |
| 9 | 2026-11-03 06:33–06:47 | colab-L4 → mac | Download of run 8 outputs (zip) | – | ok | `results/colpali_v1.3_docvqa/scores.npz` (sha256 `8dbb4d9f…`), `per_query.csv`, `pip_freeze_colab.txt` | Two zips arrived (one triggered by Claude at 06:33, one by the user at 06:47); contents identical (`diff -r`); `metrics.json` identical to the copy taken from the page. Runtime deleted afterwards. |
| 10 | 2026-11-03 06:48 | mac | `scripts/evaluate_scores.py` on run 8's `scores.npz` | `b470e86` + uncommitted script | ok | `results/checks/20261103_064839_reevaluate_colpali_scores.log` | Independent CPU re-evaluation reproduces Colab's metrics exactly (nDCG@5 0.575169). |
