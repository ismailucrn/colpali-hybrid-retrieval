#!/usr/bin/env bash
# Fresh-clone reproduction check of everything that runs on CPU.
#
# Clones the committed HEAD of this repository into a temporary directory, re-runs the CPU pipeline
# (BM25, split, dev selection, single test evaluation, failure analysis, post-hoc sets 1 and 2, figures, report
# tables) from the committed score matrices, and compares the outputs with the committed ones
# (scripts/checks/compare_results.py). Not re-run: ColPali and BGE-M3 encoding and the cross-encoder
# (need a GPU / torch); their committed score files are inputs.
#
# Usage:  bash scripts/checks/reproduce_cpu_pipeline.sh [python]
#         python defaults to `python3`; it needs requirements-cpu.txt installed.
set -euo pipefail

PY="${1:-python3}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "repo:   $REPO @ $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD || echo ' (working tree has uncommitted changes; the clone uses HEAD)')"
echo "python: $("$PY" --version 2>&1) ($PY)"
echo "clone:  $WORK/repo"
git clone -q "$REPO" "$WORK/repo"
cd "$WORK/repo"

run() { echo "== $*"; "$PY" "$@" > "$WORK/out.log" 2>&1 || { tail -20 "$WORK/out.log"; exit 1; }; }
run scripts/run_bm25.py --config configs/bm25_docvqa.yaml
run scripts/make_split.py --seed 0 --dev-fraction 0.5
run scripts/run_fusion_dev.py --config configs/fusion_docvqa.yaml
run scripts/run_weighted_fusion_dev.py --config configs/fusion_weighted_docvqa.yaml
run scripts/run_test_evaluation.py --config configs/test_evaluation.yaml
run scripts/failure_analysis.py
run scripts/run_bm25.py --config configs/bm25_codes_docvqa.yaml                 # post-hoc A_code
run scripts/run_cc_fusion_dev.py --config configs/fusion_cc_docvqa.yaml         # post-hoc CC, dev
run scripts/run_test_evaluation.py --config configs/posthoc_test_evaluation.yaml # post-hoc test
run scripts/run_posthoc2_dev.py --stage hybrid                                  # post-hoc set 2, dev
run scripts/run_posthoc2_dev.py --stage rerank    # uses the committed cross-encoder scores (MPS run, not repeated)
run scripts/run_test_evaluation.py --config configs/posthoc2_test_evaluation.yaml # post-hoc set 2 test
run scripts/ocr_quality.py                                                       # post-hoc set 2, OCR proxy
run scripts/checks/bge_empty_pages.py                                            # diagnostic, dev only
run scripts/make_figures.py
run scripts/make_report_tables.py
run -m pytest -q tests

echo "== compare with committed outputs"
git clone -q "$REPO" "$WORK/committed"   # untouched copy of the committed outputs
"$PY" "$WORK/committed/scripts/checks/compare_results.py" --committed "$WORK/committed" --fresh "$WORK/repo"
