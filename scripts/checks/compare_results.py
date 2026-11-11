"""Compare regenerated outputs in a fresh clone with the committed ones.

Quality numbers must match exactly. Timings, run metadata (timestamps, commit, packages, hardware) and the
resource tables are expected to differ and are ignored. Figures are compared pixel by pixel.

Usage:
    python scripts/checks/compare_results.py --committed <repo> --fresh <clone>
Exit code 0 = everything matches, 1 = at least one mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

JSON_FILES = [
    "results/bm25_docvqa/metrics.json",
    "results/splits/docvqa_split.json",
    "results/fusion_docvqa/dev_selection.json",
    "results/fusion_docvqa/dev_selection_weighted.json",
    "results/test_evaluation/metrics.json",
    "results/failure_analysis/analysis.json",
]
TEXT_FILES = [
    "results/bm25_docvqa/per_query.csv",
    "results/splits/docvqa_query_tags.csv",
    "results/test_evaluation/per_query.csv",
    "results/failure_analysis/changed_queries.csv",
    "results/failure_analysis/examples.md",
]
# keys that legitimately change between runs (anywhere in the JSON tree)
VOLATILE = {"timestamp_utc", "git_commit", "git_dirty", "python", "packages", "hardware", "command", "timings",
            "latency", "peak_memory", "resources", "config_path",
            "tokenizer"}  # run metadata added 2026-11-11 (records which BM25 tokenizer a run used; official = default)
# report macros/tables that hold timings
VOLATILE_MACROS = ("latA", "latB", "latBGE", "fusionOverhead", "idxB", "peakGPU")


def diff_json(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
            else:
                out += diff_json(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff_json(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: {str(a)[:60]} != {str(b)[:60]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--committed", required=True, type=Path)
    ap.add_argument("--fresh", required=True, type=Path)
    args = ap.parse_args()
    bad = 0

    for f in JSON_FILES:
        d = diff_json(json.loads((args.committed / f).read_text()), json.loads((args.fresh / f).read_text()))
        # the committed BM25 run resolved the text revision at run time; the config now pins it
        d = [x for x in d if not x.startswith("/config/data/text_revision")]
        print(f"{'OK  ' if not d else 'DIFF'} {f}" + ("" if not d else f"  ({len(d)} differences)"))
        for x in d[:10]:
            print("     ", x)
        bad += bool(d)

    for f in TEXT_FILES:
        same = (args.committed / f).read_text() == (args.fresh / f).read_text()
        print(f"{'OK  ' if same else 'DIFF'} {f}")
        bad += not same

    def macros(p: Path) -> list[str]:
        return [ln for ln in p.read_text().splitlines() if not any(f"\\{m}}}" in ln for m in VOLATILE_MACROS)]

    same = macros(args.committed / "report/generated/numbers.tex") == macros(args.fresh / "report/generated/numbers.tex")
    print(f"{'OK  ' if same else 'DIFF'} report/generated/numbers.tex (timing macros ignored)")
    bad += not same
    for p in sorted((args.committed / "report/generated").glob("tab_*.tex")):
        if p.name == "tab_resources.tex":
            continue
        same = p.read_text() == (args.fresh / "report/generated" / p.name).read_text()
        print(f"{'OK  ' if same else 'DIFF'} report/generated/{p.name}")
        bad += not same

    from PIL import Image, ImageChops

    for p in sorted((args.committed / "results/figures").glob("*.png")):
        a = Image.open(p).convert("RGB")
        b = Image.open(args.fresh / "results/figures" / p.name).convert("RGB")
        same = a.size == b.size and ImageChops.difference(a, b).getbbox() is None
        print(f"{'OK  ' if same else 'DIFF'} results/figures/{p.name} (pixels)")
        bad += not same

    print("RESULT:", "all quality outputs reproduced exactly" if not bad else f"{bad} file(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
