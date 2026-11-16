"""Post-hoc set 3, H1 (descriptive, after the single test run): where does the visual reranker's change come from?

For B_vr = ColPali's top 20 reordered by Qwen3-VL-Reranker-2B (frozen N = 20), on dev and test separately:
  - ColPali misses (gold rank > 5) and how many are reachable (gold rank 6-20);
  - fixed = ColPali miss -> B_vr top 5; broken = ColPali top 5 -> B_vr miss;
  - gold pages at rank 1 before and after.
Reads frozen scores only; changes no setting. Writes results/posthoc3/rerank_flow.json.

Usage:
    python scripts/posthoc3_rerank_flow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_fusion_dev import load_scores  # noqa: E402
from src.data import load_task  # noqa: E402
from src.evaluation import evaluate, scores_to_run  # noqa: E402
from src.rerank import rerank_top_n  # noqa: E402
from src.runinfo import REPO_ROOT, collect_metadata  # noqa: E402


def main() -> None:
    task = load_task("vidore/docvqa_test_subsampled", "49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6", "test")
    split = json.loads((REPO_ROOT / "results/splits/docvqa_split.json").read_text())
    n = json.loads((REPO_ROOT / "results/posthoc3/dev_selection_rerank.json").read_text())["systems"]["B_vr"]["selected_n"]
    B = load_scores("results/colpali_v1.3_docvqa/scores.npz", task.queries, task.page_ids)
    VR = load_scores("results/visual_reranker_docvqa/scores.npz", task.queries, task.page_ids)
    R = rerank_top_n(B, VR, n, task.page_ids)
    rb = {q: v["gold_rank"] for q, v in evaluate(task.qrels, scores_to_run(B, task.queries, task.page_ids))[1].items()}
    rr = {q: v["gold_rank"] for q, v in evaluate(task.qrels, scores_to_run(R, task.queries, task.page_ids))[1].items()}

    record = collect_metadata({}, "none", "cpu", 0)
    record["label"] = f"descriptive, after the post-hoc set 3 test run; B_vr N = {n} (frozen)"
    for name in ("dev", "test"):
        qs = split[name]
        record[name] = {
            "n": len(qs),
            "B_misses": sum(rb[q] > 5 for q in qs),
            "B_misses_reachable_rank_6_to_N": sum(5 < rb[q] <= n for q in qs),
            "fixed": sum(rb[q] > 5 and rr[q] <= 5 for q in qs),
            "broken": sum(rb[q] <= 5 and rr[q] > 5 for q in qs),
            "rank1_B": sum(rb[q] == 1 for q in qs),
            "rank1_B_vr": sum(rr[q] == 1 for q in qs),
        }
        print(name, json.dumps(record[name]))
    out = REPO_ROOT / "results/posthoc3/rerank_flow.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
