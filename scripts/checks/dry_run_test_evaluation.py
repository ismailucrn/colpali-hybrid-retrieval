"""Exercise scripts/run_test_evaluation.py WITHOUT touching the test split: the dev queries are passed in
place of the test list and outputs go to a scratch directory. Used once before the real test run.

Usage:
    python scripts/checks/dry_run_test_evaluation.py --out /tmp/test_eval_dry_run
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    split = json.loads((REPO / "results/splits/docvqa_split.json").read_text())
    (out / "split_dev_as_test.json").write_text(json.dumps({"dev": split["dev"], "test": split["dev"]}))
    cfg = yaml.safe_load((REPO / "configs/test_evaluation.yaml").read_text())
    cfg["splits"], cfg["output_dir"] = str(out / "split_dev_as_test.json"), str(out / "results")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    import scripts.run_test_evaluation as rte

    sys.argv = ["run_test_evaluation.py", "--config", str(out / "config.yaml")]
    rte.main()


if __name__ == "__main__":
    main()
