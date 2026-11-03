"""Dry run of scripts/run_colpali.py on CPU with a fake model (no GPU, no 6 GB download).

Exercises the real data loading, processor, scoring, evaluation and output writing; scores are
meaningless (random embeddings), so the metrics should be near chance (nDCG@5 ~ 0.0x).

Usage:
    python scripts/checks/dry_run_colpali.py --out /tmp/colpali_dry_run
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import colpali_engine.models as cm  # noqa: E402


class FakeModel(torch.nn.Module):
    device = torch.device("cpu")
    dtype = torch.bfloat16

    def forward(self, **batch):
        ids = batch["input_ids"]
        g = torch.Generator().manual_seed(int(ids.sum()) % 2**31)
        x = torch.randn(ids.shape[0], ids.shape[1], 128, generator=g)
        x = x / x.norm(dim=-1, keepdim=True)
        return (x * batch["attention_mask"].unsqueeze(-1)).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO / "configs/colpali_v1.3_docvqa.yaml"))
    parser.add_argument("--out", required=True, help="output directory (outside results/)")
    args = parser.parse_args()

    cm.ColPali.from_pretrained = classmethod(lambda cls, *a, **k: FakeModel())
    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["device"] = "cpu"
    cfg["latency"] = {"n_queries": 5, "warmup": 1}
    cfg["output_dir"] = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    cfg_path = os.path.join(args.out, "config.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"))

    import run_colpali

    sys.argv = ["run_colpali.py", "--config", cfg_path]
    run_colpali.main()


if __name__ == "__main__":
    main()
