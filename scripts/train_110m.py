#!/usr/bin/env python3
"""Train the 110M masked diffusion LM (resumable; see slurm/train_110m.sbatch)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.training.budget import TrainBudget, compare_budgets
from c3.training.model import ModelConfig
from c3.training.train import TrainConfig, train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", help="index.json from build_training_shards.py")
    ap.add_argument("--out-dir", help="checkpoint directory (must persist across jobs)")
    ap.add_argument("--max-tokens", type=int, default=6_000_000_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--budget-only", action="store_true",
                    help="print the compute budget and exit without training")
    a = ap.parse_args()

    mcfg = ModelConfig(max_len=a.seq_len)
    print(compare_budgets(mcfg.n_params, a.n_gpus))
    print("\nthis run: " + TrainBudget(mcfg.n_params, a.max_tokens, a.n_gpus).report())
    if a.budget_only:
        return 0
    if not a.index or not a.out_dir:
        raise SystemExit("--index and --out-dir are required unless --budget-only")

    cfg = TrainConfig(index_path=a.index, out_dir=a.out_dir, seq_len=a.seq_len,
                      batch_size=a.batch_size, grad_accum=a.grad_accum, lr=a.lr,
                      max_tokens=a.max_tokens, ckpt_every=a.ckpt_every,
                      wandb_project=a.wandb_project)
    train(cfg, mcfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
