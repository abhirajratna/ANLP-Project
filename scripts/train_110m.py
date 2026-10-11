#!/usr/bin/env python3
"""Train the 110M masked diffusion LM (resumable; see slurm/train_110m.sbatch)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The budget helpers are torch-free, so --budget-only works in a plain analysis env.
from c3.training.budget import N_PARAMS_DEFAULT, TrainBudget, compare_budgets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", help="index.json from build_training_shards.py")
    ap.add_argument("--out-dir", help="checkpoint directory (must persist across jobs); use a "
                                      "fresh one per run -- a mismatched checkpoint is refused")
    ap.add_argument("--max-tokens", type=int, default=10_000_000_000,
                    help="real (non-pad) training tokens; plan v2: 10-20B")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--window-mode", choices=["chunks", "prefix"], default="chunks",
                    help="chunks: every 256-token chunk of every document (no packing); "
                         "prefix: only each document's first 256 tokens (the eval setting)")
    ap.add_argument("--objective", choices=["mdlm", "ar"], default="mdlm",
                    help="mdlm: the diffusion model; ar: the same-size autoregressive reference")
    ap.add_argument("--init-from", default=None, help="weights-only start, for fine-tuning (T9)")
    ap.add_argument("--visible-prefix", type=int, default=0,
                    help="never mask the first N tokens of a window (T9: the rho marker)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default=None, help="wandb team, e.g. ANLP-project")
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--budget-only", action="store_true",
                    help="print the compute budget and exit without training")
    a = ap.parse_args()

    n_params = N_PARAMS_DEFAULT
    print(compare_budgets(n_params, a.n_gpus))
    print("\nthis run: " + TrainBudget(n_params, a.max_tokens, a.n_gpus).report())
    if a.budget_only:
        return 0
    if not a.index or not a.out_dir:
        raise SystemExit("--index and --out-dir are required unless --budget-only")

    from c3.training.model import ModelConfig
    from c3.training.train import TrainConfig, train

    mcfg = ModelConfig(max_len=a.seq_len, causal=a.objective == "ar")
    cfg = TrainConfig(index_path=a.index, out_dir=a.out_dir, seq_len=a.seq_len,
                      batch_size=a.batch_size, grad_accum=a.grad_accum, lr=a.lr,
                      warmup_steps=a.warmup_steps, max_tokens=a.max_tokens,
                      ckpt_every=a.ckpt_every, eval_every=a.eval_every,
                      eval_batches=a.eval_batches, ema_decay=a.ema_decay,
                      window_mode=a.window_mode, objective=a.objective,
                      init_from=a.init_from, visible_prefix=a.visible_prefix,
                      run_name=a.run_name,
                      seed=a.seed, wandb_project=a.wandb_project,
                      wandb_entity=a.wandb_entity)
    train(cfg, mcfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
