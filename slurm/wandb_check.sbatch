#!/bin/bash
#SBATCH --job-name=c3-wandb-check
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:10:00
#SBATCH --output=/home2/%u/c3/logs/wandb-check-%j.out
#
# Can this account's W&B key write to the team project from a compute node?
# A teammate's key started returning HTTP 403 on 2026-10-12, so every logging job
# checks this first: prints WANDB_OK, or WANDB_FAIL with the error. On a failure,
# jobs log offline (c3.training.train.start_wandb falls back by itself) and are
# synced later with slurm/wandb_backfill.sbatch C3_SYNC_OFFLINE=1.
set -uo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
cd "$REPO_DIR"
WANDB_MODE=online "$C3_PY" - <<'PY'
import sys
try:
    import wandb
    run = wandb.init(entity="ANLP-project", project="c3-dllm-calibration",
                     name="wandb-write-check", tags=["check"], mode="online",
                     settings=wandb.Settings(init_timeout=120))
    run.log({"ok": 1})
    url = run.url
    run.finish()
    print(f"WANDB_OK {url}")
except Exception as exc:
    print(f"WANDB_FAIL {type(exc).__name__}: {exc}")
    sys.exit(1)
PY
