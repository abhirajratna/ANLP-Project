#!/bin/bash
#SBATCH --job-name=c3-wandb-log
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --time=01:00:00
#SBATCH --output=/home2/%u/c3/logs/wandb-log-%j.out
#
# Log result JSONs to W&B (scripts/wandb_log_results.py); C3_WANDB_ARGS is passed through:
#
#   sbatch --export=ALL,C3_WANDB_ARGS="--name v4-dev-infill --job-type dev --tags dev --infill /home2/$USER/c3-runs/c3set/dev/infill_sweep1.json" slurm/wandb_log.sbatch
#
# Never fails on a W&B error (the script exits 0); offline runs land in ./wandb.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
cd "$REPO_DIR"
if [ -z "${WANDB_MODE:-}" ]; then
    grep -qs api.wandb.ai "$HOME/.netrc" && export WANDB_MODE=online || export WANDB_MODE=offline
fi
# shellcheck disable=SC2086
"$C3_PY" scripts/wandb_log_results.py ${C3_WANDB_ARGS:?set C3_WANDB_ARGS}
