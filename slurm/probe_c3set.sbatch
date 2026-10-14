#!/bin/bash
#SBATCH --job-name=c3-probe-c3set
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069,gnode076
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M
#SBATCH --time=12:00:00
#SBATCH --output=/home2/%u/c3/logs/probe-c3set-%j.out
#
# C: set-level C3 on the JSON probe (scripts/probe_c3set.py) with the fine-tuned 110M:
#
#   sbatch --export=ALL,C3_FT_RUN=ft110m-v4 slurm/probe_c3set.sbatch
#
# Pulls /share1/$USER/probe/$C3_FT_RUN/final.pt, writes artifacts/probe/results/
# c3set_<run>.json (+ steps and calibration-state parquet, calibrators).  Exit 1 if G6
# fails.  C3_PROBE_ARGS passes through (e.g. --no-field for the item-8 gate's 'no').
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
RUN="${C3_FT_RUN:?set C3_FT_RUN, e.g. ft110m-v4}"
REMOTE="${C3_FT_REMOTE:-/share1/$USER/probe/$RUN/final.pt}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
export HF_HOME="$C3_SCRATCH/hf"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO_DIR"
CKPT="$C3_SCRATCH/probe-c3set-$RUN.pt"
rsync -a --partial-dir=.rsync-partial -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$REMOTE" "$CKPT" \
    || { echo "[probe-c3] FATAL: could not pull $REMOTE"; exit 1; }
echo "[probe-c3] $RUN on $(hostname -s), commit $C3_GIT_COMMIT"
# shellcheck disable=SC2086
"$C3_PY" scripts/probe_c3set.py --model local-110m --model-path "$CKPT" --tag "$RUN" \
    --out-dir artifacts/probe/results ${C3_PROBE_ARGS:-}
