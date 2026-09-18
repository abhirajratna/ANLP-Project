#!/bin/bash
#SBATCH --job-name=c3-traces
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
# Pin the GPU model: the audit reports results per precision, and a run that
# silently landed on different silicon would not be comparable. gnode066/069 are
# excluded as known-bad (same exclusion as run-llama.sh).
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/traces-%j.out
#
# One model arm of the C1 audit. Usage:
#   sbatch --export=ALL,C3_MODEL=llada-8b,C3_PRECISION=4bit,C3_LIMIT=20 slurm/collect_traces.sbatch
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
MODEL="${C3_MODEL:-llada-8b}"
PRECISION="${C3_PRECISION:-4bit}"
LIMIT="${C3_LIMIT:-20}"
REPEAT="${C3_REPEAT:-1}"

# `module` is a shell function and is not always defined in a non-interactive
# shell, so initialise it explicitly before relying on it.
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
export HF_HOME="${C3_SCRATCH:-/ssd_scratch/$USER}/hf"
mkdir -p "$HF_HOME" "$PERSIST/traces" logs
cd "$REPO_DIR"

# LLaDA-8B already sits on /share1. That is not mounted on compute nodes, but it
# is reachable by ssh back to the login node, so the checkpoint is pulled to
# node-local SSD -- a LAN copy rather than a 15 GB HuggingFace download, and it
# costs nothing against the /home2 quota. The copy persists on the node between
# jobs, so only the first job on a given node pays for it.
SHARED_LLADA="/share1/ishita/llada_model"
MODEL_PATH_ARG=""
if [ -n "${C3_MODEL_PATH:-}" ]; then
    MODEL_PATH_ARG="--model-path $C3_MODEL_PATH"
elif [ "$MODEL" = "llada-8b" ]; then
    LOCAL_LLADA="$C3_SCRATCH/llada_model"
    if [ ! -r "$LOCAL_LLADA/config.json" ]; then
        echo "[stage] pulling LLaDA-8B from $SHARED_LLADA to $LOCAL_LLADA"
        mkdir -p "$LOCAL_LLADA"
        rsync -a --inplace --partial -e "ssh $C3_SSH_OPTS" \
            "$USER@ada:$SHARED_LLADA/" "$LOCAL_LLADA/" \
            || { echo "[stage] FATAL: LLaDA transfer failed"; exit 1; }
    else
        echo "[stage] LLaDA-8B already cached at $LOCAL_LLADA"
    fi
    MODEL_PATH_ARG="--model-path $LOCAL_LLADA"
fi

"$C3_PY" scripts/collect_traces.py \
    --model "$MODEL" $MODEL_PATH_ARG \
    --precision "$PRECISION" \
    --docs "$PERSIST/corpus/calibration_docs.jsonl" \
    --limit "$LIMIT" --repeat "$REPEAT" \
    --out "$PERSIST/traces/${MODEL}-${PRECISION}-n${LIMIT}.parquet"
