#!/bin/bash
#SBATCH --job-name=c3-contam
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069,gnode076
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/home2/%u/c3/logs/contam-%j.out
#
# The T4 contamination screen for one model (scripts/contamination_screen.py):
# 50-token prefix, greedy 50-token continuation, longest verbatim run. About 50
# forward passes of ~100 tokens per document, so 2,000 documents fit one job.
#
#   sbatch --gres=gpu:2 --export="ALL,C3_MODEL=llada-8b-base,C3_PRECISION=fp16,C3_DEVICE_MAP=auto,C3_MAX_MEMORY=0=9GiB;1=9GiB,C3_THRESHOLD=32" slurm/contamination.sbatch
#
# Resumable: per-document results go to <out>.partial.jsonl on /home2, so a
# requeued job (node failure) continues where it stopped.
#
# C3_THRESHOLD is required until c3.CONTAM_THRESHOLD is fixed. The output is
# written to $PERSIST/contamination_<model>.json (small) for copying into artifacts/.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
MODEL="${C3_MODEL:?set C3_MODEL}"
PRESET=""
PRECISION="${C3_PRECISION:?set C3_PRECISION}"
DOCS="${C3_DOCS:-$PERSIST/corpus/calibration_docs.jsonl}"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
export HF_HOME="${C3_SCRATCH:-/ssd_scratch/$USER}/hf"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$PERSIST"
cd "$REPO_DIR"
echo "[stage] node=$(hostname) model=$MODEL precision=$PRECISION commit=$C3_GIT_COMMIT"

source "$REPO_DIR/slurm/stage_model.sh"

ARGS=(--model "$MODEL" --precision "$PRECISION" --docs "$DOCS"
      --out "$PERSIST/contamination_${MODEL}.json")
[ -n "${C3_THRESHOLD:-}" ] && ARGS+=(--threshold "$C3_THRESHOLD")
[ -n "${C3_LIMIT:-}" ] && ARGS+=(--limit "$C3_LIMIT")
[ -n "${C3_DEVICE_MAP:-}" ] && ARGS+=(--device-map "$C3_DEVICE_MAP")
[ -n "${C3_MAX_MEMORY:-}" ] && ARGS+=(--max-memory "${C3_MAX_MEMORY//;/,}")
"$C3_PY" scripts/contamination_screen.py "${ARGS[@]}" "${MODEL_PATH_ARG[@]}"
