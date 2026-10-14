#!/bin/bash
#SBATCH --job-name=c3-probe
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:2
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069,gnode076
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home2/%u/c3/logs/probe-%j.out
#
# Probes a node's GPUs, then starts training on that node.
#
# A node can accept GPU jobs while its GPUs are unusable: on 2026-09-28 gnode076
# failed the first training run, and several other users' jobs, with "CUDA
# unknown error".  This job takes the allocation training needs, checks every
# GPU (scripts/probe_gpus.py) both directly and through `srun torchrun` with an
# NCCL all-reduce (the path training uses), and only then submits
# train_110m.sbatch pinned to this node, with verify_110m.sh after it.
#
# If the probe fails, the node is added to C3_EXCLUDE and this job resubmits
# itself, up to C3_PROBE_MAX attempts.  For 4-GPU training submit with
#     sbatch --gres=gpu:4 --export=ALL,C3_NGPU=4,C3_ACCUM=4 slurm/launch_train.sbatch
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
NGPU="${C3_NGPU:-2}"
EXCLUDE="${C3_EXCLUDE:-gnode066,gnode069,gnode076}"
ATTEMPT="${C3_PROBE_ATTEMPT:-1}"
MAX="${C3_PROBE_MAX:-4}"
NODE="$(hostname -s)"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
cd "$REPO_DIR"
echo "[probe] attempt $ATTEMPT/$MAX on $NODE, excluding $EXCLUDE"

probe_ok=1
timeout 120 "$C3_PY" scripts/probe_gpus.py "$NGPU" || probe_ok=0
if [ "$probe_ok" = 1 ]; then
    timeout 180 srun --kill-on-bad-exit=1 "$(dirname "$C3_PY")/torchrun" \
        --nproc_per_node="$NGPU" --nnodes=1 scripts/probe_gpus.py "$NGPU" || probe_ok=0
fi

if [ "$probe_ok" = 0 ]; then
    EXCLUDE="$EXCLUDE,$NODE"
    if [ "$ATTEMPT" -ge "$MAX" ]; then
        echo "[probe] FATAL: $NODE failed and $MAX attempts are used up; nothing submitted"
        exit 1
    fi
    NEXT=$(sbatch --parsable --exclude="$EXCLUDE" \
        --export=ALL,C3_EXCLUDE="$EXCLUDE",C3_PROBE_ATTEMPT=$((ATTEMPT + 1)) \
        slurm/launch_train.sbatch)
    echo "[probe] $NODE FAILED; resubmitted as job $NEXT, excluding $EXCLUDE"
    echo "$(date +%F_%T) probe=$SLURM_JOB_ID failed on $NODE -> probe=$NEXT" >> logs/t3_jobs.txt
    exit 1
fi

# Pinned to the node just verified; the probe's GPUs free up as this job ends.
GRES=()
if [ -n "${C3_NGPU:-}" ]; then GRES=(--gres="gpu:$NGPU"); fi
TRAIN=$(sbatch --parsable --nodelist="$NODE" "${GRES[@]}" slurm/train_110m.sbatch)
VERIFY=$(sbatch --parsable --dependency=afterok:"$TRAIN" --exclude="$EXCLUDE" slurm/verify_110m.sh)
echo "[probe] $NODE passed; submitted train=$TRAIN verify=$VERIFY"
echo "$(date +%F_%T) probe=$SLURM_JOB_ID ok on $NODE -> train=$TRAIN verify=$VERIFY" >> logs/t3_jobs.txt
