#!/bin/bash
#SBATCH --job-name=c3-train110m
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --gres=gpu:4
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
# medium QOS caps this account's *aggregate* usage at cpu=40, gres/gpu=4,
# mem=125000M. A 32-CPU request is fine on its own but is rejected outright --
# "CPU count per node can not be satisfied" -- whenever another of this
# account's jobs already holds CPUs. 16 is ample: the loader memory-maps shards
# and does no tokenisation.
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home2/%u/c3/logs/train110m-%j.out
#
# The 110M controlled instrument.
#
# Two constraints shape this. medium QOS caps a job at 4 days, so the run must
# survive being killed: the loop checkpoints weights, optimizer moments, the AMP
# scaler and the step counter together, and reloads them on start. And a requeued
# job can land on a different node, where /ssd_scratch is empty -- so the durable
# copies of both the shards and the checkpoint live on /share1 and are staged in
# and out over ssh, since /share1 is not mounted on compute nodes.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"

LOCAL_SHARDS="$C3_SCRATCH/c3-shards"
CKPT_DIR="$C3_SCRATCH/train110m"
mkdir -p "$LOCAL_SHARDS" "$CKPT_DIR"
cd "$REPO_DIR"

echo "[stage] node=$(hostname); syncing shards from $BLOBS"
rsync -a --inplace --partial -e "ssh $SSH_OPTS" \
    "$USER@ada:$BLOBS/shards/" "$LOCAL_SHARDS/" \
    || { echo "[stage] FATAL: shard pull failed"; exit 1; }

# Shard paths in index.json are absolute and were written on another node.
"$C3_PY" - "$LOCAL_SHARDS/index.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); idx = json.loads(p.read_text())
for s in idx["shards"]:
    s["path"] = str(p.parent / pathlib.Path(s["path"]).name)
p.write_text(json.dumps(idx, indent=2))
print(f"[stage] {idx['total_tokens']/1e9:.2f}B tokens, "
      f"{idx['calibration_documents_skipped']} calibration docs excluded")
PY

# Resume: pull the last checkpoint if this node has none.
if [ ! -f "$CKPT_DIR/checkpoint.pt" ]; then
    rsync -a --inplace --partial -e "ssh $SSH_OPTS" \
        "$USER@ada:$BLOBS/train110m/checkpoint.pt" "$CKPT_DIR/" 2>/dev/null \
        && echo "[stage] resumed checkpoint from $BLOBS" \
        || echo "[stage] no prior checkpoint; starting fresh"
fi

push_ckpt() {
    # --exclude '*.tmp': the trainer writes to a temp name and renames, so a
    # push never picks up a half-written checkpoint.
    ssh $SSH_OPTS "$USER@ada" "mkdir -p '$BLOBS/train110m'" 2>/dev/null || true
    rsync -a --inplace --partial --exclude '*.tmp' -e "ssh $SSH_OPTS" \
        "$CKPT_DIR/" "$USER@ada:$BLOBS/train110m/" 2>/dev/null \
        && echo "[stage] checkpoint pushed to $BLOBS at $(date +%H:%M:%S)"
}

( while true; do sleep 1800; push_ckpt; done ) &
SYNC_PID=$!

trap 'echo "[slurm] walltime signal"; push_ckpt; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

srun --kill-on-bad-exit=1 "$(dirname "$C3_PY")/torchrun" \
     --nproc_per_node=4 --nnodes=1 \
     scripts/train_110m.py \
       --index "$LOCAL_SHARDS/index.json" \
       --out-dir "$CKPT_DIR" \
       --max-tokens "${C3_MAX_TOKENS:-6000000000}" \
       --batch-size "${C3_BATCH:-32}" \
       --grad-accum "${C3_ACCUM:-4}" \
       --wandb-project "${C3_WANDB:-c3-dllm-calibration}" &
wait $!

kill $SYNC_PID 2>/dev/null || true
push_ckpt
echo "[train] finished; final checkpoint on $BLOBS/train110m"
