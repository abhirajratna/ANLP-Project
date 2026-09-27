#!/bin/bash
#SBATCH --job-name=c3-train110m
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --gres=gpu:2
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# medium QOS caps this account's *aggregate* usage at cpu=40, gres/gpu=4,
# mem=125000M. A 32-CPU request is fine on its own but is rejected outright --
# "CPU count per node can not be satisfied" -- whenever another of this
# account's jobs already holds CPUs. 16 is ample: the loader memory-maps shards
# and does no tokenisation.
#
# 2 GPUs by default: the 4-GPU aggregate cap is shared with trace collection,
# and 10-20B tokens on 2 x 2080 Ti (~1.3-2.6 days at the measured 74.7k tok/s)
# still fits one 4-day job.  For 4 GPUs submit with
#     sbatch --gres=gpu:4 --export=ALL,C3_NGPU=4 slurm/train_110m.sbatch
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home2/%u/c3/logs/train110m-%j.out
#
# The 110M controlled instrument.
#
# Two constraints shape this. medium QOS caps a job at 4 days, so the run must
# survive being killed: the loop checkpoints weights, EMA, optimizer moments,
# the AMP scaler and the step counter together, and reloads them on start. And
# a requeued job can land on a different node, where /ssd_scratch holds either
# nothing or an *older* checkpoint from an earlier visit -- so the durable copies
# of the shards and the checkpoint live on /share1, are staged over ssh (/share1
# is not mounted on compute nodes), and the newer of the local and durable copy
# is chosen by the step recorded in the checkpoint's JSON sidecar.
#
# Each run has its own directory, C3_RUN_NAME (default train110m-v2).  The
# trainer refuses to resume a checkpoint written with a different configuration,
# or one from the proposal-era trainer, so a new run cannot silently continue
# trial weights.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
RUN="${C3_RUN_NAME:-train110m-v2}"
NGPU="${C3_NGPU:-2}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"

# The 2080 Ti has ~10.6 GiB usable and this run peaks near 9.4 GiB, so
# allocator fragmentation is the difference between fitting and an OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOCAL_SHARDS="$C3_SCRATCH/c3-shards"
CKPT_DIR="$C3_SCRATCH/$RUN"
mkdir -p "$LOCAL_SHARDS" "$CKPT_DIR"
cd "$REPO_DIR"

echo "[stage] node=$(hostname) run=$RUN gpus=$NGPU commit=$C3_GIT_COMMIT"
rsync -a --partial -e "ssh $SSH_OPTS" \
    "$USER@ada:$BLOBS/shards/" "$LOCAL_SHARDS/" \
    || { echo "[stage] FATAL: shard pull failed"; exit 1; }

# Shard paths in index.json are absolute and were written on another node.
"$C3_PY" - "$LOCAL_SHARDS/index.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); idx = json.loads(p.read_text())
for key in ("shards", "val_shards"):
    for s in idx.get(key, []):
        for f in ("path", "offsets"):
            if f in s:
                s[f] = str(p.parent / pathlib.Path(s[f]).name)
p.write_text(json.dumps(idx, indent=2))
ex = idx.get("exclusion", {})
print(f"[stage] {idx['total_tokens']/1e9:.2f}B tokens, {idx.get('val_documents', 0)} val docs, "
      f"{idx['calibration_documents_skipped']} held-out docs excluded "
      f"(by key: {ex.get('skipped_by_key')})")
if idx.get("format") != "doc-offsets-v2":
    sys.exit("[stage] FATAL: shards were built without document boundaries; rebuild them")
PY

# Resume from whichever copy of the checkpoint is newer.
step_of() { "$C3_PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" "$1" 2>/dev/null || echo -1; }
REMOTE_SIDE="$C3_SCRATCH/.remote-$RUN.json"
rm -f "$REMOTE_SIDE"
rsync -a -e "ssh $SSH_OPTS" "$USER@ada:$BLOBS/$RUN/checkpoint.json" "$REMOTE_SIDE" 2>/dev/null || true
LOCAL_STEP=$( [ -f "$CKPT_DIR/checkpoint.json" ] && step_of "$CKPT_DIR/checkpoint.json" || echo -1 )
REMOTE_STEP=$( [ -f "$REMOTE_SIDE" ] && step_of "$REMOTE_SIDE" || echo -1 )
echo "[stage] checkpoint steps: local=$LOCAL_STEP durable=$REMOTE_STEP"
if [ "$REMOTE_STEP" -gt "$LOCAL_STEP" ]; then
    rm -f "$CKPT_DIR/checkpoint.pt" "$CKPT_DIR/checkpoint.json"
    rsync -a --partial-dir=.rsync-partial -e "ssh $SSH_OPTS" \
        "$USER@ada:$BLOBS/$RUN/checkpoint.pt" "$USER@ada:$BLOBS/$RUN/checkpoint.json" \
        "$USER@ada:$BLOBS/$RUN/history.jsonl" "$CKPT_DIR/" 2>/dev/null \
        && echo "[stage] resuming from durable checkpoint at step $REMOTE_STEP" \
        || echo "[stage] WARNING: durable checkpoint pull failed"
elif [ "$LOCAL_STEP" -ge 0 ]; then
    echo "[stage] resuming from local checkpoint at step $LOCAL_STEP"
else
    echo "[stage] no prior checkpoint; starting fresh"
fi

push_ckpt() {
    # --partial-dir keeps an interrupted transfer out of the destination file, so
    # the durable copy is always a complete checkpoint.  --exclude '*.tmp': the
    # trainer writes to a temp name and renames.  Never push an older checkpoint
    # over a newer durable one.
    local lstep rstep
    lstep=$( [ -f "$CKPT_DIR/checkpoint.json" ] && step_of "$CKPT_DIR/checkpoint.json" || echo -1 )
    rsync -a -e "ssh $SSH_OPTS" "$USER@ada:$BLOBS/$RUN/checkpoint.json" "$REMOTE_SIDE" 2>/dev/null || true
    rstep=$( [ -f "$REMOTE_SIDE" ] && step_of "$REMOTE_SIDE" || echo -1 )
    if [ "$lstep" -lt "$rstep" ]; then
        echo "[stage] not pushing: local step $lstep < durable step $rstep"; return 0
    fi
    ssh $SSH_OPTS "$USER@ada" "mkdir -p '$BLOBS/$RUN'" 2>/dev/null || true
    rsync -a --partial-dir=.rsync-partial --exclude '*.tmp' -e "ssh $SSH_OPTS" \
        "$CKPT_DIR/" "$USER@ada:$BLOBS/$RUN/" 2>/dev/null \
        && echo "[stage] checkpoint (step $lstep) pushed to $BLOBS/$RUN at $(date +%H:%M:%S)"
}

( while true; do sleep 1800; push_ckpt; done ) &
SYNC_PID=$!

# Push on EXIT, not just on the happy path. With `set -e` a crashing srun skipped
# the final push entirely, stranding the newest checkpoint on a node's local SSD
# where nothing else can reach it.
trap 'kill $SYNC_PID 2>/dev/null || true; wait $SYNC_PID 2>/dev/null || true; push_ckpt' EXIT
trap 'echo "[slurm] walltime signal"; kill $SYNC_PID 2>/dev/null || true; push_ckpt; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

srun --kill-on-bad-exit=1 "$(dirname "$C3_PY")/torchrun" \
     --nproc_per_node="$NGPU" --nnodes=1 \
     scripts/train_110m.py \
       --index "$LOCAL_SHARDS/index.json" \
       --out-dir "$CKPT_DIR" \
       --max-tokens "${C3_MAX_TOKENS:-10000000000}" \
       --batch-size "${C3_BATCH:-32}" \
       --grad-accum "${C3_ACCUM:-4}" \
       --window-mode "${C3_WINDOWS:-chunks}" \
       --n-gpus "$NGPU" \
       --run-name "$RUN" \
       --wandb-project "${C3_WANDB:-c3-dllm-calibration}" &
TRAIN_RC=0
wait $! || TRAIN_RC=$?
echo "[train] torchrun exited with status $TRAIN_RC; EXIT trap will push the checkpoint"
exit $TRAIN_RC
