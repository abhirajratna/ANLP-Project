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
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --array=0-0
#SBATCH --output=/home2/%u/c3/logs/traces-%A_%a.out
#
# One model arm of the audit as a SLURM array: task I collects shard I of
# C3_NUM_SHARDS (documents with doc_id % N == I). Pass --array=0-(N-1) with the
# same N. Examples (see slurm/CLUSTER.md for the full list):
#
#   # T2 pilot, 20 documents, fp16 split over two cards
#   sbatch --gres=gpu:2 --export="ALL,C3_PRESET=t2-pilot,C3_PRECISION=fp16,C3_LIMIT=20,C3_DEVICE_MAP=auto,C3_MAX_MEMORY=0=9GiB;1=9GiB,C3_RUN=t2pilot-llada-base-fp16" slurm/collect_traces.sbatch
#
#   # a sweep in 4 shards
#   sbatch --array=0-3 --export="ALL,C3_MODEL=local-110m,C3_PRECISION=fp32,C3_MODEL_REMOTE=/share1/$USER/train110m/final.pt,C3_NUM_SHARDS=4,C3_RUN=main-110m" slurm/collect_traces.sbatch
#
# C3_MAX_MEMORY separates devices with ';' because --export splits on ',', so
# quote the whole --export value. Other knobs: C3_MODEL_PATH (a path already on
# the node), C3_DOCS, C3_TASK (gsm8k|mbpp), C3_REPEAT (floor runs, 1 shard only),
# C3_BATCH_SIZE, C3_EXTRA_ARGS (passed through to scripts/collect_traces.py).
#
# Parts are written to node-local SSD and pushed to /share1/$USER/traces/$RUN
# every 30 minutes and on exit; a requeued or resubmitted task pulls its shard's
# parts back and resumes, so a killed job loses at most one part.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"

PRESET="${C3_PRESET:-}"
MODEL="${C3_MODEL:-}"
PRECISION="${C3_PRECISION:?set C3_PRECISION (fp16, fp32, 4bit, ...): see slurm/CLUSTER.md}"
NUM_SHARDS="${C3_NUM_SHARDS:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
REPEAT="${C3_REPEAT:-1}"
RUN="${C3_RUN:?set C3_RUN, the name of this collection run (output directory)}"
DOCS="${C3_DOCS:-$PERSIST/corpus/calibration_docs.jsonl}"

if [ "$SHARD" -ge "$NUM_SHARDS" ]; then
    echo "[stage] FATAL: array task $SHARD >= C3_NUM_SHARDS=$NUM_SHARDS"; exit 1
fi

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
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT_DIR="$C3_SCRATCH/traces/$RUN"
mkdir -p "$HF_HOME" "$OUT_DIR" logs
cd "$REPO_DIR"
echo "[stage] node=$(hostname) run=$RUN shard=$SHARD/$NUM_SHARDS commit=$C3_GIT_COMMIT"

# ---- model weights ---------------------------------------------------------
# LLaDA-8B sits on /share1, but which variant it holds was never recorded, so it
# is used only when its tokenizer config says it is the variant requested
# (a chat template means Instruct). Otherwise the pinned revision is downloaded
# from the hub into HF_HOME on node-local SSD, never /home2.
SHARED_LLADA="/share1/ishita/llada_model"
MODEL_PATH_ARG=()
if [ -n "${C3_MODEL_REMOTE:-}" ]; then
    LOCAL_COPY="$C3_SCRATCH/models/$(basename "$C3_MODEL_REMOTE")"
    mkdir -p "$(dirname "$LOCAL_COPY")"
    echo "[stage] pulling $C3_MODEL_REMOTE"
    rsync -a --partial-dir=.rsync-partial -e "ssh $SSH_OPTS" \
        "$USER@ada:$C3_MODEL_REMOTE" "$LOCAL_COPY" \
        || { echo "[stage] FATAL: model transfer failed"; exit 1; }
    MODEL_PATH_ARG=(--model-path "$LOCAL_COPY")
elif [ -n "${C3_MODEL_PATH:-}" ]; then
    MODEL_PATH_ARG=(--model-path "$C3_MODEL_PATH")
else
    WANT="$MODEL"
    [ -z "$WANT" ] && [ "$PRESET" = "t2-pilot" ] && WANT="llada-8b-base"
    case "$WANT" in
        llada-8b|llada-8b-instruct) WANT_VARIANT=instruct ;;
        llada-8b-base) WANT_VARIANT=base ;;
        *) WANT_VARIANT="" ;;
    esac
    if [ -n "$WANT_VARIANT" ]; then
        if ssh $SSH_OPTS "$USER@ada" "grep -q chat_template $SHARED_LLADA/tokenizer_config.json" 2>/dev/null; then
            SHARED_VARIANT=instruct
        else
            SHARED_VARIANT=base
        fi
        echo "[stage] $SHARED_LLADA holds LLaDA-8B-$SHARED_VARIANT; requested $WANT_VARIANT"
        if [ "$SHARED_VARIANT" = "$WANT_VARIANT" ]; then
            LOCAL_LLADA="$C3_SCRATCH/llada_model_$SHARED_VARIANT"
            if [ ! -r "$LOCAL_LLADA/config.json" ]; then
                mkdir -p "$LOCAL_LLADA"
                rsync -a --inplace --partial -e "ssh $SSH_OPTS" \
                    "$USER@ada:$SHARED_LLADA/" "$LOCAL_LLADA/" \
                    || { echo "[stage] FATAL: LLaDA transfer failed"; exit 1; }
            fi
            MODEL_PATH_ARG=(--model-path "$LOCAL_LLADA")
        else
            echo "[stage] downloading the pinned hub revision into $HF_HOME"
        fi
    fi
fi

# ---- resume ---------------------------------------------------------------
if [ "$NUM_SHARDS" -gt 1 ] || [ "$REPEAT" -eq 1 ]; then
    PATTERN="$RUN.shard${SHARD}of${NUM_SHARDS}.*"
else
    PATTERN="$RUN.*"
fi
rsync -a --exclude '*.tmp' --include "$PATTERN" --exclude '*' -e "ssh $SSH_OPTS" \
    "$USER@ada:$BLOBS/traces/$RUN/" "$OUT_DIR/" 2>/dev/null \
    && echo "[stage] pulled durable parts: $(ls "$OUT_DIR" | grep -c "\.p[0-9]*\.parquet$" || true)" \
    || echo "[stage] no durable parts yet"

push_out() {
    ssh $SSH_OPTS "$USER@ada" "mkdir -p '$BLOBS/traces/$RUN'" 2>/dev/null || true
    # Only this task's files, never *.tmp: a part becomes visible when complete.
    rsync -a --exclude '*.tmp' --include "$PATTERN" --include "$RUN.parquet" \
        --include "$RUN.meta.json" --exclude '*' -e "ssh $SSH_OPTS" \
        "$OUT_DIR/" "$USER@ada:$BLOBS/traces/$RUN/" 2>/dev/null \
        && echo "[stage] pushed to $BLOBS/traces/$RUN at $(date +%H:%M:%S)"
}
( while true; do sleep 1800; push_out; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; wait $SYNC_PID 2>/dev/null || true; push_out' EXIT
trap 'echo "[slurm] walltime signal"; kill $SYNC_PID 2>/dev/null || true; push_out; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

# ---- collect ----------------------------------------------------------------
ARGS=(--precision "$PRECISION" --device cuda --batch-size "${C3_BATCH_SIZE:-1}")
[ -n "$PRESET" ] && ARGS+=(--preset "$PRESET")
[ -n "$MODEL" ] && ARGS+=(--model "$MODEL")
[ -n "${C3_LIMIT:-}" ] && ARGS+=(--limit "$C3_LIMIT")
[ -n "${C3_DEVICE_MAP:-}" ] && ARGS+=(--device-map "$C3_DEVICE_MAP")
[ -n "${C3_MAX_MEMORY:-}" ] && ARGS+=(--max-memory "${C3_MAX_MEMORY//;/,}")
if [ -n "${C3_TASK:-}" ]; then
    ARGS+=(--task "$C3_TASK")
else
    ARGS+=(--docs "$DOCS")
fi
if [ "$REPEAT" -gt 1 ]; then
    [ "$NUM_SHARDS" -eq 1 ] || { echo "[stage] FATAL: C3_REPEAT needs C3_NUM_SHARDS=1"; exit 1; }
    ARGS+=(--repeat "$REPEAT" --out "$OUT_DIR/$RUN.parquet")
else
    ARGS+=(--shard "$SHARD" --num-shards "$NUM_SHARDS" --resume --out "$OUT_DIR/$RUN")
fi
# shellcheck disable=SC2086
"$C3_PY" scripts/collect_traces.py "${ARGS[@]}" "${MODEL_PATH_ARG[@]}" ${C3_EXTRA_ARGS:-} &
RC=0
wait $! || RC=$?
echo "[collect] exited with status $RC; EXIT trap pushes the parts"
exit $RC
