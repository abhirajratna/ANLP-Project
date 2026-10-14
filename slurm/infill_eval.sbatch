#!/bin/bash
#SBATCH --job-name=c3-infill
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069,gnode076
#SBATCH --cpus-per-task=6
#SBATCH --mem=18000M
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-3
#SBATCH --output=/home2/%u/c3/logs/infill-%A_%a.out
#
# FineWeb infilling with a knob grid (scripts/eval_infill.py), one shard per array task:
#
#   sbatch --export=ALL,C3_MODE=dev,C3_HALF=dev-eval,C3_TAG=sweep1,C3_GRID=prereg/c3set_policies_dev.json,C3_CAL_DIR=/home2/$USER/c3-runs/c3set/dev/cal/traces slurm/infill_eval.sbatch
#
# Records go straight to /home2/$USER/c3-runs/c3set/$C3_MODE/infill/$C3_TAG (small,
# appended per batch, so a requeued task resumes); step logs (C3_EXTRA_ARGS=
# --record-steps) go to node SSD and are pushed to /share1/$USER/c3set/$MODE/steps/$TAG.
# C3_HALF=report needs C3_REPORT_HALF_OK=1.  Knobs: C3_NUM_SHARDS (= array size),
# C3_BATCH_SIZE (32), C3_LIMIT, C3_ONLY (policy names), C3_MODEL_REMOTE.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
MODE="${C3_MODE:?set C3_MODE=dev or report}"
HALF="${C3_HALF:?set C3_HALF (dev-fit, dev-eval, report)}"
TAG="${C3_TAG:?set C3_TAG, the name of this sweep}"
GRID="${C3_GRID:?set C3_GRID, the knob grid JSON}"
NUM_SHARDS="${C3_NUM_SHARDS:-4}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
export C3_MODEL_REMOTE="${C3_MODEL_REMOTE:-/share1/$USER/train110m-v4/final.pt}"
MODEL=local-110m PRESET=""

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
export HF_HOME="$C3_SCRATCH/hf"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO_DIR"
OUT="$PERSIST/c3set/$MODE/infill/$TAG"
STEPS_DIR="$C3_SCRATCH/c3set-steps/$MODE/$TAG"
mkdir -p "$OUT" "$STEPS_DIR" "$HF_HOME"
echo "[infill] node=$(hostname) mode=$MODE half=$HALF tag=$TAG shard=$SHARD/$NUM_SHARDS commit=$C3_GIT_COMMIT"
source "$REPO_DIR/slurm/stage_model.sh"

push_steps() {
    if ls "$STEPS_DIR"/*.parquet >/dev/null 2>&1; then
        ssh $SSH_OPTS "$USER@$C3_LOGIN_HOST" "mkdir -p '$BLOBS/c3set/$MODE/steps/$TAG'" 2>/dev/null || true
        rsync -a --exclude '*.tmp' -e "ssh $SSH_OPTS" "$STEPS_DIR/" \
            "$USER@$C3_LOGIN_HOST:$BLOBS/c3set/$MODE/steps/$TAG/" 2>/dev/null \
            && echo "[infill] step logs pushed at $(date +%H:%M:%S)" \
            || echo "[infill] WARNING: step-log push failed"
    fi
    return 0
}
trap push_steps EXIT
trap 'echo "[slurm] walltime signal"; push_steps; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

CAL=()
if [ -n "${C3_CAL_DIR:-}" ]; then
    [ -r "$C3_CAL_DIR/tok.json" ] && CAL+=(--calibrator "$C3_CAL_DIR/tok.json")
    [ -r "$C3_CAL_DIR/set.json" ] && CAL+=(--set-calibrator "$C3_CAL_DIR/set.json")
fi
ARGS=(--model local-110m --device cuda --docs "$PERSIST/corpus/calibration_docs.jsonl"
      --manifest artifacts/split_manifest.json --half "$HALF" --configs "$GRID"
      --unigram-docs "$PERSIST/corpus/reserve_docs.jsonl" --shard "$SHARD" --num-shards "$NUM_SHARDS"
      --batch-size "${C3_BATCH_SIZE:-32}" --out-dir "$OUT" --steps-dir "$STEPS_DIR" --label "$MODE-$TAG")
[ -n "${C3_LIMIT:-}" ] && ARGS+=(--limit "$C3_LIMIT")
[ -n "${C3_ONLY:-}" ] && { read -r -a only <<< "$C3_ONLY"; ARGS+=(--only "${only[@]}"); }
# shellcheck disable=SC2086
"$C3_PY" scripts/eval_infill.py "${ARGS[@]}" "${CAL[@]}" "${MODEL_PATH_ARG[@]}" ${C3_EXTRA_ARGS:-} &
RC=0
wait $! || RC=$?
echo "[infill] exited with status $RC"
exit $RC
