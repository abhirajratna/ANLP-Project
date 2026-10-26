#!/bin/bash
#SBATCH --job-name=c3-t12
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:2
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069,gnode076
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --array=0-0
#SBATCH --output=/home2/%u/c3/logs/t12-%A_%a.out
#
# T12 on Ada (plan to the final, Parth 3-5): LLaDA-8B-Instruct, fp16 over two 2080 Ti,
# scripts/run_t12.py with one policy grid; array task I runs shard I of C3_NUM_SHARDS.
# Ada is the overflow for the Jarvis sweep -- whole tasks only, so no comparison mixes
# precisions (and the C3a calibrator's traces are fp16 too).
#
#   # the driver smoke run: 5 fit-half GSM8K problems, top-k k=8
#   sbatch --export=ALL,C3_TASK=gsm8k,C3_HALF=fit,C3_LIMIT=5,C3_TAG=smoke,C3_CONFIGS=prereg/t12_smoke.json slurm/t12.sbatch
#   # the sanity check at Fast-dLLM's settings (lm-eval 5-shot), 200 fit-half problems, 4 shards
#   sbatch --array=0-3 --export=ALL,C3_TASK=gsm8k,C3_HALF=fit,C3_LIMIT=200,C3_PROMPT=lmeval,C3_NUM_SHARDS=4,C3_TAG=sanity,C3_CONFIGS=<grid> slurm/t12.sbatch
#
# Knobs: C3_TASK (gsm8k|mbpp), C3_HALF (fit|report|all), C3_TAG (output name), C3_CONFIGS
# (grid JSON, relative to the repo), C3_LIMIT, C3_PROMPT (chat|lmeval), C3_FEWSHOT (5),
# C3_CALIBRATOR (frozen C3a calibrator), C3_NUM_SHARDS, C3_EXTRA_ARGS.  Records go
# straight to /home2/$USER/c3-runs/t12/$C3_TAG (small, fsync'd per problem), so a
# requeued task resumes where it stopped.  The report half is the sweep's, not a
# pre-registered endpoint (plan to the final: "T12 is not a pre-registered endpoint").
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
TASK="${C3_TASK:?set C3_TASK (gsm8k or mbpp)}"
HALF="${C3_HALF:?set C3_HALF (fit, report or all)}"
TAG="${C3_TAG:?set C3_TAG, the name of this run}"
CONFIGS="${C3_CONFIGS:?set C3_CONFIGS, the policy grid JSON}"
NUM_SHARDS="${C3_NUM_SHARDS:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
MODEL=llada-8b-instruct PRESET=""
[ "$SHARD" -lt "$NUM_SHARDS" ] || { echo "[t12] FATAL: task $SHARD >= C3_NUM_SHARDS=$NUM_SHARDS"; exit 1; }

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
OUT="$PERSIST/t12/$TAG"
TASKS="$PERSIST/corpus/tasks"
mkdir -p "$HF_HOME" "$OUT" "$TASKS" logs
cd "$REPO_DIR"
echo "[t12] node=$(hostname) task=$TASK half=$HALF tag=$TAG shard=$SHARD/$NUM_SHARDS commit=$C3_GIT_COMMIT"

# The task file at the pinned hub revision (c3.data.tasks.TASKS), once per account.
if [ ! -s "$TASKS/${TASK}_test.jsonl" ]; then
    "$C3_PY" scripts/prepare_tasks.py --tasks "$TASK" --out-dir "$TASKS"
fi
source "$REPO_DIR/slurm/stage_model.sh"

trap 'echo "[slurm] walltime signal"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

ARGS=(--model "$MODEL" --precision fp16 --device cuda --device-map auto --max-memory "0=9GiB,1=9GiB"
      --task "$TASK" --task-jsonl "$TASKS/${TASK}_test.jsonl" --half "$HALF"
      --manifest "artifacts/split_manifest_${TASK}.json" --configs "$CONFIGS"
      --prompt "${C3_PROMPT:-chat}" --fewshot "${C3_FEWSHOT:-5}"
      --unigram-docs "$PERSIST/corpus/reserve_docs.jsonl"
      --shard "$SHARD" --num-shards "$NUM_SHARDS" --label "ada-$TAG" --out-dir "$OUT")
[ -n "${C3_LIMIT:-}" ] && ARGS+=(--limit "$C3_LIMIT")
[ -n "${C3_CALIBRATOR:-}" ] && ARGS+=(--calibrator "$C3_CALIBRATOR")
# shellcheck disable=SC2086
"$C3_PY" scripts/run_t12.py "${ARGS[@]}" "${MODEL_PATH_ARG[@]}" ${C3_EXTRA_ARGS:-} &
RC=0
wait $! || RC=$?
echo "[t12] exited with status $RC -> $OUT"
exit $RC
