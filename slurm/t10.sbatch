#!/bin/bash
#SBATCH --job-name=c3-t10
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home2/%u/c3/logs/t10-%j.out
#
# T10: parallel-sequential disagreement and the (B) proxies on real text (110M, fp32,
# 300 fit-half documents x 10 states), and the disagreement proxy calibrated on the
# probe grids (scripts/run_t10.py).
#
#     sbatch --export=ALL,C3_T10_RUN=train110m-v3 slurm/t10.sbatch
#     C3_T10_ARGS="--n-docs 5 --probe-limit 20 --probe-ks 2 --n-boot 200" \
#         sbatch --export=ALL,C3_T10_RUN=smoke,C3_T10_PROBES=ft110m-v3-visible slurm/t10.sbatch
#     sbatch --gres=gpu:2 --time=04:00:00 \
#         --export=ALL,C3_T10_RUN=c1-design,C3_T10_MODEL=llada-8b-base slurm/t10.sbatch
#
# The base checkpoint (C3_T10_CKPT_REMOTE) and each probe model's
# /share1/$USER/probe/<tag>/final.pt are pulled to node SSD through $C3_LOGIN_HOST
# (/share1 is not mounted here); the corpus is read from /home2.  An 8B model
# (C3_T10_MODEL) runs real text only, as in the C1 sweep: fp16 split over two cards
# (C3_MAX_MEMORY, devices separated by ';'), weights staged by slurm/stage_model.sh.
# Results go to artifacts/c2/ in the checkout ($C3_REPO; commit them).
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
RUN="${C3_T10_RUN:?set C3_T10_RUN, e.g. train110m-v3}"
MODEL="${C3_T10_MODEL:-local-110m}"
CK_REMOTE="${C3_T10_CKPT_REMOTE:-/share1/$USER/v2/train110m-v3/final.pt}"
PROBES="${C3_T10_PROBES-ft110m-v3-visible ft110m-v3}"     # set it empty for real text only
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
export HF_HOME="$C3_SCRATCH/hf"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
cd "$REPO_DIR"
CORPUS="${C3_T10_CORPUS:-$C3_PERSIST/corpus/calibration_docs.jsonl}"

pull() {
    rsync -a --partial-dir=.rsync-partial -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$1" "$2" \
        || { echo "[t10] FATAL: could not pull $1"; exit 1; }
}
mkdir -p "$C3_SCRATCH/t10"
ARGS=()
if [ "$MODEL" = local-110m ]; then
    PRECISION=fp32
    SOURCE="$CK_REMOTE"
    CK="$C3_SCRATCH/t10/$(basename "$(dirname "$CK_REMOTE")").pt"
    pull "$CK_REMOTE" "$CK"
    ARGS+=(--model-path "$CK" --precision "$PRECISION" --batch-size "${C3_BATCH_SIZE:-32}")
    for tag in $PROBES; do
        pull "/share1/$USER/probe/$tag/final.pt" "$C3_SCRATCH/t10/$tag.pt"
        ARGS+=(--probe "$tag=$C3_SCRATCH/t10/$tag.pt")
    done
else
    PRECISION="${C3_PRECISION:-fp16}"
    PROBES=""                                   # the probe models are 110M fine-tunes
    source "$REPO_DIR/slurm/stage_model.sh"     # MODEL -> MODEL_PATH_ARG
    SOURCE="${MODEL_PATH_ARG[*]:-hub revision}"
    MAXMEM="${C3_MAX_MEMORY:-0=9GiB;1=9GiB}"
    ARGS+=(${MODEL_PATH_ARG[@]+"${MODEL_PATH_ARG[@]}"} --precision "$PRECISION"
           --device-map auto --max-memory "${MAXMEM//;/,}" --batch-size "${C3_BATCH_SIZE:-4}")
fi
OUT="artifacts/c2/t10_${MODEL}_${PRECISION}_$RUN.json"
echo "[t10] $RUN: $MODEL $PRECISION ($SOURCE), probes [${PROBES}] on $(hostname -s), commit $C3_GIT_COMMIT"

"$C3_PY" scripts/run_t10.py --model "$MODEL" --corpus "$CORPUS" --run "$RUN" \
    --out-dir artifacts/c2 "${ARGS[@]}" ${C3_T10_ARGS:-}

mkdir -p logs
echo "$(date +%F_%T) t10=${SLURM_JOB_ID:-?} $RUN ($MODEL $PRECISION, $SOURCE; probes [${PROBES}]; args [${C3_T10_ARGS:-}]) -> $OUT; commit $C3_GIT_COMMIT" >> logs/t3_jobs.txt
echo "[t10] done: $OUT -- commit it"
