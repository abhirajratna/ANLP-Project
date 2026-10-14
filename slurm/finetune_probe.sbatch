#!/bin/bash
#SBATCH --job-name=c3-ftprobe
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --gres=gpu:2
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home2/%u/c3/logs/ftprobe-%j.out
#
# T9: fine-tune a 110M checkpoint on the 200k probe training instances, then run
# the probe grid on the result (the main T9 numbers).
#
#     sbatch --export=ALL,C3_BASE_REMOTE=/share1/$USER/v2/train110m-v3/final.pt,C3_FT_RUN=ft110m-v3 \
#         slurm/finetune_probe.sbatch                      # add --gres=gpu:4 and C3_NGPU=4 for 4 GPUs
#
# The shards are rebuilt on the node from artifacts/probe/spec.json (30 s; checked
# against the manifest's text digest), the base model is pulled from /share1, and
# training is scripts/train_110m.py with --init-from: same loss, one instance per
# window (--window-mode prefix).  Output: /share1/$USER/probe/$C3_FT_RUN/ (final.pt,
# history.jsonl, train_summary.json) and artifacts/probe/results/grid_$C3_FT_RUN.json.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
BASE_REMOTE="${C3_BASE_REMOTE:?set C3_BASE_REMOTE to the base final.pt on /share1}"
RUN="${C3_FT_RUN:?set C3_FT_RUN, e.g. ft110m-v3}"
NGPU="${C3_NGPU:-2}"
DEST="${C3_FT_DEST:-/share1/$USER/probe/$RUN}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
export HF_HOME="$C3_SCRATCH/hf"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -z "${WANDB_MODE:-}" ]; then
    grep -qs api.wandb.ai "$HOME/.netrc" && export WANDB_MODE=online || export WANDB_MODE=offline
fi
cd "$REPO_DIR"
"$C3_PY" scripts/probe_gpus.py "$NGPU" || { echo "[ft] FATAL: GPUs on $(hostname) unusable"; exit 1; }

SHARDS="$C3_SCRATCH/probe-shards"
rm -rf "$SHARDS"
"$C3_PY" scripts/build_probe_shards.py --out-dir "$SHARDS"

# C3_FT_VISIBLE=1: never mask the rho marker during fine-tuning (it is always observed
# at evaluation); its token count comes from the shard index.
VISIBLE=0
if [ "${C3_FT_VISIBLE:-0}" = 1 ]; then
    VISIBLE=$("$C3_PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['visible_prefix_tokens'])" "$SHARDS/index.json")
fi

BASE="$C3_SCRATCH/probe-base-$RUN.pt"
rsync -a --partial-dir=.rsync-partial -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$BASE_REMOTE" "$BASE" \
    || { echo "[ft] FATAL: could not pull $BASE_REMOTE"; exit 1; }
OUT="$C3_SCRATCH/$RUN"
rm -rf "$OUT"                       # a fresh fine-tune each time; it takes about an hour
echo "[ft] $RUN on $(hostname -s), $NGPU GPUs, base $BASE_REMOTE, visible prefix $VISIBLE, commit $C3_GIT_COMMIT"

srun --kill-on-bad-exit=1 "$(dirname "$C3_PY")/torchrun" --nproc_per_node="$NGPU" --nnodes=1 \
    scripts/train_110m.py \
      --index "$SHARDS/index.json" --out-dir "$OUT" --init-from "$BASE" \
      --window-mode prefix --visible-prefix "$VISIBLE" --max-tokens "${C3_FT_TOKENS:-100000000}" \
      --batch-size "${C3_BATCH:-16}" --grad-accum "${C3_ACCUM:-2}" \
      --lr "${C3_FT_LR:-1e-4}" --warmup-steps "${C3_FT_WARMUP:-300}" --ema-decay 0.999 \
      --eval-every 500 --ckpt-every 1000 --eval-batches 16 --n-gpus "$NGPU" \
      --run-name "$RUN" --wandb-project "${C3_WANDB:-c3-dllm-calibration}" \
      --wandb-entity "${C3_WANDB_ENTITY:-ANLP-project}"

ssh $SSH_OPTS "$USER@$C3_LOGIN_HOST" "mkdir -p '$DEST'"
rsync -a -e "ssh $SSH_OPTS" "$OUT/final.pt" "$OUT/history.jsonl" "$OUT/train_summary.json" \
    "$USER@$C3_LOGIN_HOST:$DEST/" || { echo "[ft] FATAL: could not push to $DEST"; exit 1; }
mkdir -p artifacts/probe/results
cp "$OUT/history.jsonl" "artifacts/probe/results/history_$RUN.jsonl"
echo "[ft] pushed final.pt to $DEST"
echo "$(date +%F_%T) ftprobe=${SLURM_JOB_ID:-?} $RUN from $BASE_REMOTE -> $DEST" >> logs/t3_jobs.txt

if [ "${C3_FT_GRID:-1}" = 1 ]; then
    "$C3_PY" scripts/run_probe.py --model local-110m --model-path "$OUT/final.pt" \
        --tag "$RUN" --out-dir artifacts/probe/results ${C3_PROBE_ARGS:-}
    "$C3_PY" scripts/probe_rho_check.py --model-path "$OUT/final.pt" --tag "$RUN"
fi
echo "[ft] done -- commit artifacts/probe/results/*_$RUN.*"
