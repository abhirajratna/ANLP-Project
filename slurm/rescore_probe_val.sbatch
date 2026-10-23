#!/bin/bash
#SBATCH --job-name=c3-rescore
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home2/%u/c3/logs/rescore-%j.out
#
# T9: rescore the fine-tunes' validation NELBO with the rho marker visible
# (scripts/rescore_probe_val.py), so ft110m-v3 and ft110m-v3-visible compare.
#
#     sbatch slurm/rescore_probe_val.sbatch            # C3_RESCORE_RUNS="ft110m-v3 ft110m-v3-visible"
#
# Shards are rebuilt on the node from artifacts/probe/spec.json, as in
# finetune_probe.sbatch; checkpoints are pulled from /share1/$USER/probe/<run>/.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
RUNS="${C3_RESCORE_RUNS:-ft110m-v3 ft110m-v3-visible}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
cd "$REPO_DIR"
SHARDS="$C3_SCRATCH/probe-shards"
rm -rf "$SHARDS"
"$C3_PY" scripts/build_probe_shards.py --out-dir "$SHARDS"
ARGS=()
for run in $RUNS; do
    ck="$C3_SCRATCH/rescore/$run.pt"
    mkdir -p "$(dirname "$ck")"
    rsync -a -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:/share1/$USER/probe/$run/final.pt" "$ck" \
        || { echo "[rescore] FATAL: could not pull $run"; exit 1; }
    ARGS+=(--run "$run=$ck")
done
"$C3_PY" scripts/rescore_probe_val.py --index "$SHARDS/index.json" "${ARGS[@]}"
echo "$(date +%F_%T) rescore=${SLURM_JOB_ID:-?} $RUNS -> artifacts/probe/results/val_rescored.json" >> logs/t3_jobs.txt
