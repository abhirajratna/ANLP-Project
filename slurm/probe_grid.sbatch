#!/bin/bash
#SBATCH --job-name=c3-probegrid
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069,gnode076
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M
#SBATCH --time=08:00:00
#SBATCH --output=/home2/%u/c3/logs/probegrid-%j.out
#
# T9: the rho x k x selection-rule grid on a 110M checkpoint (scripts/run_probe.py),
# then the real-model identity test (tests/test_probe.py) on the same checkpoint.
#
#     sbatch --export=ALL,C3_PROBE_CKPT_REMOTE=/share1/$USER/v2/train110m-v3/final.pt,C3_PROBE_TAG=untuned-v2 \
#         slurm/probe_grid.sbatch
#
# The checkpoint is pulled from /share1 through $C3_LOGIN_HOST; results go to
# artifacts/probe/results/ in the repo checkout (commit them).
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
CK_REMOTE="${C3_PROBE_CKPT_REMOTE:?set C3_PROBE_CKPT_REMOTE to a final.pt on /share1}"
TAG="${C3_PROBE_TAG:?set C3_PROBE_TAG, e.g. untuned-v2}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
export HF_HOME="$C3_SCRATCH/hf"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
cd "$REPO_DIR"

CK="$C3_SCRATCH/probe-$TAG.pt"
rsync -a --partial-dir=.rsync-partial -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$CK_REMOTE" "$CK" \
    || { echo "[probe] FATAL: could not pull $CK_REMOTE"; exit 1; }
echo "[probe] $TAG: $CK_REMOTE ($(du -h "$CK" | cut -f1)) on $(hostname -s), commit $C3_GIT_COMMIT"

"$C3_PY" scripts/run_probe.py --model local-110m --model-path "$CK" --tag "$TAG" \
    --out-dir artifacts/probe/results ${C3_PROBE_ARGS:-}

if [ "${C3_PROBE_TEST:-1}" = 1 ]; then
    C3_PROBE_CKPT="$CK" "$C3_PY" -m pytest tests/test_probe.py -q -p no:cacheprovider \
        -k "real_110m or one_token" 2>&1 | tail -5
fi
echo "[probe] done: artifacts/probe/results/grid_$TAG.json -- commit it"
