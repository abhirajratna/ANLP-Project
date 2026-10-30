#!/bin/bash
#SBATCH --job-name=c3-fit-c3a
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --cpus-per-task=11
#SBATCH --mem=32000M
#SBATCH --time=06:00:00
#SBATCH --output=/home2/%u/c3/logs/fit-c3a-%j.out
#
# T11 (plan to the final, Parth 2): fit and freeze the C3a calibrators on the fit half of
# the LLaDA-8B-Instruct task traces, one per task, with scripts/fit_c3a_calibrator.py.
#
#   sbatch slurm/fit_c3a.sbatch                        # gsm8k and mbpp
#   sbatch --export=ALL,C3_TASKS=gsm8k slurm/fit_c3a.sbatch
#
# The traces are abhiraj's (/share1/abhiraj.ratna/traces/<run>/, readable once he grants
# access).  /share1 is not mounted on compute nodes, so each run directory (parts plus
# its zero-copy <run>.dataset.json, whose parts resolve relative to it) is copied to
# node-local SSD first.  Only fit-half rows are read (the script filters while reading).
# The calibrators land in $REPO/artifacts/c3/llada-8b-instruct_fp16_c3a-<task>.json, to be
# committed by PR.  Extra script arguments: C3_FIT_ARGS (e.g. --features median_ell).
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
TRACES="${C3_TRACES_REMOTE:-/share1/abhiraj.ratna/traces}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
declare -A RUN=([gsm8k]=c1-llada-instruct-gsm8k [mbpp]=c1-llada-instruct-mbpp-v2)

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-11}"
cd "$REPO_DIR"
WORK="$C3_SCRATCH/fit-c3a"
mkdir -p "$WORK" artifacts/c3 logs
echo "[fit-c3a] node=$(hostname) commit=$C3_GIT_COMMIT tasks='${C3_TASKS:-gsm8k mbpp}'"

for task in ${C3_TASKS:-gsm8k mbpp}; do
    run="${RUN[$task]:?no trace run for task $task}"
    rsync -a --exclude '*.tmp' -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$TRACES/$run/" "$WORK/$run/" \
        || { echo "[fit-c3a] FATAL: cannot read $TRACES/$run (read grant?)"; exit 1; }
    out="artifacts/c3/llada-8b-instruct_fp16_c3a-$task.json"
    # shellcheck disable=SC2086
    "$C3_PY" scripts/fit_c3a_calibrator.py --traces "$WORK/$run/$run.dataset.json" \
        --manifest "artifacts/split_manifest_$task.json" ${C3_FIT_ARGS:-} --out "$out"
done
echo "$(date +%F_%T) fit-c3a=${SLURM_JOB_ID:-?} commit=$C3_GIT_COMMIT" >> logs/c3_jobs.txt
