#!/bin/bash
#SBATCH --job-name=c3-merge
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home2/%u/c3/logs/merge-%j.out
#
# Merge one collection run zero-copy and validate it (C1 sweep, step 6):
#
#   sbatch --export="ALL,C3_RUN=c1-llada-base,C3_GPUS=2,C3_CORPUS=/home2/$USER/c3-runs/corpus/calibration_docs.jsonl" slurm/merge_validate.sbatch
#
# C3_MERGE_ARGS=--allow-mixed-commits, for a run whose shards were collected at
# different commits, goes to both the merge and the validator.
#
# /share1 is not mounted here, so the run's parts are pulled to node-local SSD,
# merged into $RUN.dataset.json (a manifest of the validated parts; no second
# copy of the data), validated by scripts/validate_run.py, and the manifest is
# pushed back next to the parts.  The summary lands in artifacts/c1/$RUN.json
# in the repository on /home2.  Exit status: the validator's.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
RUN="${C3_RUN:?set C3_RUN, the collection run to merge}"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
DIR="$C3_SCRATCH/merge/$RUN"
mkdir -p "$DIR"
cd "$REPO_DIR"
echo "[merge] node=$(hostname) run=$RUN commit=$C3_GIT_COMMIT"

rsync -a --exclude '*.tmp' -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$BLOBS/traces/$RUN/" "$DIR/" \
    || { echo "[merge] FATAL: could not pull $BLOBS/traces/$RUN"; exit 1; }
echo "[merge] pulled $(ls "$DIR" | grep -c '\.parquet$') parts, $(du -sh "$DIR" | cut -f1)"

"$C3_PY" scripts/merge_traces.py --stem "$DIR/$RUN" --out "$DIR/$RUN.dataset.json" \
    ${C3_MERGE_ARGS:-}
RC=0
"$C3_PY" scripts/validate_run.py --dataset "$DIR/$RUN.dataset.json" --run "$RUN" \
    --gpus-per-task "${C3_GPUS:-2}" ${C3_CORPUS:+--corpus "$C3_CORPUS"} \
    --out "artifacts/c1/$RUN.json" ${C3_MERGE_ARGS:-} || RC=$?
rsync -a -e "ssh $SSH_OPTS" "$DIR/$RUN.dataset.json" "$USER@$C3_LOGIN_HOST:$BLOBS/traces/$RUN/" \
    && echo "[merge] manifest pushed to $BLOBS/traces/$RUN/$RUN.dataset.json" \
    || { echo "[merge] FATAL: could not push the manifest"; exit 1; }
echo "[merge] validation exit status $RC; summary artifacts/c1/$RUN.json"
exit $RC
