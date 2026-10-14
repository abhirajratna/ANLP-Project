#!/bin/bash
#SBATCH --job-name=c3-merge
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=22G
#SBATCH --time=08:00:00
#SBATCH --output=/home2/%u/c3/logs/merge-%j.out
#
# Merge one trace collection run and check it against the design (no GPU).
#
#     sbatch --export=ALL,C3_RUN=main-110m-v3 slurm/merge_traces.sbatch
#
# The parts are pulled from /share1/$USER/traces/$C3_RUN to node-local SSD, merged
# there (scripts/merge_traces.py refuses duplicate or missing documents and
# mismatched shards) and validated (scripts/validate_traces.py: rows per document
# by mask kind against the closed form and artifacts/trace_size_estimate.json).
# The merged file stays on this node: next to its parts it would not fit in the
# /share1 quota.  C3_PUSH_MERGED=1 pushes it to /share1/$USER/traces/$C3_RUN.parquet
# -- only once there is room, i.e. after the parts have been removed.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
RUN="${C3_RUN:?set C3_RUN, the trace run to merge}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
cd "$REPO_DIR"

WORK="$C3_SCRATCH/traces-merge/$RUN"
mkdir -p "$WORK/merged"
rsync -a --exclude '*.tmp' -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$BLOBS/traces/$RUN/" "$WORK/" \
    || { echo "[merge] FATAL: could not pull $BLOBS/traces/$RUN"; exit 1; }
echo "[merge] $RUN: $(ls "$WORK" | grep -c '\.parquet$') part files, $(du -sh "$WORK" | cut -f1) on $(hostname -s)"
# C3_MERGE_ARGS=--allow-mixed-commits only after checking that nothing the collection
# runs changed between the commits (main-110m-v3: shards 0-1 at 56eb0b9, 2-3 at 26fbe8c,
# which differ only in result files and this merge tooling).
"$C3_PY" scripts/merge_traces.py --stem "$WORK/$RUN" --out "$WORK/merged/$RUN.parquet" ${C3_MERGE_ARGS:-}
mkdir -p artifacts/traces_110m
cp "$WORK/merged/$RUN.meta.json" "artifacts/traces_110m/$RUN.meta.json"
"$C3_PY" scripts/validate_traces.py --traces "$WORK/merged/$RUN.parquet" \
    --out "artifacts/traces_110m/validate_$RUN.json"
echo "$(date +%F_%T) merge=${SLURM_JOB_ID:-?} $RUN -> $(hostname -s):$WORK/merged/$RUN.parquet (validated)" >> logs/t3_jobs.txt
if [ "${C3_PUSH_MERGED:-0}" = 1 ]; then
    rsync -a --partial-dir=.rsync-partial -e "ssh $SSH_OPTS" "$WORK/merged/$RUN.parquet" \
        "$WORK/merged/$RUN.meta.json" "$USER@$C3_LOGIN_HOST:$BLOBS/traces/" \
        && echo "[merge] pushed to $BLOBS/traces/$RUN.parquet"
fi
echo "[merge] done: commit artifacts/traces_110m/*$RUN*"
