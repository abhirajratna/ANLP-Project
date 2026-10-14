#!/bin/bash
#SBATCH --job-name=c3-wandb
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home2/%u/c3/logs/wandb-%j.out
#
# Upload finished runs to W&B (no GPU).  The wandb library ships in the env tarball,
# so this runs on a compute node; the login node has no `wandb` command.
#
#     sbatch --export=ALL,C3_BACKFILL="train110m-v3 ft110m-v1-dryrun" slurm/wandb_backfill.sbatch
#     sbatch --export=ALL,C3_SYNC_OFFLINE=1 slurm/wandb_backfill.sbatch
#
# C3_BACKFILL: run ids whose history.jsonl is replayed (scripts/wandb_backfill.py);
# found as artifacts/history_<id>.jsonl or artifacts/probe/results/history_<id>.jsonl,
# with artifacts/{train_summary,gate_g2,eval_110m}_<id>.json as the run summary, or the
# fine-tune's train_summary.json from $BLOBS/probe/<id>/.  A missing history is
# skipped with a warning; a run id that already exists in W&B is refused.
# C3_SYNC_OFFLINE=1: also `wandb sync` every ./wandb/offline-run-* not yet synced
# (runs whose live start failed and fell back to offline; c3.training.train.start_wandb).
set -uo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
cd "$REPO_DIR"
grep -qs api.wandb.ai "$HOME/.netrc" || { echo "[wandb] FATAL: no api.wandb.ai entry in ~/.netrc"; exit 1; }
WORK="$C3_SCRATCH/wandb-backfill"
mkdir -p "$WORK"
RC=0

for id in ${C3_BACKFILL:-}; do
    hist=""
    for h in "artifacts/history_$id.jsonl" "artifacts/probe/results/history_$id.jsonl"; do
        [ -f "$h" ] && hist="$h" && break
    done
    [ -n "$hist" ] || { echo "[wandb] WARNING: no history for $id; skipped"; RC=1; continue; }
    summ=()
    for kind in train_summary gate_g2 eval_110m; do
        [ -f "artifacts/${kind}_$id.json" ] && summ+=("artifacts/${kind}_$id.json")
    done
    if [ ! -f "artifacts/train_summary_$id.json" ] \
            && rsync -a -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$BLOBS/probe/$id/train_summary.json" \
               "$WORK/train_summary_$id.json" 2>/dev/null; then
        summ+=("$WORK/train_summary_$id.json")
    fi
    "$C3_PY" scripts/wandb_backfill.py --history "$hist" --run-id "$id" --summary "${summ[@]}" \
        || { echo "[wandb] $id: backfill failed"; RC=1; }
done

if [ "${C3_SYNC_OFFLINE:-0}" = 1 ]; then
    for d in wandb/offline-run-*; do
        [ -d "$d" ] || continue
        if ls "$d"/*.wandb.synced >/dev/null 2>&1; then
            echo "[wandb] $d: already synced"; continue
        fi
        "$(dirname "$C3_PY")/wandb" sync "$d" || { echo "[wandb] $d: sync failed"; RC=1; }
    done
fi
echo "$(date +%F_%T) wandb=${SLURM_JOB_ID:-?} backfill [${C3_BACKFILL:-}] sync_offline=${C3_SYNC_OFFLINE:-0} rc=$RC" >> logs/t3_jobs.txt
exit $RC
