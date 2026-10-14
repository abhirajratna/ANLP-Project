#!/bin/bash
#SBATCH --job-name=c3-shards
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069,gnode076
#SBATCH --cpus-per-task=16
#SBATCH --mem=48000M
#SBATCH --time=1-00:00:00
#SBATCH --output=/home2/%u/c3/logs/shards-%j.out
#
# Tokenises the 110M training stream with every evaluation document removed:
# document-aligned shards (no packing) plus a small validation split.
#
# Exclusion uses the v2 manifest from scripts/build_exclusion.py (FineWeb id +
# text hash + first-256-token hash, covering the main, pilot, time-split and
# validation documents).  Without it the proposal-era manifest (text hashes
# only) is used and the job prints a warning.  The build fails if a held-out
# document sampled from the stream is never matched.
#
# Shards are built on node-local SSD and pushed to /share1, not /home2: 10B
# tokens is 20 GB as uint16 and /home2 has ~5 GB free.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"

LOCAL_SHARDS="$C3_SCRATCH/c3-shards"
export HF_HOME="$C3_SCRATCH/hf"
rm -rf "$LOCAL_SHARDS"
mkdir -p "$HF_HOME" "$LOCAL_SHARDS"
cd "$REPO_DIR"
c3_stage_fineweb        # all 14 files: the build reads the whole stream

# C3_MANIFEST: one or more space-separated manifests, e.g. the v2 record plus the
# time-split set (artifacts/exclusion_timesplit.json, marked external, so it is
# excluded if it ever appears but not required to match).
read -r -a MANIFESTS <<< "${C3_MANIFEST:-$PERSIST/corpus/exclusion_v2.json}"
if [ -z "${C3_MANIFEST:-}" ] && [ ! -f "${MANIFESTS[0]}" ]; then
    echo "[shards] WARNING: ${MANIFESTS[0]} not found; falling back to the proposal-era manifest"
    MANIFESTS=("$PERSIST/corpus/exclusion_manifest.json")
fi
for m in "${MANIFESTS[@]}"; do
    [ -f "$m" ] || { echo "[shards] FATAL: exclusion manifest $m not found"; exit 1; }
done
echo "[shards] exclusion manifests: ${MANIFESTS[*]}"

"$C3_PY" scripts/build_training_shards.py \
    --out-dir "$LOCAL_SHARDS" \
    --manifest "${MANIFESTS[@]}" \
    --target-tokens "${C3_MAX_TOKENS:-10000000000}" \
    --val-fraction "${C3_VAL_FRACTION:-0.0005}"

NODE="$(hostname -s)"
echo "$(date +%F_%T) shards built on $NODE:$LOCAL_SHARDS (job ${SLURM_JOB_ID:-?})" >> logs/shards_node.txt

# C3_SHARDS_PUSH=0: keep the shards on this node's SSD only, for when /share1
# cannot hold them next to the FineWeb copy (~27 GB + ~19 GB). Training must then
# run on this node: C3_TRAIN_HERE=1 submits it pinned here, with C3_TRAIN_BLOBS
# (an existing directory whose shards/ is empty) as the durable home of its
# checkpoints. A requeue lands back on this node, where the shards still are.
# Push them later from this node, once there is room, to lift the pinning.
if [ "${C3_SHARDS_PUSH:-1}" = 0 ]; then
    echo "[stage] C3_SHARDS_PUSH=0: shards stay on $NODE:$LOCAL_SHARDS ($(du -sh "$LOCAL_SHARDS" | cut -f1)); not pushed"
    if [ "${C3_TRAIN_HERE:-0}" = 1 ]; then
        TB="${C3_TRAIN_BLOBS:?set C3_TRAIN_BLOBS with C3_TRAIN_HERE=1}"
        TRAIN=$(sbatch --parsable --nodelist="$NODE" --export=ALL,C3_BLOBS="$TB",C3_ENV_BLOBS="$BLOBS" slurm/train_110m.sbatch)
        VERIFY=$(sbatch --parsable --dependency=afterok:"$TRAIN" --export=ALL,C3_BLOBS="$TB",C3_ENV_BLOBS="$BLOBS" slurm/verify_110m.sh)
        echo "[stage] submitted train=$TRAIN verify=$VERIFY pinned to $NODE, checkpoints -> $TB"
        echo "$(date +%F_%T) shards=${SLURM_JOB_ID:-?} on $NODE -> train=$TRAIN verify=$VERIFY (C3_BLOBS=$TB)" >> logs/t3_jobs.txt
    fi
    exit 0
fi

echo "[stage] pushing shards to $BLOBS/shards"
ssh $SSH_OPTS "$USER@$C3_LOGIN_HOST" "rm -rf '$BLOBS/shards.new' && mkdir -p '$BLOBS/shards.new'"
rsync -a --partial -e "ssh $SSH_OPTS" \
    "$LOCAL_SHARDS/" "$USER@$C3_LOGIN_HOST:$BLOBS/shards.new/" \
    || { echo "[stage] FATAL: shard push failed"; exit 1; }
# Swap in only once the whole set has arrived.
ssh $SSH_OPTS "$USER@$C3_LOGIN_HOST" "rm -rf '$BLOBS/shards.old'; [ -d '$BLOBS/shards' ] && mv '$BLOBS/shards' '$BLOBS/shards.old'; mv '$BLOBS/shards.new' '$BLOBS/shards'"
echo "[stage] done: $(du -sh "$LOCAL_SHARDS" | cut -f1) -> $BLOBS/shards"
