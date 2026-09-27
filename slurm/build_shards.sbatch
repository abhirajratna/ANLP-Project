#!/bin/bash
#SBATCH --job-name=c3-shards
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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

MANIFEST="${C3_MANIFEST:-$PERSIST/corpus/exclusion_v2.json}"
if [ ! -f "$MANIFEST" ]; then
    echo "[shards] WARNING: $MANIFEST not found; falling back to the proposal-era manifest"
    MANIFEST="$PERSIST/corpus/exclusion_manifest.json"
fi

"$C3_PY" scripts/build_training_shards.py \
    --out-dir "$LOCAL_SHARDS" \
    --manifest "$MANIFEST" \
    --target-tokens "${C3_MAX_TOKENS:-10000000000}" \
    --val-fraction "${C3_VAL_FRACTION:-0.0005}"

echo "[stage] pushing shards to $BLOBS/shards"
ssh $SSH_OPTS "$USER@ada" "rm -rf '$BLOBS/shards.new' && mkdir -p '$BLOBS/shards.new'"
rsync -a --partial -e "ssh $SSH_OPTS" \
    "$LOCAL_SHARDS/" "$USER@ada:$BLOBS/shards.new/" \
    || { echo "[stage] FATAL: shard push failed"; exit 1; }
# Swap in only once the whole set has arrived.
ssh $SSH_OPTS "$USER@ada" "rm -rf '$BLOBS/shards.old'; [ -d '$BLOBS/shards' ] && mv '$BLOBS/shards' '$BLOBS/shards.old'; mv '$BLOBS/shards.new' '$BLOBS/shards'"
echo "[stage] done: $(du -sh "$LOCAL_SHARDS" | cut -f1) -> $BLOBS/shards"
