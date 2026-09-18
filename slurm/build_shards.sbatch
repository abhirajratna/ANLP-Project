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
# Tokenises the 110M training stream with the calibration documents removed.
#
# Shards are built on node-local SSD and then pushed to /share1, not /home2:
# 6B tokens is 12 GB as uint16 and /home2 has ~5 GB free. /share1 has ~73 GB and
# a 3,000-inode quota, which suits ~30 large shard files exactly.
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
mkdir -p "$HF_HOME" "$LOCAL_SHARDS"
cd "$REPO_DIR"

"$C3_PY" scripts/build_training_shards.py \
    --out-dir "$LOCAL_SHARDS" \
    --manifest "$PERSIST/corpus/exclusion_manifest.json" \
    --target-tokens "${C3_MAX_TOKENS:-6000000000}"

echo "[stage] pushing shards to $BLOBS/shards"
ssh $SSH_OPTS "$USER@ada" "mkdir -p '$BLOBS/shards'"
rsync -a --inplace --partial -e "ssh $SSH_OPTS" \
    "$LOCAL_SHARDS/" "$USER@ada:$BLOBS/shards/" \
    || { echo "[stage] FATAL: shard push failed"; exit 1; }
echo "[stage] done: $(du -sh "$LOCAL_SHARDS" | cut -f1) -> $BLOBS/shards"
