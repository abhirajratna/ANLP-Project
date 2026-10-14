#!/bin/bash
#SBATCH --job-name=c3-fineweb
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home2/%u/c3/logs/fineweb-%j.out
#
# Downloads the data T3 needs, once, and keeps a verified copy on /share1:
#   https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT
# 14 parquet files, 28.5 GB, at a pinned revision (scripts/fetch_fineweb.py).
# The corpus, exclusion and shard jobs stage this copy to node-local SSD
# (c3_stage_fineweb in activate_env.sh) instead of streaming from the Hub.
#
# Safe to rerun: files already on /share1 are reused, and only missing ones are
# downloaded.  /share1/$USER/fineweb-10BT/MANIFEST.json marks a complete copy.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
cd "$REPO_DIR"

export HF_HOME="$C3_SCRATCH/hf"
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DISABLE_PROGRESS_BARS=1
DATA="$C3_SCRATCH/fineweb-10BT"
DURABLE="$C3_BLOBS/fineweb-10BT"
mkdir -p "$HF_HOME" "$DATA"

# Start from whatever an earlier run stored: a LAN copy, not an internet one.
ssh $C3_SSH_OPTS "$USER@$C3_LOGIN_HOST" "mkdir -p '$DURABLE'"
rsync -a --partial-dir=.rsync-partial -e "ssh $C3_SSH_OPTS" "$USER@$C3_LOGIN_HOST:$DURABLE/" "$DATA/"

# $DATA only ever holds verified files, so push them even when the download
# fails partway: a rerun on another node then resumes from them.
push_files() {
    rsync -a --partial-dir=.rsync-partial --exclude MANIFEST.json --exclude .rsync-partial \
        -e "ssh $C3_SSH_OPTS" "$DATA/" "$USER@$C3_LOGIN_HOST:$DURABLE/"
}
trap 'push_files || echo "[fetch] WARNING: could not push files to $DURABLE"' EXIT

"$C3_PY" scripts/fetch_fineweb.py download --dir "$DATA"

push_files
trap - EXIT
# The manifest goes last: its presence on /share1 means every file arrived.
rsync -a -e "ssh $C3_SSH_OPTS" "$DATA/MANIFEST.json" "$USER@$C3_LOGIN_HOST:$DURABLE/MANIFEST.json"
echo "[fetch] stored $(ssh $C3_SSH_OPTS "$USER@$C3_LOGIN_HOST" "du -sh '$DURABLE' | cut -f1") at $DURABLE"
