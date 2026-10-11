#!/bin/bash
#SBATCH --job-name=c3-pushshards
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home2/%u/c3/logs/pushshards-%j.out
#
# Pushes shards built with C3_SHARDS_PUSH=0 from the node that built them to
# /share1, so training is no longer pinned to that node.  Submit pinned there:
#
#     sbatch --nodelist=<node> --export=ALL,C3_SHARDS_DEST=/share1/$USER/v2/shards \
#         slurm/push_shards.sbatch
#
# The shards go to $C3_SHARDS_DEST.new and are renamed into place only after the
# pushed copy matches the node copy file for file (names and sizes).  The
# destination must be absent or empty: this job never moves or deletes existing
# shards (unlike the push in build_shards.sbatch, which keeps the previous set
# as shards.old).  SHA256SUMS of every file is written next to the shards, so a
# later job can verify the copy it pulls (`sha256sum -c SHA256SUMS`).
set -euo pipefail
DEST="${C3_SHARDS_DEST:?set C3_SHARDS_DEST, e.g. /share1/$USER/v2/shards}"
LOCAL="${C3_LOCAL_SHARDS:-/ssd_scratch/$USER/c3-shards}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
NODE="$(hostname -s)"
cd "${C3_REPO:-/home2/$USER/c3}"

[ -f "$LOCAL/index.json" ] || { echo "[push] FATAL: no shards at $NODE:$LOCAL"; exit 1; }
"${C3_PY:-python3}" - "$LOCAL/index.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); idx = json.loads(p.read_text())
names = {pathlib.Path(s[f]).name for k in ("shards", "val_shards") for s in idx[k]
         for f in ("path", "offsets")}
missing = sorted(n for n in names if not (p.parent / n).is_file())
print(f"[push] index: {idx['total_tokens']/1e9:.2f}B tokens, {len(idx['shards'])} train + "
      f"{len(idx['val_shards'])} val shards; exclusion {idx['exclusion'].get('by_role')}")
if missing:
    sys.exit(f"[push] FATAL: index.json lists files missing on this node: {missing[:5]}")
PY

echo "[push] hashing $(find "$LOCAL" -maxdepth 1 -type f ! -name SHA256SUMS | wc -l) files on $NODE"
( cd "$LOCAL" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' | sort \
    | xargs -P 4 -n 8 sha256sum | sort -k2 > SHA256SUMS )

ssh $SSH_OPTS "$USER@ada" "set -e
    if [ -d '$DEST' ] && [ -n \"\$(ls -A '$DEST')\" ]; then
        echo '[push] FATAL: $DEST is not empty; refusing to replace it'; exit 1
    fi
    mkdir -p '$DEST.new'"
echo "[push] $NODE:$LOCAL ($(du -sh "$LOCAL" | cut -f1)) -> ada:$DEST.new"
t0=$(date +%s)
rsync -a --partial-dir=.rsync-partial -e "ssh $SSH_OPTS" "$LOCAL/" "$USER@ada:$DEST.new/" \
    || { echo "[push] FATAL: rsync failed; resubmit to resume"; exit 1; }
echo "[push] transferred in $(( $(date +%s) - t0 )) s"

# The pushed copy must match the node copy: same file names, same sizes.
LOCAL_LIST="$(cd "$LOCAL" && find . -maxdepth 1 -type f -printf '%f %s\n' | sort)"
REMOTE_LIST="$(ssh $SSH_OPTS "$USER@ada" "cd '$DEST.new' && find . -maxdepth 1 -type f -printf '%f %s\n' | sort")"
if [ "$LOCAL_LIST" != "$REMOTE_LIST" ]; then
    echo "[push] FATAL: pushed copy differs from the node copy:"
    diff <(echo "$LOCAL_LIST") <(echo "$REMOTE_LIST") | head -20
    exit 1
fi
NFILES=$(echo "$LOCAL_LIST" | wc -l)
NBYTES=$(echo "$LOCAL_LIST" | awk '{s += $2} END {print s}')
echo "[push] verified: $NFILES files, $NBYTES bytes, names and sizes identical"

# index.json and SHA256SUMS are small: compare their hashes as well.
for f in index.json SHA256SUMS; do
    l=$(sha256sum "$LOCAL/$f" | cut -d' ' -f1)
    r=$(ssh $SSH_OPTS "$USER@ada" "sha256sum '$DEST.new/$f'" | cut -d' ' -f1)
    [ "$l" = "$r" ] || { echo "[push] FATAL: $f differs after transfer"; exit 1; }
done

ssh $SSH_OPTS "$USER@ada" "set -e; if [ -d '$DEST' ]; then rmdir '$DEST'; fi; mv '$DEST.new' '$DEST'"
echo "[push] done: $DEST ($NFILES files, $(( NBYTES / 1000000000 )) GB) from $NODE"
echo "$(date +%F_%T) pushshards=${SLURM_JOB_ID:-?} $NODE:$LOCAL -> $DEST ($NFILES files, $NBYTES bytes, verified)" >> logs/t3_jobs.txt
