# c3_stage_fineweb [N]: make the verified FineWeb-Edu parquet files available on
# this node's SSD and point C3_FINEWEB_DIR at them, so the job reads that copy
# instead of streaming from the Hub.  N > 0 needs only the first N files, for
# jobs that read just the start of the stream; the samplers fail loudly if the
# stream ends before their pool.
#
# Sources, in order:
#   1. the node's own copy, when `scripts/fetch_fineweb.py check` passes on it.
#      /ssd_scratch persists on a node between jobs, so only the first job on a
#      node pays for a transfer;
#   2. the durable copy at $C3_BLOBS/fineweb-10BT on /share1, if one is still
#      there (a LAN copy).  It was deleted on 2026-10-12 to make room for the v2
#      shards, so this source is normally skipped;
#   3. the pinned revision, downloaded from the Hub straight to the node
#      (`scripts/fetch_fineweb.py download`; compute nodes have internet access).
#      Each file is checked against the Hub's sha256 before it is used.
#
# Sourced by activate_env.sh (needs C3_PY, C3_SCRATCH, C3_BLOBS, C3_SSH_OPTS,
# C3_LOGIN_HOST);
# tests/test_stage_fineweb.py runs it with stub commands.  C3_FINEWEB_LOCAL
# overrides the node-local directory (default $C3_SCRATCH/fineweb-10BT).
c3_stage_fineweb() {
    local first="${1:-0}" repo="${C3_REPO:-/home2/$USER/c3}"
    local src="$C3_BLOBS/fineweb-10BT" dst="${C3_FINEWEB_LOCAL:-$C3_SCRATCH/fineweb-10BT}"
    local fetch="$repo/scripts/fetch_fineweb.py" rc=0
    mkdir -p "$dst"
    exec 8>"$C3_SCRATCH/.c3-fineweb.lock"
    flock 8
    local share1=absent
    if [ -f "$dst/MANIFEST.json" ] && "$C3_PY" "$fetch" check --dir "$dst" --first "$first"; then
        share1=unused
    else
        # 0: a complete copy is there; 1: none; 255: the gateway is unreachable, which
        # must not be mistaken for "no copy" (it was, on 2026-10-12, when `ada` stopped
        # accepting ssh from compute nodes and a job fell through to a stalled download).
        local probe=0
        ssh $C3_SSH_OPTS "$USER@$C3_LOGIN_HOST" "test -f '$src/MANIFEST.json'" || probe=$?
        case "$probe" in 0) share1=present ;; 255) share1=unreachable ;; esac
        if [ "$share1" = unreachable ]; then
            echo "[fineweb] WARNING: cannot ssh to $USER@$C3_LOGIN_HOST; /share1 not checked" \
                 "(set C3_LOGIN_HOST to a reachable gateway)"
        fi
    fi
    if [ "$share1" = unused ]; then
        echo "[fineweb] using the copy already on $(hostname -s):$dst"
    elif [ "$share1" = present ] \
            && rsync -a -e "ssh $C3_SSH_OPTS" "$USER@$C3_LOGIN_HOST:$src/MANIFEST.json" "$dst/MANIFEST.json"; then
        "$C3_PY" "$fetch" names --dir "$dst" --first "$first" > "$dst/.staging-list" || rc=1
        echo "[fineweb] staging $(wc -l < "$dst/.staging-list") file(s) from $src to $dst"
        [ "$rc" = 0 ] && rsync -a --partial-dir=.rsync-partial --files-from="$dst/.staging-list" \
            -e "ssh $C3_SSH_OPTS" "$USER@$C3_LOGIN_HOST:$src/" "$dst/" \
            || { echo "[fineweb] staging from $src failed"; rc=1; }
    else
        echo "[fineweb] no verified copy on $(hostname -s) and none at $src ($share1);" \
             "downloading the pinned revision from the Hub to $dst"
        HF_HOME="${HF_HOME:-$C3_SCRATCH/hf}" HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 \
            HF_HUB_DISABLE_PROGRESS_BARS=1 \
            "$C3_PY" "$fetch" download --dir "$dst" --first "$first" || rc=1
    fi
    if [ "$rc" = 0 ]; then
        "$C3_PY" "$fetch" check --dir "$dst" --first "$first" || rc=1
    fi
    flock -u 8
    exec 8>&-
    if [ "$rc" != 0 ]; then
        echo "[fineweb] FATAL: no verified FineWeb copy in $dst"
        return 1
    fi
    export C3_FINEWEB_DIR="$dst"
}
