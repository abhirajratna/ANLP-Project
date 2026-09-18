# Sourced by every job: makes $C3_PY point at a usable interpreter.
#
# /share1 is not mounted on compute nodes, but it IS reachable from them by ssh
# back to the login node -- the same pattern as run-llama.sh. Large blobs live
# there (~73 GB free, 3,000-inode quota) and are staged to node-local SSD.
#
# Extraction is locked, atomic and validated. Several jobs can land on the same
# node at once, and an unlocked extract lets the second one see bin/python appear
# mid-unpack and start running against a half-written tree. That surfaces far
# away from the cause, as e.g.
#   ImportError: .../libcusparseLt.so.0: cannot read file data
# and it persists, because the next job sees bin/python and skips re-extraction.
C3_BLOBS="${C3_BLOBS:-/share1/$USER}"
C3_PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
C3_SCRATCH="${C3_SCRATCH:-/ssd_scratch/$USER}"
C3_SCRATCH_ENV="$C3_SCRATCH/c3-env"
C3_SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"

mkdir -p "$C3_SCRATCH"

_c3_env_ok() {
    [ -x "$C3_SCRATCH_ENV/bin/python" ] || return 1
    # A working interpreter is not enough: the failure mode is a truncated
    # shared library, which only shows up on import.
    "$C3_SCRATCH_ENV/bin/python" -c "import torch, numpy" >/dev/null 2>&1
}

if ! _c3_env_ok; then
    exec 9>"$C3_SCRATCH/.c3-env.lock"
    flock 9
    if ! _c3_env_ok; then
        echo "[env] preparing environment on $(hostname)"
        rm -rf "$C3_SCRATCH_ENV"
        TARBALL="$C3_SCRATCH/c3-env.tar.gz"
        if [ -r "$C3_PERSIST/c3-env.tar.gz" ]; then
            TARBALL="$C3_PERSIST/c3-env.tar.gz"
        elif [ ! -r "$TARBALL" ]; then
            echo "[env] pulling env tarball from $C3_BLOBS"
            rsync -a --inplace --partial -e "ssh $C3_SSH_OPTS" \
                "$USER@ada:$C3_BLOBS/c3-env.tar.gz" "$TARBALL" \
                || { echo "[env] FATAL: could not fetch env tarball"; exit 1; }
        fi
        # Unpack to a private directory and rename into place, so no other job
        # can observe a partially extracted tree.
        TMPD="$C3_SCRATCH/.c3-env.unpack.$$"
        rm -rf "$TMPD"; mkdir -p "$TMPD"
        tar -xzf "$TARBALL" -C "$TMPD" || { echo "[env] FATAL: extract failed"; exit 1; }
        mv "$TMPD/c3-env" "$C3_SCRATCH_ENV"
        rm -rf "$TMPD"
        _c3_env_ok || { echo "[env] FATAL: extracted environment fails to import torch"; exit 1; }
    fi
    flock -u 9
    exec 9>&-
fi

export C3_PY="$C3_SCRATCH_ENV/bin/python"
echo "[env] $("$C3_PY" -V 2>&1) ready on $(hostname)"
