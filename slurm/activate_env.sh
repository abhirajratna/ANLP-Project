# Sourced by every job: makes $C3_PY point at a usable interpreter.
#
# /share1 is not mounted on compute nodes, but it IS reachable from them by ssh
# to a login gateway. Since the October 2026 update that is ada-gw1 (or ada-gw2;
# picked by login_host.sh): the old `ada` host refuses SSH. Large blobs live there
# (100 GB quota, 3,000 inodes) and are staged to node-local SSD.
#
# /ssd_scratch deletes files 7 days after they were created, so an environment
# unpacked days ago can lose files in the middle of a multi-day job. It is
# reused only if unpacked within C3_ENV_MAX_AGE_H hours (default 48): with the
# 4-day job limit, no file a job uses is ever older than 6 days.
#
# Extraction is locked, atomic and validated. Several jobs can land on the same
# node at once, and an unlocked extract lets the second one see bin/python appear
# mid-unpack and start running against a half-written tree. That surfaces far
# away from the cause, as e.g.
#   ImportError: .../libcusparseLt.so.0: cannot read file data
# and it persists, because the next job sees bin/python and skips re-extraction.
C3_BLOBS="${C3_BLOBS:-/share1/$USER}"
# Where jobs ssh/rsync to reach /share1: since the October 2026 cluster reset `ada`
# refuses ssh from compute nodes, so login_host.sh (sourced below) picks the first
# gateway that answers (ada-gw1, then ada-gw2) unless C3_LOGIN_HOST is already set.
# Since the same reset (Slurm 26.05), a job asking for more than ~3 GB per CPU has
# its cpus-per-task raised to fit (--mem=32G with 8 CPUs -> 11; 48G/16 -> 17;
# 64G/16 -> 22) after sbatch has exported SLURM_TRES_PER_TASK=cpu=<requested>, and
# every `srun` inside the job then dies with
#   srun: fatal: cpus-per-task set by two different environment variables
# (job 17339; reproduced and fixed by this block in test job 17348).  Make the two
# agree with what was actually allocated.
if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [[ "${SLURM_TRES_PER_TASK:-}" =~ ^cpu=[0-9]+$ ]] \
        && [ "${SLURM_TRES_PER_TASK#cpu=}" != "$SLURM_CPUS_PER_TASK" ]; then
    export SLURM_TRES_PER_TASK="cpu=$SLURM_CPUS_PER_TASK"
fi
C3_PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
C3_SCRATCH="${C3_SCRATCH:-/ssd_scratch/$USER}"
C3_SCRATCH_ENV="$C3_SCRATCH/c3-env"
C3_SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
# The login host /share1 is reached through (not hard-coded: `ada` closed to SSH).
source "$(dirname "${BASH_SOURCE[0]}")/login_host.sh"
C3_ENV_MAX_AGE_H="${C3_ENV_MAX_AGE_H:-48}"

mkdir -p "$C3_SCRATCH"

_c3_env_ok() {
    [ -x "$C3_SCRATCH_ENV/bin/python" ] || return 1
    local fresh
    fresh=$(find "$C3_SCRATCH_ENV/.c3-unpacked" -mmin -$((C3_ENV_MAX_AGE_H * 60)) 2>/dev/null)
    [ -n "$fresh" ] || return 1
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
        else
            # Always sync: a tarball left on this node may predate an env rebuild.
            # C3_ENV_BLOBS: where the tarball lives when C3_BLOBS points elsewhere.
            echo "[env] syncing env tarball from ${C3_ENV_BLOBS:-$C3_BLOBS}"
            rsync -a --inplace --partial -e "ssh $C3_SSH_OPTS" \
                "$USER@$C3_LOGIN_HOST:${C3_ENV_BLOBS:-$C3_BLOBS}/c3-env.tar.gz" "$TARBALL" \
                || { echo "[env] FATAL: could not fetch env tarball"; exit 1; }
        fi
        # Unpack to a private directory and rename into place, so no other job
        # can observe a partially extracted tree.
        TMPD="$C3_SCRATCH/.c3-env.unpack.$$"
        rm -rf "$TMPD"; mkdir -p "$TMPD"
        tar -xzf "$TARBALL" -C "$TMPD" || { echo "[env] FATAL: extract failed"; exit 1; }
        mv "$TMPD/c3-env" "$C3_SCRATCH_ENV"
        rm -rf "$TMPD"
        touch "$C3_SCRATCH_ENV/.c3-unpacked"
        _c3_env_ok || { echo "[env] FATAL: extracted environment fails to import torch"; exit 1; }
    fi
    flock -u 9
    exec 9>&-
fi

export C3_PY="$C3_SCRATCH_ENV/bin/python"
echo "[env] $("$C3_PY" -V 2>&1) ready on $(hostname)"

# c3_stage_fineweb [N]: node-local FineWeb copy -> C3_FINEWEB_DIR (slurm/stage_fineweb.sh).
source "$(dirname "${BASH_SOURCE[0]}")/stage_fineweb.sh"
