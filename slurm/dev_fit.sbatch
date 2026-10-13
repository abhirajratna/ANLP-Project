#!/bin/bash
#SBATCH --job-name=c3-devfit
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --output=/home2/%u/c3-dev/logs/devfit-%j.out
#
# Development on the FIT HALF of one merged C1 run (plan v2 Sec. 5: nothing on the
# report half before the pre-registration is frozen):
#
#   sbatch --export="ALL,C3_REPO=/home2/$USER/c3-dev,C3_RUN=c1-llada-base,C3_DEV_DOCS=150" slurm/dev_fit.sbatch
#
# Pulls the run's parts to node-local SSD, keeps only fit-half documents
# (scripts/make_dev_subset.py; C3_DEV_DOCS of them, 0 = all), and runs E1/E2
# (analyze_traces.py on the dev manifest), E3 (analyze_e3.py --half fit) and T8
# (run_controls.py --half fit) with small resample counts, logging each step's
# peak memory and wall time.  Outputs go to runs/dev/ in the repo; none is a result.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3-dev}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
RUN="${C3_RUN:?set C3_RUN}"
N="${C3_DEV_DOCS:-150}"
BOOT="${C3_DEV_BOOT:-500}"
MANIFEST="${C3_MANIFEST:-artifacts/split_manifest.json}"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
export HF_HOME="$C3_SCRATCH/hf"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
SRC="$C3_SCRATCH/merge/$RUN"
OUT="$REPO_DIR/runs/dev/$RUN-fit$N"
mkdir -p "$SRC" "$OUT"
cd "$REPO_DIR"
echo "[dev] node=$(hostname) run=$RUN docs=$N boot=$BOOT commit=$C3_GIT_COMMIT"

rsync -a --exclude '*.tmp' -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$BLOBS/traces/$RUN/" "$SRC/"
echo "[dev] pulled $(du -sh "$SRC" | cut -f1)"

step() {   # step <name> <command...>: run, recording peak RSS and wall time
    local name=$1; shift
    local t0=$SECONDS rc=0
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -f "[dev] $name: %e s, max RSS %M kB" "$@" > "$OUT/$name.log" 2>&1 || rc=$?
        grep "^\[dev\] $name:" "$OUT/$name.log" || true
    else
        "$@" > "$OUT/$name.log" 2>&1 || rc=$?
        echo "[dev] $name: $((SECONDS - t0)) s"
    fi
    tail -4 "$OUT/$name.log" | sed "s/^/[dev]   /"
    echo "[dev] $name exit $rc"
}

step subset "$C3_PY" scripts/make_dev_subset.py --dataset "$SRC/$RUN.dataset.json" \
    --manifest "$MANIFEST" --n-docs "$N" --out-trace "$OUT/trace.parquet" \
    --out-manifest "$OUT/dev_manifest.json"
step e1e2 "$C3_PY" scripts/analyze_traces.py --traces "$OUT/trace.parquet" --split half \
    --manifest "$OUT/dev_manifest.json" --n-boot "$BOOT" --eq-boot 50 --n-jobs 8 --knots 9 \
    --no-figures --out "$OUT/e1e2" --macro-prefix Dev
step e3 "$C3_PY" scripts/analyze_e3.py --traces "$OUT/trace.parquet" --split half --half fit \
    --manifest "$MANIFEST" --n-boot "$BOOT" --knots 9 --reference both --out "$OUT/e3" \
    --macro-prefix Dev
SRCARGS=(--docs "$PERSIST/corpus/calibration_docs.jsonl")
case "$RUN" in
    *gsm8k*) SRCARGS=(--task gsm8k --task-jsonl "$PERSIST/corpus/tasks/gsm8k_test.jsonl") ;;
    *mbpp*)  SRCARGS=(--task mbpp --task-jsonl "$PERSIST/corpus/tasks/mbpp_test.jsonl") ;;
esac
step t8 "$C3_PY" scripts/run_controls.py --traces "$OUT/trace.parquet" --split half --half fit \
    --manifest "$MANIFEST" --n-boot "$BOOT" --knots 9 \
    --controls position token_type temperature --temperature softmax "${SRCARGS[@]}" \
    --out "$OUT/t8" --macro-prefix Dev
echo "[dev] done -> $OUT"
