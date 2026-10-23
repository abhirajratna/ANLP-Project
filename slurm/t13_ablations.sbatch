#!/bin/bash
#SBATCH --job-name=c3-t13
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --cpus-per-task=20
#SBATCH --mem=60000M
#SBATCH --time=12:00:00
#SBATCH --output=/home2/%u/c3/logs/t13-%j.out
#
# T13 ablations on the fit-half dev copy of the 110M v4 trace (CPU; development only,
# dev-fit -> dev-eval, the report half is never read).  C3_STEPS lists the steps:
#
#   statistic  ablation 1: run_controls.py --controls statistic (pools won by margin /
#              entropy instead of ell)                          -> $OUT/statistic/
#   recal      ablation 2: compare_recalibration.py, and every method's tok.json fitted
#              on the dev-fit decoder arms                      -> $OUT/recal.json, $OUT/cal/recal/<m>/
#   sizes      ablation 3: nested dev-fit subsets (C3_SIZES, default "100 300 all"),
#              analyze_setcal.py on each: held-out set log-loss and the calibrators
#                                                               -> $OUT/sizes/, $OUT/cal/sizes/n<N>/
#   infill     analyze_infill.py over C3_INFILL_RUNS (dirs under the c3set dev infill
#              root) -> $OUT/infill_$C3_INFILL_TAG.json; C3_INFILL_ARGS passes --new/--pairs
#
#   sbatch --export=ALL,C3_REPO=/home2/$USER/c3-t13,C3_STEPS="statistic recal sizes" slurm/t13_ablations.sbatch
#
# The dev copy and its manifest are the ones c3set_analysis.sbatch's subset step wrote to
# /share1/$USER/c3set/dev/.  Token features follow the item-8 gate (decode features).
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
RUN="${C3_RUN:-main-110m-v4}"
STEPS="${C3_STEPS:?set C3_STEPS, e.g. \"statistic recal sizes\"}"
OUT="${C3_OUT:-/home2/$USER/c3-runs/t13/dev}"
INFILL_ROOT="${C3_INFILL_ROOT:-/home2/$USER/c3-runs/c3set/dev/infill}"
N_BOOT="${C3_N_BOOT:-10000}"
SIZES="${C3_SIZES:-100 300 all}"
TOKF=(--token-features ${C3_TOKEN_FEATURES:-median_ell mask_rate log_n_masked})
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"
cd "$REPO_DIR"
mkdir -p "$OUT" logs
WORK="$C3_SCRATCH/t13"
mkdir -p "$WORK/dev"
echo "[t13] node=$(hostname) steps='$STEPS' commit=$C3_GIT_COMMIT out=$OUT"

pull_dev() {
    if [ ! -r "$WORK/dev/$RUN.fit.parquet" ]; then
        rsync -a --exclude steps -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$BLOBS/c3set/dev/" "$WORK/dev/" \
            || { echo "[t13] FATAL: no dev copy on $BLOBS/c3set/dev"; exit 1; }
    fi
    TRACE="$WORK/dev/$RUN.fit.parquet"
    MANIFEST="$WORK/dev/$RUN.fit.manifest.json"
}

for step in $STEPS; do
    t0=$(date +%s)
    echo "[t13] ---- $step"
    case "$step" in
    statistic)
        pull_dev
        "$C3_PY" scripts/run_controls.py --traces "$TRACE" --controls statistic --split half \
            --half report --manifest "$MANIFEST" --arms uniform decoder:topk_conf:k4 \
            --n-boot "$N_BOOT" --macro-prefix TThirteenStat --out "$OUT/statistic"
        ;;
    recal)
        pull_dev
        "$C3_PY" scripts/compare_recalibration.py --traces "$TRACE" --manifest "$MANIFEST" \
            --n-boot "$N_BOOT" --save-calibrators "$OUT/cal/recal" --label dev --out "$OUT/recal.json"
        ;;
    sizes)
        pull_dev
        # shellcheck disable=SC2086
        "$C3_PY" scripts/make_size_manifests.py --manifest "$MANIFEST" --sizes $SIZES \
            --out-dir "$OUT/sizes"
        for m in "$OUT"/sizes/n*.manifest.json; do
            n="$(basename "$m" .manifest.json)"
            "$C3_PY" scripts/analyze_setcal.py --traces "$TRACE" --manifest "$m" "${TOKF[@]}" \
                --n-boot "$N_BOOT" --save-calibrators "$OUT/cal/sizes/$n" --label "dev-$n" \
                --out "$OUT/sizes/setcal_$n.json"
        done
        ;;
    infill)
        TAG="${C3_INFILL_TAG:?set C3_INFILL_TAG}"
        RUNS=()
        for r in ${C3_INFILL_RUNS:?set C3_INFILL_RUNS, infill dirs under $INFILL_ROOT}; do
            RUNS+=("$INFILL_ROOT/$r")
        done
        # shellcheck disable=SC2086
        "$C3_PY" scripts/analyze_infill.py --runs "${RUNS[@]}" --n-boot "$N_BOOT" \
            --label "dev-$TAG" ${C3_INFILL_ARGS:-} --out "$OUT/infill_$TAG.json"
        ;;
    *)
        echo "[t13] FATAL: unknown step $step"; exit 2 ;;
    esac
    echo "[t13] $step done in $(( $(date +%s) - t0 )) s"
done
echo "$(date +%F_%T) t13=${SLURM_JOB_ID:-?} steps='$STEPS' commit=$C3_GIT_COMMIT" >> logs/t13_jobs.txt
