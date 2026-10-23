#!/bin/bash
# T13 ablations on the 110M v4 dev split (development only; dev-fit -> dev-eval).
# Run from the T13 checkout on ada-gw1 (not c3-dev, whose jobs read their code at start):
#
#   bash slurm/submit_t13.sh cpu                 # ablations 1-3 on traces (one CPU job)
#   bash slurm/submit_t13.sh recal  [after]      # ablation 2: C3a/C3-indep under each calibrator
#   bash slurm/submit_t13.sh sizes  [after]      # ablation 3: C3 with calibrators from 100/300/all docs
#   bash slurm/submit_t13.sh block  [BL]         # ablation 5: every policy, block-wise (BL=32)
#   bash slurm/submit_t13.sh analyze <recal|sizes|block32> [after]
#
# [after] is a job id the infill jobs wait for (the cpu job that writes the calibrators).
# Infill uses the s50 task (50% scattered masks of positions 64-191, deviations item 14) and
# the dev2 grid; the baselines and trace-calibrated C3 already sit in infill/sweep1-s50.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
T13="/home2/$USER/c3-runs/t13/dev"
GRID=prereg/c3set_policies_dev2.json
S50="--mask-rate 0.5"
METHODS="none temperature temperature_rate histogram isotonic context"
LOG=logs/t13_jobs.txt
mkdir -p logs
dep() { [ -n "${1:-}" ] && echo "--dependency=afterok:$1" || true; }
note() { echo "$(date +%F_%T) $*" | tee -a "$LOG"; }
infill() {   # tag cal_dir only extra after [array]
    local E="ALL,C3_REPO=$REPO,C3_MODE=dev,C3_HALF=dev-eval,C3_TAG=$1,C3_GRID=$REPO/$GRID"
    [ -n "$2" ] && E="$E,C3_CAL_DIR=$2"
    [ -n "$3" ] && E="$E,C3_ONLY=$3"
    E="$E,C3_EXTRA_ARGS=$4,C3_NUM_SHARDS=${6:-2}"
    # shellcheck disable=SC2046
    sbatch --parsable --array=0-$(( ${6:-2} - 1 )) $(dep "${5:-}") --export="$E" slurm/infill_eval.sbatch
}

case "${1:?usage: submit_t13.sh cpu|recal|sizes|block|analyze}" in
cpu)
    J=$(sbatch --parsable --export="ALL,C3_REPO=$REPO,C3_STEPS=statistic recal sizes" slurm/t13_ablations.sbatch)
    note "t13 cpu (statistic recal sizes): $J"
    ;;
recal)
    for m in $METHODS; do
        J=$(infill t13-recal "$T13/cal/recal/$m" "c3a c3indep" "$S50 --cal-tag $m" "${2:-}")
        note "t13 infill recal $m: $J"
    done
    ;;
sizes)
    # the "all" size is every dev-fit document (manifest value 0) of the dev copy
    N_ALL=$(python3 -c "import json,sys; print(sum(v == 0 for v in json.load(open(sys.argv[1])).values()))" \
        "/home2/$USER/c3-runs/c3set/dev/main-110m-v4.fit.manifest.json")
    for n in ${C3_SIZES:-100 300} "$N_ALL"; do
        J=$(infill t13-sizes "$T13/cal/sizes/n$n" "c3a c3indep c3set" "$S50 --cal-tag n$n" "${2:-}")
        note "t13 infill sizes n$n: $J"
    done
    ;;
block)
    BL="${2:-32}"
    J=$(infill "t13-block$BL" "/home2/$USER/c3-runs/c3set/dev/cal/traces" "" \
        "$S50 --block-length $BL --tag b$BL" "" 4)
    note "t13 infill block $BL: $J"
    ;;
analyze)
    what="${2:?analyze recal|sizes|block}"
    case "$what" in
    recal) RUNS="sweep1-s50 t13-recal"
           ARGS="--new c3indep@context --pairs"
           for m in $METHODS; do ARGS="$ARGS c3a@$m threshold c3indep@$m c3indep@context"; done ;;
    sizes) RUNS="sweep1-s50 t13-sizes"; ARGS="--new c3set" ;;
    block*) BL="${what#block}"; RUNS="sweep1-s50 t13-block$BL"; ARGS="--new c3set --pairs"
            for p in topk_conf margin pc_sampler threshold entropy_bounded klass c3a c3indep c3set; do
                ARGS="$ARGS $p@b$BL $p"; done ;;
    esac
    J=$(sbatch --parsable $(dep "${3:-}") --export="ALL,C3_REPO=$REPO,C3_STEPS=infill,C3_INFILL_TAG=$what,C3_INFILL_RUNS=$RUNS,C3_INFILL_ARGS=$ARGS" slurm/t13_ablations.sbatch)
    note "t13 analyze $what: $J"
    ;;
*)
    echo "unknown subcommand $1"; exit 2 ;;
esac
