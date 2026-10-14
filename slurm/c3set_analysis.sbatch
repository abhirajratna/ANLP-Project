#!/bin/bash
#SBATCH --job-name=c3-c3set
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --cpus-per-task=14
#SBATCH --mem=42000M
#SBATCH --time=12:00:00
#SBATCH --output=/home2/%u/c3/logs/c3set-%j.out
#
# Set-level C3 on the v4 traces (CPU).  C3_STEPS lists the steps, in order:
#
#   floor     measure_floor.py on the 5-repeat t2floor run -> artifacts/t2_pilot/
#   subset    fit-half dev copy of the trace (make_dev_subset.py --columns scalar),
#             pushed to /share1/$USER/c3set/dev/ (dev mode only)
#   fieldtest B, scripts/field_test.py                      -> $OUT/fieldtest.json
#   setcal    A, scripts/analyze_setcal.py on the decoder arms -> $OUT/setcal.json,
#             calibrators in $OUT/cal/traces/{tok,set}.json
#   onpolicy  A on the free-running step logs of $C3_STEPS_TAG -> $OUT/setcal_onpolicy.json,
#             calibrators in $OUT/cal/onpolicy/; the trace calibrators scored on the same
#             states -> $OUT/setcal_onpolicy_tracecal.json
#   fitcal    calibrators refitted on every fit document (the freeze) -> $OUT/cal/final-<src>/
#   infill    scripts/analyze_infill.py over $OUT/infill/$C3_INFILL_TAG -> $OUT/infill_<tag>.json
#
#   sbatch --export=ALL,C3_MODE=dev,C3_STEPS="floor subset fieldtest setcal" slurm/c3set_analysis.sbatch
#   sbatch --export=ALL,C3_MODE=report,C3_REPORT_HALF_OK=1,C3_STEPS="fieldtest setcal" slurm/c3set_analysis.sbatch
#
# dev mode reads only fit-half documents (the dev copy), split again into dev-fit /
# dev-eval; report mode reads the whole trace with the frozen split and needs
# C3_REPORT_HALF_OK=1 (the scripts refuse otherwise).  Results go to
# /home2/$USER/c3-runs/c3set/$C3_MODE.  The calibrators' features follow the dev
# field-test gate (deviations.md item 8; C3_GATE_FILE) unless C3_TOKEN_FEATURES is set
# ("none" = own confidence only).  Other knobs: C3_SET_ARGS, C3_FIELD_ARGS,
# C3_INFILL_ARGS, C3_CAL_SOURCE (traces|onpolicy, for fitcal), C3_N_BOOT.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
RUN="${C3_RUN:-main-110m-v4}"
MODE="${C3_MODE:?set C3_MODE=dev or report}"
STEPS="${C3_STEPS:?set C3_STEPS, e.g. \"subset fieldtest setcal\"}"
OUT="${C3_OUT:-/home2/$USER/c3-runs/c3set/$MODE}"
N_BOOT="${C3_N_BOOT:-10000}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
case "$MODE" in dev|report) ;; *) echo "[c3set] FATAL: C3_MODE must be dev or report"; exit 2 ;; esac

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
export C3_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-14}"
cd "$REPO_DIR"
mkdir -p "$OUT"
WORK="$C3_SCRATCH/c3set"
mkdir -p "$WORK"
echo "[c3set] node=$(hostname) mode=$MODE steps='$STEPS' commit=$C3_GIT_COMMIT out=$OUT"

# deviations.md item 8: the dev field test decides whether the calibrators use field
# features.  C3_TOKEN_FEATURES overrides it ("none" = own confidence only).
GATE_FILE="${C3_GATE_FILE:-/home2/$USER/c3-runs/c3set/dev/fieldtest.json}"
gate_args() {
    TOKF=()
    local feats="${C3_TOKEN_FEATURES:-}"
    if [ -z "$feats" ] && [ -r "$GATE_FILE" ]; then
        if "$C3_PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['gate']['use_field_features'] else 1)" "$GATE_FILE"; then
            feats="median_ell mask_rate log_n_masked"
        else
            feats=none
        fi
        echo "[c3set] item-8 gate from $GATE_FILE: token features = $feats"
    fi
    if [ "$feats" = none ]; then TOKF=(--token-features --no-set-field)
    elif [ -n "$feats" ]; then read -r -a f <<< "$feats"; TOKF=(--token-features "${f[@]}"); fi
}

pull_full() {          # the merged run's parts and manifest onto this node
    mkdir -p "$WORK/$RUN"
    rsync -a --exclude '*.tmp' -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$BLOBS/traces/$RUN/" "$WORK/$RUN/" \
        || { echo "[c3set] FATAL: could not pull $BLOBS/traces/$RUN"; exit 1; }
    TRACE="$WORK/$RUN/$RUN.dataset.json"
    [ -r "$TRACE" ] || { echo "[c3set] FATAL: $TRACE missing (merge_validate first)"; exit 1; }
}
pull_dev() {           # the fit-half dev copy
    mkdir -p "$WORK/dev"
    if [ ! -r "$WORK/dev/$RUN.fit.parquet" ]; then
        rsync -a -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$BLOBS/c3set/dev/" "$WORK/dev/" \
            || { echo "[c3set] FATAL: no dev copy on $BLOBS/c3set/dev (run the subset step)"; exit 1; }
    fi
    TRACE="$WORK/dev/$RUN.fit.parquet"
}
inputs() {             # TRACE and MANIFEST for the scoring steps
    if [ "$MODE" = dev ]; then pull_dev; MANIFEST="$WORK/dev/$RUN.fit.manifest.json"
    else pull_full; MANIFEST=artifacts/split_manifest.json; fi
}

for step in $STEPS; do
    t0=$(date +%s)
    echo "[c3set] ---- $step"
    case "$step" in
    floor)
        F="$WORK/t2floor"
        mkdir -p "$F"
        rsync -a -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$BLOBS/traces/t2floor-local-110m-v4-fp32/" "$F/"
        mkdir -p artifacts/t2_pilot
        "$C3_PY" scripts/measure_floor.py --traces "$F"/t2floor-local-110m-v4-fp32.rep*.parquet \
            --out artifacts/t2_pilot/floor_local-110m-v4_fp32.json
        ;;
    subset)
        [ "$MODE" = dev ] || { echo "[c3set] FATAL: subset is a dev step"; exit 2; }
        pull_full
        mkdir -p "$WORK/dev"
        "$C3_PY" scripts/make_dev_subset.py --dataset "$TRACE" --manifest artifacts/split_manifest.json \
            --columns scalar --out-trace "$WORK/dev/$RUN.fit.parquet" \
            --out-manifest "$WORK/dev/$RUN.fit.manifest.json"
        ssh $SSH_OPTS "$USER@$C3_LOGIN_HOST" "mkdir -p '$BLOBS/c3set/dev'"
        rsync -a -e "ssh $SSH_OPTS" "$WORK/dev/" "$USER@$C3_LOGIN_HOST:$BLOBS/c3set/dev/"
        cp "$WORK/dev/$RUN.fit.manifest.json" "$WORK/dev/$RUN.fit.meta.json" "$OUT/"
        rm -rf "$WORK/$RUN"            # the full trace is not needed again in dev mode
        ;;
    fieldtest)
        inputs
        "$C3_PY" scripts/field_test.py --traces "$TRACE" --manifest "$MANIFEST" --n-boot "$N_BOOT" \
            --label "$MODE" ${C3_FIELD_ARGS:-} --out "$OUT/fieldtest.json"
        ;;
    setcal)
        inputs
        gate_args
        "$C3_PY" scripts/analyze_setcal.py --traces "$TRACE" --manifest "$MANIFEST" "${TOKF[@]}" \
            --n-boot "$N_BOOT" --save-calibrators "$OUT/cal/traces" --label "$MODE-traces" \
            ${C3_SET_ARGS:-} --out "$OUT/setcal.json"
        ;;
    onpolicy)
        TAG="${C3_STEPS_TAG:?set C3_STEPS_TAG, the infill tag whose step logs to read}"
        S="$WORK/steps/$TAG"
        mkdir -p "$S"
        rsync -a -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$BLOBS/c3set/$MODE/steps/$TAG/" "$S/"
        [ "$MODE" = dev ] && { pull_dev; MANIFEST="$WORK/dev/$RUN.fit.manifest.json"; } \
            || MANIFEST=artifacts/split_manifest.json
        gate_args
        "$C3_PY" scripts/analyze_setcal.py --onpolicy "$S" --manifest "$MANIFEST" "${TOKF[@]}" \
            --n-boot "$N_BOOT" --save-calibrators "$OUT/cal/onpolicy" --label "$MODE-onpolicy-$TAG" \
            ${C3_SET_ARGS:-} --out "$OUT/setcal_onpolicy.json"
        # The trace-fitted calibrators on the same free-running states: the source with
        # the lower held-out set log-loss is the one frozen (Phase 5).
        if [ -r "$OUT/cal/traces/tok.json" ]; then
            "$C3_PY" scripts/analyze_setcal.py --onpolicy "$S" --manifest "$MANIFEST"                 --use-calibrators "$OUT/cal/traces" --n-boot "$N_BOOT"                 --label "$MODE-onpolicy-$TAG-tracecal" --out "$OUT/setcal_onpolicy_tracecal.json"
        fi
        ;;
    fitcal)
        [ "$MODE" = dev ] || { echo "[c3set] FATAL: calibrators are refitted on the fit half, in dev mode"; exit 2; }
        SRC="${C3_CAL_SOURCE:-traces}"
        ALL="$WORK/fit_all.manifest.json"
        pull_dev
        "$C3_PY" -c "import json,sys; m=json.load(open(sys.argv[1])); json.dump({k: 0 for k in m}, open(sys.argv[2], 'w'))" \
            "$WORK/dev/$RUN.fit.manifest.json" "$ALL"
        if [ "$SRC" = traces ]; then IN=(--traces "$TRACE")
        else S="$WORK/steps/${C3_STEPS_TAG:?set C3_STEPS_TAG}"; mkdir -p "$S"
             rsync -a -e "ssh $SSH_OPTS" "$USER@$C3_LOGIN_HOST:$BLOBS/c3set/$MODE/steps/$C3_STEPS_TAG/" "$S/"
             IN=(--onpolicy "$S"); fi
        gate_args
        "$C3_PY" scripts/analyze_setcal.py "${IN[@]}" --manifest "$ALL" --fit-only "${TOKF[@]}" \
            --save-calibrators "$OUT/cal/final-$SRC" --label "final-$SRC" ${C3_SET_ARGS:-} \
            --out "$OUT/cal/final-$SRC/fit.json"
        ;;
    infill)
        TAG="${C3_INFILL_TAG:?set C3_INFILL_TAG}"
        "$C3_PY" scripts/analyze_infill.py --runs "$OUT/infill/$TAG" --n-boot "$N_BOOT" \
            --label "$MODE-$TAG" --plot "$OUT/infill_$TAG.png" ${C3_INFILL_ARGS:-} \
            --out "$OUT/infill_$TAG.json"
        ;;
    *)
        echo "[c3set] FATAL: unknown step $step"; exit 2 ;;
    esac
    echo "[c3set] $step done in $(( $(date +%s) - t0 )) s"
done
echo "$(date +%F_%T) c3set=${SLURM_JOB_ID:-?} mode=$MODE steps='$STEPS' commit=$C3_GIT_COMMIT" >> logs/c3set_jobs.txt
