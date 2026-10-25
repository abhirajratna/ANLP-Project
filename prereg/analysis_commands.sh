#!/usr/bin/env bash
# Pre-registered analysis commands (plan v2 T2): every endpoint, computed on the report half.
#
# Frozen with prereg/frozen_config.json at tag prereg-v1.  A change after the tag is
# a deviation (prereg/deviations.md).  Run from the repository root, at the tagged
# commit, after the C1 runs are merged and validated (artifacts/c1/<run>.json all_ok)
# and after the pre-registration is uploaded to OSF:
#
#   C3_REPORT_HALF_OK=1 TRACES=<dir with <run>/<run>.dataset.json> bash prereg/analysis_commands.sh
#
# TRACES holds both accounts' runs: the 8B runs from /share1/abhiraj.ratna/traces, and
# main-110m-v3 and timesplit-110m-v3 (the 110M, fp32) from /share1/ojas.k/traces.
#
# Sizing, measured on the whole fit half of LLaDA-Base (993 documents, 60M rows; job
# 21625, development only; the report half is the same size): E1/E2 without Eq. 2/3,
# 21 min and 38 GB; E3, 3 min and 21 GB; T8, 80 min and 52 GB (all at 200 resamples).
# Run each on a CPU job with >= 64 GB (u22 gives one CPU per 3 GB: slurm/CLUSTER.md).
# The Eq. 2/3 bootstrap refits per resample, ~3.7 s each on 150 documents with 8 CPUs, and
# ~310-475 s (extrapolated) with one worker on a whole FineWeb run (~120M rows, fit and report halves;
# slurm/prereg_analysis.sbatch), so it runs at 1,000 resamples (deviation 15).
#
# Nothing here is fitted on the report half: calibrators, splines, knots, ell bins
# and temperatures come from the fit half (manifest value 0); every number comes
# from the report half (value 1).  Intervals: 10,000 document-bootstrap resamples
# (Eq. 2/3: 1,000, deviation 15), percentile; declared only if the interval excludes 0 and |effect| > 2 x floor.
set -euo pipefail

if [ "${C3_REPORT_HALF_OK:-}" != "1" ]; then
    echo "refusing: these commands read the report half. Set C3_REPORT_HALF_OK=1 only after" >&2
    echo "the pre-registration (tag prereg-v1) is signed off and uploaded to OSF." >&2
    exit 1
fi

PY=${PY:-python}
TRACES=${TRACES:?directory holding <run>/<run>.dataset.json for every C1 run}
OUT=${OUT:-results}
N_JOBS=${N_JOBS:-16}
CORPUS=${CORPUS:-runs/corpus/calibration_docs.jsonl}
TASKS_DIR=${TASKS_DIR:-runs/corpus/tasks}
BOOT=(--n-boot 10000)

floor_of() {   # the T1 non-determinism floor of a model family, in Delta units
    "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['plan_v2']['floor_delta'])" "$1"
}
FLOOR_LLADA=$(floor_of artifacts/t2_pilot/floor_llada_base_fp16.json)
FLOOR_DREAM=$(floor_of "${FLOOR_DREAM_JSON:-artifacts/t2_pilot/floor_dream-7b-base_fp16.json}")
FLOOR_MDLM=$(floor_of artifacts/t2_pilot/floor_local-110m-v3_fp32.json)

# label        run directory                  manifest                               floor           corpus / task
ARMS=(
  "LladaBase           c1-llada-base             artifacts/split_manifest.json          $FLOOR_LLADA  fineweb"
  "DreamBase           c1-dream-base             artifacts/split_manifest.json          $FLOOR_DREAM  fineweb"
  "LladaInstructGsm    c1-llada-instruct-gsm8k   artifacts/split_manifest_gsm8k.json    $FLOOR_LLADA  gsm8k"
  "LladaInstructMbpp   c1-llada-instruct-mbpp-v2 artifacts/split_manifest_mbpp.json     $FLOOR_LLADA  mbpp"
  "DreamInstructGsm    c1-dream-instruct-gsm8k   artifacts/split_manifest_gsm8k.json    $FLOOR_DREAM  gsm8k"
  "DreamInstructMbpp   c1-dream-instruct-mbpp    artifacts/split_manifest_mbpp.json     $FLOOR_DREAM  mbpp"
  "Mdlm                main-110m-v3              artifacts/split_manifest.json          $FLOOR_MDLM   fineweb"
)
# Members of the E1 and E2 Holm families (plan v2 Sec. 5: models x datasets); every
# other arm (decoder k = 2 and 8, uniform_matched) is reported as exploratory.
PRIMARY_ARMS=(uniform decoder:topk_conf:k4)

# The runs live under two Ada accounts, and /share1 has no ACLs, so each account runs
# the arms whose traces it holds: a run missing from TRACES is skipped and named.  The
# Holm summary needs every arm.  It runs once $OUT/c1/<label>/results.json exists for
# all of them.  Copy the other account's $OUT/c1/<label> directories in, then re-run
# with SUMMARY_ONLY=1.
have() { [ -e "$TRACES/$1/$1.dataset.json" ] || { echo "[prereg] $1 not in TRACES; skipped here"; return 1; }; }
summarize() {   # Holm across models x datasets for the two primary contrasts; gamma_2..4 within arm
    local inputs=() missing=() label
    for row in "${ARMS[@]}"; do
        read -r label _ <<<"$row"
        if [ -e "$OUT/c1/$label/results.json" ]; then inputs+=("$label=$OUT/c1/$label"); else missing+=("$label"); fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "[prereg] Holm summary waits for: ${missing[*]}"
        return 0
    fi
    "$PY" scripts/summarize_c1.py --results "${inputs[@]}" \
        --primary-arms "${PRIMARY_ARMS[@]}" --out "$OUT/c1/summary"
}
if [ "${SUMMARY_ONLY:-0}" = 1 ]; then summarize; exit 0; fi

# ---------------------------------------------------------------- E1, E2, Eq. 2/3
for row in "${ARMS[@]}"; do
    read -r LABEL RUN MANIFEST FLOOR DATA <<<"$row"
    have "$RUN" || continue
    T="$TRACES/$RUN/$RUN.dataset.json"
    COMMON=(--traces "$T" --split half --manifest "$MANIFEST" "${BOOT[@]}" --eq-boot 1000
            --floor "$FLOOR" --n-jobs "$N_JOBS")
    # Primary: Delta(m), Delta(32) - Delta(1), ECE and smooth ECE, the Prop. 2
    # decomposition (isotonic own, interacted context calibrator), ICC, Eq. 2/3 with
    # and without state FE, the native gap; 9 knots.
    "$PY" scripts/analyze_traces.py "${COMMON[@]}" --knots 9 \
        --out "$OUT/c1/$LABEL" --macro-prefix "COne$LABEL"
    # Knot sensitivity panel (reported alongside, never instead); Eq. 2/3 at 200 resamples.
    "$PY" scripts/analyze_traces.py "${COMMON[@]}" --knots 5 --eq-boot 200 --no-figures \
        --out "$OUT/c1/${LABEL}_knots5" --macro-prefix "COne${LABEL}KnotsFive"
    "$PY" scripts/analyze_traces.py "${COMMON[@]}" --knots 13 --eq-boot 200 --no-figures \
        --out "$OUT/c1/${LABEL}_knots13" --macro-prefix "COne${LABEL}KnotsThirteen"
    # Context-calibrator sensitivity: plan v2 T6(2)'s additive form (the interacted form is
    # primary).  The calibrator enters only the field term, so E1/E2 only.
    "$PY" scripts/analyze_traces.py "${COMMON[@]}" --knots 9 --context-model additive \
        --skip-regression --no-figures \
        --out "$OUT/c1/${LABEL}_additive" --macro-prefix "COne${LABEL}Additive"
    # Contamination (T4): the 8B FineWeb arms without the documents flagged at
    # c3.CONTAM_THRESHOLD = 32 of 50 tokens.  The 110M was trained with the calibration
    # documents excluded (artifacts/exclusion_manifest.json), so it has no such variant.
    case "$LABEL" in
        LladaBase) MODEL=llada-8b-base ;;
        DreamBase) MODEL=dream-7b-base ;;
        *) MODEL= ;;
    esac
    if [ -n "$MODEL" ]; then
        "$PY" scripts/analyze_traces.py "${COMMON[@]}" --knots 9 \
            --exclude-docs "artifacts/contamination_$MODEL.json" --skip-regression \
            --out "$OUT/c1/${LABEL}_clean" --macro-prefix "COne${LABEL}Clean"
    fi
done

summarize

# Time-split set (T4, guaranteed unseen): its own split manifest, written with
#   python scripts/make_split_manifest.py --docs runs/corpus/t3/timesplit_docs.jsonl \
#       --pilot-docs runs/corpus/reserve_docs.jsonl --exclusion-manifest '' --any-ids \
#       --out artifacts/split_manifest_timesplit.json
# before any of its traces is analysed.
for row in "LladaBaseTimeSplit c1ts-llada-base $FLOOR_LLADA" "DreamBaseTimeSplit c1ts-dream-base $FLOOR_DREAM" \
           "MdlmTimeSplit timesplit-110m-v3 $FLOOR_MDLM"; do
    read -r LABEL RUN FLOOR <<<"$row"
    T="$TRACES/$RUN/$RUN.dataset.json"
    [ -e "$T" ] || { echo "[prereg] $RUN not collected; skipped"; continue; }
    "$PY" scripts/analyze_traces.py --traces "$T" --split half \
        --manifest artifacts/split_manifest_timesplit.json "${BOOT[@]}" --skip-regression \
        --floor "$FLOOR" --n-jobs "$N_JOBS" --knots 9 \
        --out "$OUT/c1/$LABEL" --macro-prefix "COne$LABEL"
done

# ---------------------------------------------------------------- E3 (T7)
# Decoder states vs matched-uniform (primary) and the uniform grid, at equal
# confidence, for every decoder arm; nuisance choices from the fit half.
for row in "${ARMS[@]}"; do
    read -r LABEL RUN MANIFEST FLOOR DATA <<<"$row"
    have "$RUN" || continue
    "$PY" scripts/analyze_e3.py --traces "$TRACES/$RUN/$RUN.dataset.json" --split half \
        --half report --manifest "$MANIFEST" "${BOOT[@]}" --knots 9 --n-ell-bins 100 \
        --reference both --out "$OUT/e3/$LABEL" --macro-prefix "EThree$LABEL"
done

# ---------------------------------------------------------------- T8 controls
# Position, token type and temperature on every arm; one ablation table per run.
for row in "${ARMS[@]}"; do
    read -r LABEL RUN MANIFEST FLOOR DATA <<<"$row"
    have "$RUN" || continue
    if [ "$DATA" = fineweb ]; then
        SRC=(--docs "$CORPUS")
    else
        SRC=(--task "$DATA" --task-jsonl "$TASKS_DIR/${DATA}_test.jsonl")
    fi
    case "$RUN" in *110m*) SRC+=(--tokenizer gpt2) ;; esac     # local-110m is not in KNOWN_MODELS
    "$PY" scripts/run_controls.py --traces "$TRACES/$RUN/$RUN.dataset.json" --split half \
        --half report --manifest "$MANIFEST" "${BOOT[@]}" --knots 9 \
        --controls position token_type temperature --temperature logodds softmax "${SRC[@]}" \
        --out "$OUT/t8/$LABEL" --macro-prefix "TEight$LABEL"
done
# Precision: LLaDA-8B-Base fp16 against 4-bit on the 200 documents of
# artifacts/t8_precision_docs.json (fit-half documents give the 4-bit calibrators).
if have c1-llada-base && have c1t8-llada-base-4bit; then
    "$PY" scripts/run_controls.py --traces "$TRACES/c1-llada-base/c1-llada-base.dataset.json" \
        --precision-trace "$TRACES/c1t8-llada-base-4bit/c1t8-llada-base-4bit.dataset.json" \
        --controls precision --split half --half report --manifest artifacts/split_manifest.json \
        "${BOOT[@]}" --eq-boot 1000 --n-jobs "$N_JOBS" --knots 9 \
        --out "$OUT/t8/LladaBasePrecision" --macro-prefix "TEightLladaBasePrecision"
fi

echo "[prereg] done -> $OUT"
