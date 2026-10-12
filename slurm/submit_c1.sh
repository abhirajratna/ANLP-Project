#!/bin/bash
# The C1 trace sweep on both 8B families (plan v2 T5-T7; mid report Sec. 8), on Ada.
#
#   bash slurm/submit_c1.sh env              # the Python env tarball, if it is missing
#   bash slurm/submit_c1.sh smoke            # 6 runs x 5 documents, full design
#   bash slurm/submit_c1.sh main [a b c d]   # (a)-(d), in priority order (default all)
#   bash slurm/submit_c1.sh timesplit        # (e), needs corpus/timesplit_docs.jsonl
#   bash slurm/submit_c1.sh precision        # (f) T8: LLaDA-8B-Base 4-bit, 200 documents
#   bash slurm/submit_c1.sh floor dream-7b-base   # the T1 floor for Dream (LLaDA's is in t2_pilot)
#   bash slurm/submit_c1.sh merge <run> [gpus-per-task]
#   DRY_RUN=1 bash slurm/submit_c1.sh main   # print the sbatch lines, submit nothing
#   C3_SBATCH_ARGS="--dependency=afterok:<env job>" bash slurm/submit_c1.sh smoke
#
# Every submission is appended to logs/c1_jobs.txt: date, job id, run name, commit,
# and the full sbatch arguments.  QOS: 4 GPUs (two 2-GPU fp16 tasks at once),
# 8 queued jobs.  Arrays are throttled %2 and submitted in priority order, so
# with SLURM's FIFO-by-age both slots go to the earliest unfinished run.
#
# Settings (plan v2; the user's 29 Sep decisions): fp16 split over two 2080 Ti
# (device_map=auto, 9 GiB each), the full design (collect_traces defaults: 9
# rates x 2 uniform masks, top-k trajectories k = 2, 4, 8 with every state, a
# matched uniform state per decoder state, top-20).  Overflowing documents are
# skipped and flagged (--on-nonfinite skip); runs are merged by merge_validate.
#
# C3_DREAM_NO_MATCHED=1 drops the matched-uniform arm for the Dream runs only
# (the documented fallback if the ETA misses 16 Oct; record it as a deviation).
set -euo pipefail
cd "${C3_REPO:-/home2/$USER/c3}"
mkdir -p logs artifacts/c1
LOG=logs/c1_jobs.txt
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
FP16="C3_PRECISION=fp16,C3_DEVICE_MAP=auto,C3_MAX_MEMORY=0=9GiB;1=9GiB"
EXTRA="--on-nonfinite skip --merge-as none"
DREAM_EXTRA="$EXTRA${C3_DREAM_NO_MATCHED:+ --no-uniform-matched}"

submit() {   # submit <run> <sbatch args...>; C3_SBATCH_ARGS adds options, e.g. a dependency
    local run=$1; shift
    # shellcheck disable=SC2206
    local pre=(${C3_SBATCH_ARGS:-})
    set -- "${pre[@]}" "$@"
    if [ -n "${DRY_RUN:-}" ]; then
        echo "DRY_RUN sbatch $*"
        return 0
    fi
    local id
    id=$(sbatch --parsable "$@")
    printf '%s\tjob=%s\trun=%s\tcommit=%s\tsbatch %s\n' "$(date -Is)" "$id" "$run" "$COMMIT" "$*" >> "$LOG"
    echo "[c1] $run -> job $id"
}

need() { [ -r "$1" ] || { echo "[c1] FATAL: $1 is missing"; exit 1; }; }

check_corpus() {   # the corpus must hash to the committed value before anything collects on it
    need "$PERSIST/corpus/calibration_docs.jsonl"
    echo "af8898577a91687c221dba64308504a427c8cfa75a27d00a5ebabf71d5f18b07  $PERSIST/corpus/calibration_docs.jsonl" \
        | sha256sum -c --quiet || { echo "[c1] FATAL: calibration_docs.jsonl hash mismatch"; exit 1; }
}

trace() {   # trace <run> <shards> <gres> <exports>
    local run=$1 n=$2 gres=$3 exports=$4
    submit "$run" --gres="$gres" --array="0-$((n - 1))%2" --job-name="c3-$run" \
        --export="ALL,C3_RUN=$run,C3_NUM_SHARDS=$n,$exports" slurm/collect_traces.sbatch
}

case "${1:-}" in
env)
    submit buildenv slurm/build_env.sbatch
    ;;
smoke)
    check_corpus
    T="$PERSIST/corpus/tasks"
    trace c1smoke-llada-base 1 gpu:2 "$FP16,C3_MODEL=llada-8b-base,C3_LIMIT=5,C3_EXTRA_ARGS=$EXTRA"
    trace c1smoke-llada-instruct-gsm8k 1 gpu:2 "$FP16,C3_MODEL=llada-8b-instruct,C3_TASK=gsm8k,C3_LIMIT=5,C3_EXTRA_ARGS=$EXTRA --task-jsonl $T/gsm8k_test.jsonl"
    trace c1smoke-llada-instruct-mbpp 1 gpu:2 "$FP16,C3_MODEL=llada-8b-instruct,C3_TASK=mbpp,C3_LIMIT=5,C3_EXTRA_ARGS=$EXTRA --task-jsonl $T/mbpp_test.jsonl"
    trace c1smoke-dream-base 1 gpu:2 "$FP16,C3_MODEL=dream-7b-base,C3_LIMIT=5,C3_EXTRA_ARGS=$DREAM_EXTRA"
    trace c1smoke-dream-instruct-gsm8k 1 gpu:2 "$FP16,C3_MODEL=dream-7b-instruct,C3_TASK=gsm8k,C3_LIMIT=5,C3_EXTRA_ARGS=$DREAM_EXTRA --task-jsonl $T/gsm8k_test.jsonl"
    trace c1smoke-dream-instruct-mbpp 1 gpu:2 "$FP16,C3_MODEL=dream-7b-instruct,C3_TASK=mbpp,C3_LIMIT=5,C3_EXTRA_ARGS=$DREAM_EXTRA --task-jsonl $T/mbpp_test.jsonl"
    ;;
main)
    # main [a] [b] [c] [d]: the runs to submit (default all four, in this order)
    check_corpus
    T="$PERSIST/corpus/tasks"
    shift
    WANT="${*:-a b c d}"
    for x in $WANT; do
        case "$x" in
        a)  # (a) LLaDA-8B-Base, 2,000 FineWeb documents, full design: ~34 slot-hours
            trace c1-llada-base 2 gpu:2 "$FP16,C3_MODEL=llada-8b-base,C3_EXTRA_ARGS=$EXTRA" ;;
        b)  # (b) LLaDA-8B-Instruct on GSM8K (1,319) and MBPP (500)
            trace c1-llada-instruct-gsm8k 1 gpu:2 "$FP16,C3_MODEL=llada-8b-instruct,C3_TASK=gsm8k,C3_EXTRA_ARGS=$EXTRA --task-jsonl $T/gsm8k_test.jsonl"
            trace c1-llada-instruct-mbpp 1 gpu:2 "$FP16,C3_MODEL=llada-8b-instruct,C3_TASK=mbpp,C3_EXTRA_ARGS=$EXTRA --task-jsonl $T/mbpp_test.jsonl" ;;
        c)  # (c) Dream-7B-Base, 2,000 documents
            trace c1-dream-base 2 gpu:2 "$FP16,C3_MODEL=dream-7b-base,C3_EXTRA_ARGS=$DREAM_EXTRA" ;;
        d)  # (d) Dream-7B-Instruct on GSM8K and MBPP
            trace c1-dream-instruct-gsm8k 1 gpu:2 "$FP16,C3_MODEL=dream-7b-instruct,C3_TASK=gsm8k,C3_EXTRA_ARGS=$DREAM_EXTRA --task-jsonl $T/gsm8k_test.jsonl"
            trace c1-dream-instruct-mbpp 1 gpu:2 "$FP16,C3_MODEL=dream-7b-instruct,C3_TASK=mbpp,C3_EXTRA_ARGS=$DREAM_EXTRA --task-jsonl $T/mbpp_test.jsonl" ;;
        *)  echo "[c1] unknown run '$x' (want a, b, c or d)"; exit 2 ;;
        esac
    done
    ;;
timesplit)
    # (e) both base models on the 500-document time-split set (from branch t3-110m-probe)
    need "$PERSIST/corpus/timesplit_docs.jsonl"
    D="$PERSIST/corpus/timesplit_docs.jsonl"
    echo "[c1] time-split corpus sha256 $(sha256sum "$D" | cut -d' ' -f1)"
    trace c1ts-llada-base 1 gpu:2 "$FP16,C3_MODEL=llada-8b-base,C3_DOCS=$D,C3_EXTRA_ARGS=$EXTRA"
    trace c1ts-dream-base 1 gpu:2 "$FP16,C3_MODEL=dream-7b-base,C3_DOCS=$D,C3_EXTRA_ARGS=$DREAM_EXTRA"
    ;;
precision)
    # (f) T8 precision control: LLaDA-8B-Base in 4-bit on one GPU, the 200 documents of
    # artifacts/t8_precision_docs.json (scripts/make_precision_subset.py); never the main runs.
    D="$PERSIST/corpus/precision_docs.jsonl"
    need "$D"
    echo "a6c4ddcfdff39f58ad24f2eadb3df6e0aba2e0b36af7f54370611d5c41a35590  $D" | sha256sum -c --quiet \
        || { echo "[c1] FATAL: precision_docs.jsonl hash mismatch"; exit 1; }
    trace c1t8-llada-base-4bit 1 gpu:1 "C3_PRECISION=4bit,C3_MODEL=llada-8b-base,C3_DOCS=$D,C3_EXTRA_ARGS=$EXTRA"
    ;;
floor)
    # The non-determinism floor (plan v2 T1 step 4) for a model other than LLaDA-8B-Base,
    # exactly as the T2 pilot measured it: t2-pilot design, the first 20 reserve
    # documents, 5 repeats; then scripts/measure_floor.py on the five files.
    MODEL="${2:?usage: submit_c1.sh floor <model>}"
    D="$PERSIST/corpus/reserve_docs.jsonl"
    need "$D"
    echo "1ae28fab52f09a06380c0049b286d1acb5173b3282fa96cb3fefd9a74ce39628  $D" | sha256sum -c --quiet \
        || { echo "[c1] FATAL: reserve_docs.jsonl hash mismatch"; exit 1; }
    RUN="t2floor-$MODEL-fp16"
    submit "$RUN" --gres=gpu:2 --job-name="c3-$RUN" \
        --export="ALL,C3_RUN=$RUN,C3_PRESET=t2-pilot,C3_MODEL=$MODEL,C3_LIMIT=20,C3_DOCS=$D,C3_REPEAT=5,$FP16" \
        slurm/collect_traces.sbatch
    ;;
merge)
    RUN="${2:?usage: submit_c1.sh merge <run> [gpus-per-task]}"
    CORPUS=""
    case "$RUN" in
        *-base) CORPUS="$PERSIST/corpus/calibration_docs.jsonl" ;;
        c1ts-*) CORPUS="$PERSIST/corpus/timesplit_docs.jsonl" ;;
        c1t8-*) CORPUS="$PERSIST/corpus/precision_docs.jsonl" ;;
    esac
    submit "merge:$RUN" --job-name="c3-merge-$RUN" \
        --export="ALL,C3_RUN=$RUN,C3_GPUS=${3:-2}${CORPUS:+,C3_CORPUS=$CORPUS}" slurm/merge_validate.sbatch
    ;;
*)
    sed -n '2,25p' "$0"
    exit 2
    ;;
esac
