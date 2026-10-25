#!/bin/bash
# T12 on a Jarvis Labs GPU container, started by `jl run` (plan to the final, Parth 5):
# set up, verify, check, calibrate or sweep, then EXIT -- the laptop side (scripts/jl_t12.sh)
# fetches the outputs and pauses the instance.  Target: "2x RTX-PRO6000" (Blackwell, sm_120,
# 96 GB each, 100 GB SSD); an H100 works too.  One worker per visible GPU, each holding the
# whole fp16 model (~16 GB).  This script runs ON THE INSTANCE; do not start it by hand
# except for debugging.  The laptop wrapper builds the upload (a git bundle of the commit,
# so the files are LF and the split manifests keep their frozen sha256; the Ada-only inputs),
# `jl run` ships it and runs scripts/jl_t12.sh's bootstrap, which ends in
#
#   bash scripts/jarvis_t12.sh job calib|smoke|report|report-gsm8k|report-mbpp
#
# Phases (every one: setup -> verify-gpu -> ..., fit half only unless it says report):
#   calib    setup, verify-gpu, check (fit-half GSM8K at Fast-dLLM's settings: timing and a
#            rough accuracy check against Ada), calibrate (GSM8K and MBPP fit half, 0.7 h cap),
#            pick_knobs -> the two per-task grids in runs/t12/cal-grids/
#   report   setup, verify-gpu, then the report-half sweep of GSM8K (prereg/t12_grid_gsm8k.json)
#            and MBPP (prereg/t12_grid_mbpp.json), one run_t12.py per GPU each, in that order;
#            report-gsm8k / report-mbpp run one task
#   smoke    setup without the model, verify-gpu, inputs: a few minutes on a cheap GPU to prove
#            the jl/bootstrap/uv/torch chain before the 2-GPU instance is paid for
#
# THE JOB ALWAYS ENDS, so that the instance can be paused.  One global deadline per phase
# (BUDGET_HOURS: 6 for calib, 11 for report, 1 for smoke) is written once, into
# runs/t12/.deadline-<phase>, and never moves unless NEW_BUDGET=1; every supervise() stops at
# the earlier of its own cap and that deadline, and the whole job runs under `timeout` at the
# deadline plus 15 min, so a hung pip, download or GPU cannot keep it alive.  At the end, in
# every case (success, failure, deadline, SIGTERM), the outputs are packed to
# runs/t12-out/<phase>.tgz with a status line.  Exit codes: 0 done, 3 stopped by the budget
# (incomplete, resumable), anything else a failure.  There is no self-pause here: the
# instance has no API token (a secret on a rented box), and /home survives a pause anyway.
#
# Runs go to runs/t12/<tag>/ and are resumable: run_t12 appends and fsyncs every record, cuts
# a torn last line and skips every (problem, configuration) already recorded.  A worker that
# dies is relaunched (MAX_RESTARTS, default 2) with the same arguments, so it resumes; after
# that every worker is stopped and the job fails.  A rerun needs the same number of GPUs (the
# shard files are named t12.shard<i>of<N>; a mismatch is refused), the same checkout commit
# and the same grid.  Precision is fp16 (the C3a calibrator's traces, Ada); bf16 needs
# ALLOW_BF16=1.
#
# The environment is Python 3.12 (as on Ada) built by uv, torch==2.8.0 from the cu128 index
# (Blackwell kernels), transformers==4.46.3 (pyproject.toml); torch is held by a -c constraint
# in the later install.  Jarvis keeps only /home across a pause, so the venv, the uv Python and
# the model cache live inside the checkout.
#
# Knobs (environment, baked into the upload by `jl_t12.sh prepare`): BUDGET_HOURS, NEW_BUDGET,
# CAL_GRID (default prereg/t12_cal.json), CAL_MAX_HOURS (<= 0.7), CHECK_LIMIT (16),
# CHECK_MAX_HOURS (0.5), CHECK_MAX_SPF (warn above this s/forward), MAX_RESTARTS (2),
# RUN_ARGS (extra run_t12 arguments of a report sweep), PRECISION (fp16), NGPU (default: every
# visible GPU), POLL_SECONDS (10), STOP_GRACE (30), ALLOW_BF16, ALLOW_NONHOME, C3_VENV.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."
VENV="${C3_VENV:-.venv-t12}"
PY="$VENV/bin/python"
TASKS=runs/corpus/tasks
INPUTS=scripts/jarvis_t12.inputs.sha256
PRECISION="${PRECISION:-fp16}"
MAX_RESTARTS="${MAX_RESTARTS:-2}"
TORCH_PIN=2.8.0
export HF_HOME="${HF_HOME:-$PWD/runs/hf}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PWD/.uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export UV_HTTP_TIMEOUT=120
export PATH="$HOME/.local/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

die() { echo "[jarvis] FATAL: $*" >&2; exit 1; }
need_venv() { [ -x "$PY" ] || die "setup has not run ($PY missing)"; }

check_precision() {
    case "$PRECISION" in
        fp16) ;;
        bf16) [ "${ALLOW_BF16:-0}" = 1 ] || die "PRECISION=bf16 mixes precisions with Ada and the C3a calibrator's fp16 traces; set ALLOW_BF16=1 to override" ;;
        *) die "PRECISION=$PRECISION: use fp16 (or bf16 with ALLOW_BF16=1)" ;;
    esac
}

# Jarvis keeps only /home across a pause: a checkout elsewhere loses the venv and the model.
check_home() {
    case "$PWD" in
        /home/*) ;;
        *) [ "${ALLOW_NONHOME:-0}" = 1 ] || die "$PWD is outside /home, which Jarvis wipes on pause; clone under /home (or ALLOW_NONHOME=1)" ;;
    esac
}

# The checkout is a git work tree with LF files (a copied Windows tree has CRLF and no .git).
check_checkout() {
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$PWD is not a git checkout; the upload's bootstrap makes one from the bundle"
    local crlf
    crlf="$(git ls-files --eol | grep -c 'w/crlf' || true)"
    [ "$crlf" = 0 ] || die "$crlf tracked files have CRLF endings (core.autocrlf?); they would change the frozen manifests' sha256"
}

# verify_inputs TASK|all [tasks]: the files of $INPUTS that TASK needs exist with the recorded
# sha256.  "tasks" checks the task file and split manifest only (what `check` needs).  The MBPP
# task file is not uploaded: `setup` builds it from the pinned hub revision (c3/data/tasks.py)
# and this check pins its content.
verify_inputs() {
    local task="$1" only="${2:-}" bad=0 keep want path got
    while read -r want path; do
        want="${want%$'\r'}"
        path="${path%$'\r'}"
        case "$want" in ''|'#'*) continue ;; esac
        keep=1
        case "$path" in
            */tasks/*) [ "$task" = all ] || [[ "$path" == */tasks/"${task}"_* ]] || keep=0 ;;
            */split_manifest_*) [ "$task" = all ] || [[ "$path" == *split_manifest_"${task}".* ]] || keep=0 ;;
            *c3a-*) [ -z "$only" ] || keep=0
                    [ "$task" = all ] || [[ "$path" == *"c3a-${task}."* ]] || keep=0 ;;
            *) [ -z "$only" ] || keep=0 ;;
        esac
        [ "$keep" = 1 ] || continue
        if [ ! -s "$path" ]; then
            echo "[inputs] MISSING $path"; bad=1; continue
        fi
        got="$(sha256sum "$path" | cut -d' ' -f1)"
        if [ "$got" = "$want" ]; then
            echo "[inputs] ok $path"
        else
            echo "[inputs] SHA256 MISMATCH $path (got $got, want $want)"; bad=1
        fi
    done < "$INPUTS"
    [ "$bad" = 0 ] || die "inputs missing or different from Ada's; see $INPUTS"
}

# The GPUs to drive: CUDA_VISIBLE_DEVICES if set, else 0..N-1; NGPU caps or sets N.
detect_gpus() {
    GPU_IDS=()
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=, read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
        [ -z "${NGPU:-}" ] || GPU_IDS=("${GPU_IDS[@]:0:$NGPU}")
    else
        local n i
        n="${NGPU:-$("$PY" -c 'import torch; print(torch.cuda.device_count())')}"
        for ((i = 0; i < n; i++)); do GPU_IDS+=("$i"); done
    fi
    [ "${#GPU_IDS[@]}" -ge 1 ] || die "no visible GPU"
}

# start_budget KEY DEFAULT_HOURS: the global deadline of this job, written once (a rerun on
# the same instance keeps it; NEW_BUDGET=1 starts a new one) into runs/t12/.deadline-KEY.
start_budget() {
    local f="runs/t12/.deadline-$1" now left
    mkdir -p runs/t12
    now=$(date +%s)
    if [ ! -s "$f" ] || [ "${NEW_BUDGET:-0}" = 1 ]; then
        awk -v n="$now" -v h="${BUDGET_HOURS:-$2}" 'BEGIN { printf "%d\n", n + h * 3600 }' > "$f"
    fi
    DEADLINE_FILE="$f"
    left=$(($(cat "$f") - now))
    [ "$left" -gt 0 ] || die "the budget of '$1' is spent ($f); NEW_BUDGET=1 starts a new one"
    BUDGET_LEFT_S=$left
    echo "[budget] '$1': $((left / 60)) min left, until $(date -d "@$(cat "$f")" '+%F %T')"
}

# check_shards DIR N: refuse to continue a run that was sharded differently.
check_shards() {
    local f m
    for f in "$1"/t12.shard*of*.jsonl; do
        [ -e "$f" ] || continue
        m="$(basename "$f" | sed -E 's/.*of([0-9]+)\.jsonl$/\1/')"
        [ "$m" = "$2" ] || die "$f was written for $m shards but this run has $2 GPU(s); NGPU=$m resumes it, or use a new directory"
    done
}

# build_args full|light TASK HALF GRID: validate, verify the inputs and set ARGS.  `full` (the
# sweep, the calibration) also needs the calibrator and the PC-Sampler background text; a
# missing reserve file is an error, never the silent position-only variant.
build_args() {
    check_precision
    check_checkout
    [ -f "$4" ] || die "grid $4 not found (the policy grid JSON, relative to the repo)"
    [ -s "$TASKS/${2}_test.jsonl" ] || die "$TASKS/${2}_test.jsonl missing (setup builds it)"
    ARGS=(--model llada-8b-instruct --precision "$PRECISION" --device cuda
          --task "$2" --task-jsonl "$TASKS/${2}_test.jsonl" --half "$3"
          --manifest "artifacts/split_manifest_${2}.json" --configs "$4")
    if [ "$1" = full ]; then
        verify_inputs "$2"
        ARGS+=(--unigram-docs runs/corpus/reserve_docs.jsonl
               --calibrator "artifacts/c3/llada-8b-instruct_fp16_c3a-${2}.json")
    else
        verify_inputs "$2" tasks
    fi
}

# prepare_run full|light: build_args for TASK HALF GRID, the output directory and the GPUs.
prepare_run() {
    build_args "$1" "$TASK" "$HALF" "$GRID"
    OUT="runs/t12/$TAG"
    detect_gpus
}

# Start worker $1 in the background (pid in LAST_PID): shard $1 of one task per GPU, or
# (calibrate) one task each.  A relaunch uses the same arguments, so it resumes.
launch_worker() {
    local i="$1" wout wlog
    if [ "${CAL:-0}" = 1 ]; then
        local -n wargs="CAL_ARGS_$i"
        wout="${CAL_DIRS[$i]}"
        mkdir -p "$wout"
        wlog="$wout/worker.log"
        CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" "$PY" scripts/run_t12.py "${wargs[@]}" \
            --shard 0 --num-shards 1 --label "jarvis-cal-${CAL_TASKS[$i]}" --out-dir "$wout" \
            >> "$wlog" 2>&1 &
    else
        wlog="$OUT/worker$i.log"
        CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" "$PY" scripts/run_t12.py "${ARGS[@]}" "${EXTRA[@]}" \
            --shard "$i" --num-shards "${#GPU_IDS[@]}" --label "jarvis-$TAG" --out-dir "$OUT" \
            >> "$wlog" 2>&1 &
    fi
    LAST_PID=$!
    echo "[run] worker $i: GPU ${GPU_IDS[$i]}, pid $LAST_PID, log $wlog"
    sleep 2
    kill -0 "$LAST_PID" 2>/dev/null || echo "[run] worker $i (pid $LAST_PID) is already dead: see $wlog"
}

# TERM every live worker, KILL the rest after STOP_GRACE s, and reap them.
stop_workers() {
    local p t=0
    for p in "${PIDS[@]:-}"; do [ -z "$p" ] || kill -TERM "$p" 2>/dev/null || true; done
    while [ "$t" -lt "${STOP_GRACE:-30}" ]; do
        for p in "${PIDS[@]:-}"; do
            if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then sleep 1; t=$((t + 1)); continue 2; fi
        done
        break
    done
    for p in "${PIDS[@]:-}"; do
        [ -z "$p" ] || kill -KILL "$p" 2>/dev/null || true
        [ -z "$p" ] || wait "$p" 2>/dev/null || true
    done
    PIDS=()
}

# supervise MAXH: one run_t12 per GPU (launch_worker), until they finish, MAXH hours pass or the
# global deadline comes.  A dead worker is waited for; a non-zero exit is relaunched up to
# MAX_RESTARTS times (it resumes), then every worker is stopped.  Sets WORST (the failing exit
# code, 0 if none; never reset after a crash) and STOPPED (none | cap | global).
supervise() {
    local maxh="$1" n=${#GPU_IDS[@]} i p rc now dl gdl live kind=cap
    WORST=0
    STOPPED=none
    PIDS=()
    RESTARTS=()
    mkdir -p "$OUT"
    now=$(date +%s)
    gdl=$(cat "$DEADLINE_FILE")
    dl=$(awk -v n="$now" -v h="$maxh" 'BEGIN { printf "%d", n + h * 3600 }')
    if [ "$dl" -ge "$gdl" ]; then dl=$gdl; kind=global; fi
    if [ "$now" -ge "$dl" ]; then
        echo "[run] no budget left"; STOPPED=global; return 0
    fi
    for ((i = 0; i < n; i++)); do
        RESTARTS[i]=0
        launch_worker "$i"
        PIDS[i]=$LAST_PID
    done
    trap 'stop_workers; exit 143' TERM INT
    echo "[run] $n worker(s); stopping at $(date -d "@$dl" '+%F %T') ($kind), MAX_RESTARTS=$MAX_RESTARTS"
    while :; do
        live=0
        for ((i = 0; i < n; i++)); do
            p="${PIDS[$i]}"
            [ -n "$p" ] || continue
            if kill -0 "$p" 2>/dev/null; then live=1; continue; fi
            rc=0
            wait "$p" || rc=$?
            if [ "$rc" = 0 ]; then
                echo "[run] worker $i (GPU ${GPU_IDS[$i]}) finished"
                PIDS[i]=""
            elif [ "${RESTARTS[$i]}" -lt "$MAX_RESTARTS" ]; then
                RESTARTS[i]=$((RESTARTS[i] + 1))
                echo "[run] worker $i (GPU ${GPU_IDS[$i]}) exit code $rc: relaunch ${RESTARTS[$i]}/$MAX_RESTARTS (it resumes)"
                launch_worker "$i"
                PIDS[i]=$LAST_PID
                live=1
            else
                echo "[run] worker $i (GPU ${GPU_IDS[$i]}) exit code $rc after $MAX_RESTARTS relaunches: stopping every worker"
                PIDS[i]=""
                WORST=$rc
                break
            fi
        done
        [ "$WORST" = 0 ] || break
        [ "$live" = 1 ] || break
        if [ "$(date +%s)" -ge "$dl" ]; then
            STOPPED=$kind
            echo "[run] $kind limit reached: stopping the workers (every finished record is kept)"
            break
        fi
        sleep "${POLL_SECONDS:-10}"
    done
    stop_workers
    trap 'exit 143' TERM INT
}

# summarize DIR...: per configuration, over the problems it has recorded.  Speed is the
# aggregate gen_length * N / sum(n_forward) (the metric of pick_knobs and the grids); the mean
# of per-problem ratios is a secondary column (it runs ~10% higher at tau 0.9).  With
# SUMMARY_MANIFEST set, a report-half problem is an error.
summarize() {
    "$PY" -I - "$@" <<'PYEOF'
import json, os, sys
from collections import defaultdict
from pathlib import Path
man = os.environ.get("SUMMARY_MANIFEST")
split = json.loads(Path(man).read_text()) if man else None
a = defaultdict(lambda: defaultdict(float))
glen = 256
for d in sys.argv[1:]:
    for m in Path(d).glob("t12.shard*.meta.json"):
        glen = int(json.loads(m.read_text()).get("gen_length", glen))
    for p in sorted(Path(d).glob("t12.shard*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if split is not None:
                assert int(split[str(r["doc_id"])]) == 0, f"report-half problem {r['doc_id']} in a fit-half run"
            c = a[(d, r["config"])]
            c["n"] += 1
            if r.get("error"):
                c["err"] += 1
            else:
                c["fwd"] += r["n_forward"]; c["sec"] += r["seconds"]; c["ratio"] += r["tokens_per_step"]
for (d, k), c in sorted(a.items()):
    ok = int(c["n"] - c["err"])
    if ok == 0:
        print(f"[stat] {Path(d).name} {k:44s} {int(c['n']):4d} records  {int(c['err'])} errors")
        continue
    print(f"[stat] {Path(d).name} {k:44s} {int(c['n']):4d} records  {int(c['err'])} errors  "
          f"{glen * ok / c['fwd']:6.2f} tok/step (aggregate {glen}*N/sum(NFE))  "
          f"mean of ratios {c['ratio'] / ok:6.2f}  {c['sec'] / c['fwd']:.3f} s/forward  "
          f"{c['sec'] / ok:6.1f} s/problem")
print(f"[stat] {int(sum(c['n'] for c in a.values()))} records, "
      f"{sum(c['sec'] for c in a.values()) / 3600:.2f} GPU-h of decoding")
PYEOF
}

ensure_uv() {
    command -v uv >/dev/null 2>&1 && return 0
    timeout 300 sh -c 'curl --max-time 120 -LsSf https://astral.sh/uv/install.sh | sh' >/dev/null 2>&1 \
        || die "could not install uv"
    command -v uv >/dev/null 2>&1 || die "uv is not on PATH after its installer"
}

# retry N CMD...: a dropped connection mid-download is common; uv keeps what it fetched.
retry() {
    local n="$1" t
    shift
    for ((t = 1; t <= n; t++)); do
        "$@" && return 0
        echo "[retry] try $t/$n of: $*" >&2
        sleep 15
    done
    die "failed $n times: $*"
}

py_ok() { "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; }

do_setup() {
    check_precision
    check_home
    check_checkout
    ensure_uv
    mkdir -p runs/t12
    # Python 3.12 (Ada's 3.12.4; pyproject needs >= 3.11 and Ubuntu 22.04's python3 is 3.10).
    # An existing environment of another version is rebuilt, never reused.
    if [ -x "$PY" ] && ! py_ok; then
        echo "[setup] $VENV is Python $("$PY" -c 'import platform; print(platform.python_version())'), not 3.12: rebuilding"
        rm -rf "$VENV"
    fi
    [ -x "$PY" ] || timeout 1200 uv venv -p 3.12 "$VENV"
    py_ok || die "$VENV is not Python 3.12"
    echo "[setup] python $("$PY" -c 'import platform; print(platform.python_version())')"
    # torch 2.8.0 from the CUDA 12.8 index: the cu124 wheels have no sm_120 (Blackwell) kernels,
    # so setup would pass and the run would die at the first kernel.  The constraint file keeps
    # the later installs from moving it.  The rest as pinned in pyproject.toml [models]
    # (transformers==4.46.3: do not move it).
    retry 3 timeout 3600 uv pip install --python "$PY" "torch==$TORCH_PIN" --index-url https://download.pytorch.org/whl/cu128
    printf 'torch==%s\n' "$TORCH_PIN" > "$VENV/constraints.txt"
    retry 3 timeout 1800 uv pip install --python "$PY" -c "$VENV/constraints.txt" -e ".[models]"
    uv pip freeze --python "$PY" | grep -Ei '^(torch|transformers|datasets|numpy|tokenizers|huggingface-hub)==' || true
    # Fail fast, before the 16 GB model download is paid for.
    do_verify_gpu
    [ "${SKIP_MODEL:-0}" != 1 ] || { echo "[setup] SKIP_MODEL=1: no tasks, no model"; return 0; }
    mkdir -p "$TASKS"
    # The task files at their pinned hub revisions, unless uploaded already; verify_inputs pins
    # their content (the GSM8K file is Ada's; the MBPP file is the pinned revision, 500 problems).
    local t
    for t in gsm8k mbpp; do
        [ -s "$TASKS/${t}_test.jsonl" ] || timeout 900 "$PY" scripts/prepare_tasks.py --tasks "$t" --out-dir "$TASKS"
    done
    verify_inputs all tasks
    # The pinned LLaDA-8B-Instruct revision (c3.traces.models.KNOWN_MODELS), ~16 GB, once.
    timeout 3600 "$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
from c3.traces.models import KNOWN_MODELS
spec = KNOWN_MODELS["llada-8b-instruct"]
path = snapshot_download(spec["path"], revision=spec["revision"])
print("[setup] model at", path)
PYEOF
    df -h . | tail -n 1
    echo "[setup] done; HF_HOME=$HF_HOME"
}

do_verify_gpu() {
    need_venv
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true
    "$PY" - "$TORCH_PIN" <<'PYEOF'
import sys, torch
pin = sys.argv[1]
cu = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
print("[gpu] torch", torch.__version__, "cuda", torch.version.cuda, "arches", torch.cuda.get_arch_list())
if torch.__version__.split("+")[0] != pin or cu < (12, 8):
    sys.exit(f"[gpu] FATAL: need torch {pin} built for CUDA >= 12.8 (Blackwell sm_120)")
if not torch.cuda.is_available():
    sys.exit("[gpu] FATAL: no CUDA device (driver too old for CUDA 12.8? it needs >= 570)")
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    arch = f"sm_{cap[0]}{cap[1]}"
    if arch not in torch.cuda.get_arch_list():
        sys.exit(f"[gpu] FATAL: GPU {i} {torch.cuda.get_device_name(i)} is {arch}, not in this torch build's arches")
    a = torch.randn(512, 512, device=f"cuda:{i}", dtype=torch.float16)
    s = (a @ a).float().sum().item()
    torch.cuda.synchronize(i)
    if s != s or abs(s) == float("inf"):
        sys.exit(f"[gpu] FATAL: fp16 matmul on GPU {i} is not finite")
    print(f"[gpu] GPU {i} {torch.cuda.get_device_name(i)} {arch}: fp16 matmul ok")
import transformers
assert transformers.__version__ == "4.46.3", f"transformers {transformers.__version__}, pinned 4.46.3"
PYEOF
}

# Timing and sanity at Ada sanity run 38707's settings: fit-half GSM8K, lm-eval 5-shot, top-k
# k=1 and threshold 0.9, fp16, the first CHECK_LIMIT fit-half problems.  Never touches the
# report half.
do_check() {
    need_venv
    local limit="${CHECK_LIMIT:-16}" f
    [ "$limit" -ge 1 ] && [ "$limit" -le 40 ] || die "CHECK_LIMIT must be 1..40"
    TAG="check-$(date +%Y%m%d-%H%M%S)" TASK=gsm8k HALF=fit GRID=prereg/t12_sanity.json
    EXTRA=(--limit "$limit" --prompt lmeval --fewshot 5)
    prepare_run light
    mkdir -p "$OUT"
    exec 9>"$OUT/.lock"
    flock -n 9 || die "$OUT is in use by another process"
    check_shards "$OUT" "${#GPU_IDS[@]}"
    supervise "${CHECK_MAX_HOURS:-0.5}"
    [ "$WORST" = 0 ] || die "check: a worker failed (exit $WORST)"
    [ "$STOPPED" != global ] || die "check: the global budget ended"
    SUMMARY_MANIFEST=artifacts/split_manifest_gsm8k.json summarize "$OUT" | tee "$OUT/summary.txt"
    echo "[check] Ada sanity run 38707 (2x 2080 Ti, lm-eval 5-shot, same two policies): 0.80 s/forward;"
    echo "[check]   accuracy top-k k=1 83.5, threshold 0.9 84.5; published 79.3 / 79.2 (Fast-dLLM)."
    echo "[check] NOT one to one: Ada drew its few-shot examples over 4 shards of the problems, this run draws"
    echo "[check]   over the unsharded selection (run_t12.shard_contexts), so the examples differ; ${limit} problems"
    echo "[check]   give +-10 points.  Read this as a timing check (s/forward at the 5-shot prompt length)"
    echo "[check]   plus a rough accuracy check (tens of points off means a broken pipeline, not noise)."
    if [ -n "${CHECK_MAX_SPF:-}" ]; then
        # shellcheck disable=SC2016
        awk -v lim="$CHECK_MAX_SPF" '/s\/forward/ { for (i = 1; i <= NF; i++) if ($i == "s/forward") if ($(i - 1) + 0 > lim) { print "[check] WARNING: " $(i - 1) " s/forward > CHECK_MAX_SPF=" lim; } }' "$OUT/summary.txt"
    fi
    "$PY" scripts/score_t12.py "$OUT" --task-jsonl "gsm8k=$TASKS/gsm8k_test.jsonl" --out "$OUT/score.json"
    for f in "$OUT"/worker*.log; do [ -e "$f" ] && { echo "== $f"; tail -n 2 "$f"; }; done
    return 0
}

# Fit-half-only knob calibration (CAL_GRID, default the committed prereg/t12_cal.json): GSM8K,
# the first 32 fit-half problems, on GPU 0 and MBPP, the first 24, on GPU 1; zero-shot chat,
# fp16, one shard each, 0.7 h cap (problems are the outer loop, so a stop leaves every
# configuration on the same problems).  The output directories carry the grid's hash, so a
# different grid never lands in an old directory.  Then pick_knobs on both outputs, writing
# the per-task grids to runs/t12/cal-grids/.  Never touches the report half.
do_calibrate() {
    need_venv
    local grid="${CAL_GRID:-prereg/t12_cal.json}" pick="${PICK_KNOBS:-scripts/pick_knobs.py}" gh maxh
    [ -f "$grid" ] || die "CAL_GRID=$grid not found"
    [ -f "$pick" ] || die "PICK_KNOBS=$pick not found"
    maxh="${CAL_MAX_HOURS:-0.7}"
    awk -v h="$maxh" 'BEGIN { exit !(h > 0 && h <= 0.7) }' || die "CAL_MAX_HOURS=$maxh: the calibration cap is 0.7 h"
    detect_gpus
    [ "${#GPU_IDS[@]}" -ge 2 ] || die "calibrate needs 2 GPUs (one task each), have ${#GPU_IDS[@]}"
    GPU_IDS=("${GPU_IDS[@]:0:2}")
    gh="$(sha256sum "$grid" | cut -c1-8)"
    CAL=1
    CAL_TASKS=(gsm8k mbpp)
    CAL_DIRS=("runs/t12/cal-gsm8k-$gh" "runs/t12/cal-mbpp-$gh")
    OUT=runs/t12/cal-grids
    mkdir -p "${CAL_DIRS[@]}"
    exec 9>"${CAL_DIRS[0]}/.lock"
    flock -n 9 || die "${CAL_DIRS[0]} is in use by another process"
    exec 8>"${CAL_DIRS[1]}/.lock"
    flock -n 8 || die "${CAL_DIRS[1]} is in use by another process"
    check_shards "${CAL_DIRS[0]}" 1
    check_shards "${CAL_DIRS[1]}" 1
    build_args full gsm8k fit "$grid"
    # shellcheck disable=SC2034  # read by launch_worker through a nameref
    CAL_ARGS_0=("${ARGS[@]}" --limit 32)
    build_args full mbpp fit "$grid"
    # shellcheck disable=SC2034
    CAL_ARGS_1=("${ARGS[@]}" --limit 24)
    supervise "$maxh"
    CAL=0
    [ "$WORST" = 0 ] || die "calibrate: a worker failed (exit $WORST); not picking knobs from a partial run"
    [ "$STOPPED" != global ] || die "calibrate: the global budget ended"
    summarize "${CAL_DIRS[@]}"
    "$PY" -I "$pick" "${CAL_DIRS[@]}" --out-dir "$OUT"
    echo "[calibrate] grids: $(find "$OUT" -name 't12_grid_*.json' | sort | tr '\n' ' ')"
    echo "[calibrate] review them, copy them to prereg/, commit and push; the report phase reads them from the pushed commit"
}

# do_sweep TASK: the report-half sweep of one task with its own grid prereg/t12_grid_TASK.json.
# Sets SWEEP_RC: 0 done, 3 stopped by the budget (rerun resumes), else the failing exit code.
do_sweep() {
    local task="$1"
    need_venv
    TAG="sweep-$task" TASK="$task" HALF=report GRID="prereg/t12_grid_${task}.json"
    read -ra EXTRA <<< "${RUN_ARGS:-}"
    prepare_run full
    mkdir -p "$OUT"
    exec 9>"$OUT/.lock"
    flock -n 9 || die "$OUT is in use by another process"
    check_shards "$OUT" "${#GPU_IDS[@]}"
    supervise 100000
    summarize "$OUT" || true
    SWEEP_RC=0
    if [ "$WORST" != 0 ]; then
        SWEEP_RC=$WORST
    elif [ "$STOPPED" != none ]; then
        SWEEP_RC=3
        echo "[sweep] $task stopped by the budget; rerun on this instance with NEW_BUDGET=1 to resume"
    fi
}

# The end of every job, whatever happened: stop stray workers, pack the outputs, say so.
finish_job() {
    local rc=$? what
    trap - EXIT TERM INT
    stop_workers 2>/dev/null || true
    mkdir -p runs/t12-out
    echo "phase=$PHASE rc=$rc end=$(date -u +%FT%TZ) commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)" > "runs/t12/.job-$PHASE.status"
    if tar -czf "runs/t12-out/$PHASE.tgz" --exclude='.lock*' -C runs/t12 . 2>/dev/null; then
        (cd runs/t12-out && sha256sum "$PHASE.tgz" > "$PHASE.tgz.sha256")
        echo "[finish] outputs: $PWD/runs/t12-out/$PHASE.tgz ($(du -h "runs/t12-out/$PHASE.tgz" | cut -f1))"
    else
        echo "[finish] WARNING: could not pack runs/t12 (the directories are still on /home)"
    fi
    case "$rc" in
        0) what="done" ;;
        3) what="stopped by the budget; resumable" ;;
        *) what="FAILED" ;;
    esac
    cat <<EOF
================================================================================
 JOB '$PHASE' ENDED with exit code $rc ($what).
 The instance bills until it is paused: the laptop's jl_t12.sh fetches the outputs and pauses it.
================================================================================
EOF
    exit "$rc"
}

# job PHASE: the whole job under one hard timeout.  The first call writes the deadline and
# re-executes itself under `timeout`; the second does the work.
do_job() {
    PHASE="${1:?phase: calib|smoke|report|report-gsm8k|report-mbpp}"
    local hours left tasks=()
    case "$PHASE" in
        calib) hours=6 ;;
        smoke) hours=1 ;;
        report) hours=11; tasks=(gsm8k mbpp) ;;
        report-gsm8k) hours=11; tasks=(gsm8k) ;;
        report-mbpp) hours=11; tasks=(mbpp) ;;
        *) die "unknown phase $PHASE" ;;
    esac
    start_budget "$PHASE" "$hours"
    left=$BUDGET_LEFT_S
    if [ "${T12_JOB_WRAPPED:-0}" != 1 ]; then
        echo "[job] $PHASE under a hard timeout of $(((left + 900) / 60)) min"
        exec env T12_JOB_WRAPPED=1 NEW_BUDGET=0 timeout -k 60 "$((left + 900))" bash "$SELF" job "$PHASE"
    fi
    mkdir -p runs/t12
    exec 7>"runs/t12/.lock-job-$PHASE"
    flock -n 7 || die "another job '$PHASE' is running here"
    trap finish_job EXIT
    trap 'exit 143' TERM INT
    echo "[job] $PHASE at commit $(git rev-parse HEAD) on $(hostname), $(date -u +%FT%TZ)"
    JOBRC=0
    case "$PHASE" in
        smoke)
            SKIP_MODEL=1 do_setup
            verify_inputs gsm8k
            ;;
        calib)
            do_setup
            do_check
            do_calibrate
            ;;
        report*)
            do_setup
            local t
            for t in "${tasks[@]}"; do
                do_sweep "$t"
                [ "$SWEEP_RC" = 0 ] || { JOBRC=$SWEEP_RC; break; }
            done
            ;;
    esac
    exit "$JOBRC"
}

case "${1:?usage: jarvis_t12.sh job PHASE | setup | verify-gpu | verify-inputs | check | calibrate | sweep TASK | status}" in
job)
    do_job "${2:-}"
    ;;
setup)
    do_setup
    ;;
verify-gpu)
    do_verify_gpu
    ;;
verify-inputs)
    verify_inputs "${2:-all}"
    ;;
check|calibrate|sweep)
    # Debugging on the instance: the same steps as a job, with their own budget.
    start_budget "manual-$1" 6
    case "$1" in
        check) do_check ;;
        calibrate) do_calibrate ;;
        sweep) do_sweep "${2:?task (gsm8k|mbpp)}"; exit "$SWEEP_RC" ;;
    esac
    ;;
status)
    shift
    need_venv
    dirs=("$@")
    if [ "${#dirs[@]}" = 0 ]; then
        for d in runs/t12/*/; do
            compgen -G "${d}t12.shard*.jsonl" >/dev/null && dirs+=("${d%/}")
        done
    fi
    for f in runs/t12/.deadline-*; do
        [ -e "$f" ] && echo "budget $(basename "$f"): $((($(cat "$f") - $(date +%s)) / 60)) min left"
    done
    for f in runs/t12/.job-*.status; do [ -e "$f" ] && cat "$f"; done
    for d in "${dirs[@]}"; do
        for f in "$d"/worker*.log; do [ -e "$f" ] && { echo "== $f"; tail -n 2 "$f"; }; done
    done
    [ "${#dirs[@]}" = 0 ] || summarize "${dirs[@]}"
    ;;
*)
    echo "unknown subcommand $1"; exit 2 ;;
esac
