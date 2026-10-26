#!/bin/bash
# T12 on a Jarvis Labs VM (plan to the final, Parth 3-5): set up, check, run, watch, sync, pause.
# Target: "2x RTX PRO 6000" (Blackwell, sm_120, 96 GB each, 100 GB SSD, 416 INR/h); an H100
# VM works too.  One worker per visible GPU, each holding the whole fp16 model (~16 GB).
# Run from the repository checkout on the instance, which must live under /home: Jarvis
# keeps only /home across a pause (the venv and the model cache are inside the checkout).
# The checkout must come from git, never a copy of the Windows working tree: its CRLF line
# endings change the sha256 of the split manifests (and grids) that run_t12 records and the
# prereg freezes.  From the laptop, with the branch pushed or bundled:
#
#   git clone -b parth/t12-driver https://github.com/abhirajratna/ANLP-Project ~/ANLP-Project
#   # or: git bundle create c3.bundle parth/t12-driver; scp it; git clone -b parth/t12-driver c3.bundle ~/ANLP-Project
#
#   # laptop: copy the Ada-only inputs (staged in jarvis-stage/) into the checkout on the VM
#   VM=root@<host> VM_PORT=<port> VM_REPO=/home/<user>/ANLP-Project STAGE=<jarvis-stage> \
#       bash scripts/jarvis_t12.sh upload-inputs
#   bash scripts/jarvis_t12.sh setup                       # venv, torch cu128, GPU test, tasks, model
#   bash scripts/jarvis_t12.sh check                       # ~10-15 min: s/forward and accuracy vs Ada
#   CAL_GRID=<cal grid> PICK_KNOBS=<pick_knobs.py> bash scripts/jarvis_t12.sh calibrate   # <= 0.7 h
#   GRID=prereg/t12_grid.json SYNC_DEST=user@host:/path AUTO_PAUSE=1 JL_API_KEY=... JL_MACHINE_ID=... \
#       bash scripts/jarvis_t12.sh run sweep-gsm8k gsm8k report [run_t12 args]
#   bash scripts/jarvis_t12.sh status sweep-gsm8k          # records per configuration, time left
#   bash scripts/jarvis_t12.sh sync sweep-gsm8k            # pack + copy off (also done at the end)
#
# `run` starts a supervisor in the background (log: runs/t12/<tag>/supervisor.log) that runs
# one scripts/run_t12.py per GPU (--shard i --num-shards N, log: worker<i>.log), stops them
# after MAX_HOURS (default 12; SIGTERM, then SIGKILL after 30 s), reports each exit code,
# syncs the outputs to SYNC_DEST and, only with AUTO_PAUSE=1 and a successful sync, pauses
# the instance.  It never destroys one.  Otherwise it prints a banner: PAUSE THE INSTANCE.
#
# Runs go to runs/t12/<tag>/ and are resumable: run_t12 appends and fsyncs every record, cuts
# a torn last line and skips every (problem, configuration) already recorded, so rerunning the
# same `run` line continues -- with the same number of GPUs (the shard files are named
# t12.shard<i>of<N>), the same checkout commit (no `git pull` in between) and the same GRID.
# Precision is fp16 (the C3a calibrator's traces, Ada); bf16 needs ALLOW_BF16=1.
#
# `calibrate` is the fit-half-only knob calibration: GSM8K (32 problems) on GPU 0 and MBPP (24)
# on GPU 1, one worker each, hard cap 0.7 h, then pick_knobs.py on both outputs.  It never
# syncs or pauses (the sweep follows).  `check`, `calibrate` and `run` verify the manifests,
# calibrators and task files against $INPUTS (sha256) before generating anything.
#
# Knobs: GRID (grid JSON, required for run), CAL_GRID, PICK_KNOBS, MAX_HOURS (12), CHECK_LIMIT (16), PRECISION (fp16),
# NGPU (default: every visible GPU), SYNC_DEST (rsync/scp target), AUTO_PAUSE (0), JL_API_KEY
# and JL_MACHINE_ID (for AUTO_PAUSE; the id changes on resume, see `jl list`), ALLOW_BF16,
# ALLOW_NONHOME, C3_VENV.
set -euo pipefail
ORIG_PWD="$PWD"
cd "$(dirname "$0")/.."
VENV="${C3_VENV:-.venv-t12}"
PY="$VENV/bin/python"
TASKS=runs/corpus/tasks
INPUTS=scripts/jarvis_t12.inputs.sha256
PRECISION="${PRECISION:-fp16}"
MAX_HOURS="${MAX_HOURS:-12}"
AUTO_PAUSE="${AUTO_PAUSE:-0}"
SSH_OPTS="${SYNC_SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new}"
export HF_HOME="${HF_HOME:-$PWD/runs/hf}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

die() { echo "[jarvis] FATAL: $*" >&2; exit 1; }
need_venv() { [ -x "$PY" ] || die "run 'bash scripts/jarvis_t12.sh setup' first"; }

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
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$PWD is not a git checkout; git clone the repo on the VM"
    local crlf
    crlf="$(git ls-files --eol | grep -c 'w/crlf' || true)"
    [ "$crlf" = 0 ] || die "$crlf tracked files have CRLF endings (core.autocrlf?); re-clone on the VM so the manifests keep their frozen sha256"
}

# verify_inputs TASK|all [tasks]: the files of $INPUTS that TASK needs exist with the recorded
# sha256.  "tasks" checks the task file and split manifest only (what `check` needs).
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
    [ "$bad" = 0 ] || die "inputs missing or different from Ada's; upload them (upload-inputs) and see $INPUTS"
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

# build_args full|light TASK HALF GRID: validate, verify the inputs and set ARGS.  `full` (the
# sweep, the calibration) also needs the calibrator and the PC-Sampler background text; a
# missing reserve file is an error, never the silent position-only variant.
build_args() {
    check_precision
    check_checkout
    [ -f "$4" ] || die "grid $4 not found (the policy grid JSON, relative to the repo)"
    [ -s "$TASKS/${2}_test.jsonl" ] || die "$TASKS/${2}_test.jsonl missing (run setup)"
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

# Start worker $1 in the background: shard $1 of one task per GPU, or (calibrate) one task each.
launch_worker() {
    local i="$1" wout wlog
    if [ "${CAL:-0}" = 1 ]; then
        local -n wargs="CAL_ARGS_$i"
        wout="runs/t12/cal-${CAL_TASKS[$i]}"
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
    echo "[run] worker $i: GPU ${GPU_IDS[$i]}, pid $!, log $wlog"
}

# One run_t12 per GPU (launch_worker), a wall-clock guard, each exit code reported.
supervise() {
    mkdir -p "$OUT"
    local n=${#GPU_IDS[@]} i p rc t0 secs dl timed_out=0 alive pids=()
    WORST=0
    t0=$(date +%s)
    secs=$(awk -v h="$MAX_HOURS" 'BEGIN { printf "%d", h * 3600 }')
    dl=$((t0 + secs))
    echo "$dl" > "$OUT/deadline"
    for ((i = 0; i < n; i++)); do
        launch_worker "$i"
        pids+=($!)
    done
    trap 'kill -TERM "${pids[@]}" 2>/dev/null; exit 143' TERM INT
    echo "[run] $n worker(s); MAX_HOURS=$MAX_HOURS, stopping at $(date -d "@$dl" '+%F %T')"
    while :; do
        alive=0
        for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
        [ "$alive" = 1 ] || break
        if [ "$(date +%s)" -ge "$dl" ]; then
            timed_out=1
            echo "[run] MAX_HOURS=$MAX_HOURS reached: stopping the workers (every finished record is kept)"
            kill -TERM "${pids[@]}" 2>/dev/null || true
            sleep 30
            kill -KILL "${pids[@]}" 2>/dev/null || true
            break
        fi
        sleep "${POLL_SECONDS:-10}"
    done
    for ((i = 0; i < n; i++)); do
        rc=0
        wait "${pids[$i]}" || rc=$?
        echo "[run] worker $i (GPU ${GPU_IDS[$i]}) exit code $rc"
        [ "$rc" -le "$WORST" ] || WORST=$rc
    done
    if [ "$timed_out" = 1 ]; then
        echo "[run] stopped by the budget; rerun the same command to resume"
        WORST=0
    fi
    trap - TERM INT
}

# sync_out TAG: tarball plus the run directory to SYNC_DEST; returns 0 only if both arrived.
sync_out() {
    local out="runs/t12/$1"
    tar -czf "$out.tgz" -C runs/t12 "$1"
    echo "[sync] $out.tgz ($(du -h "$out.tgz" | cut -f1))"
    if [ -z "${SYNC_DEST:-}" ]; then
        echo "[sync] SYNC_DEST is not set: nothing has left the VM"
        return 1
    fi
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --partial -e "ssh $SSH_OPTS" "$out/" "$SYNC_DEST/$1/" &&
            rsync -a --partial -e "ssh $SSH_OPTS" "$out.tgz" "$SYNC_DEST/" || return 1
    else
        # shellcheck disable=SC2086
        scp -r $SSH_OPTS "$out" "$out.tgz" "$SYNC_DEST/" || return 1
    fi
    echo "[sync] copied to $SYNC_DEST"
}

# The end of a run: sync, then pause (opt-in) or tell the user to.  Never destroys.
finish() {
    local synced=0 note
    sync_out "$TAG" && synced=1 || true
    if [ "$AUTO_PAUSE" != 1 ]; then
        note="AUTO_PAUSE is off"
    elif [ "$synced" != 1 ]; then
        note="AUTO_PAUSE refused: the outputs are not synced (set SYNC_DEST)"
    elif [ -z "${JL_API_KEY:-}" ] || [ -z "${JL_MACHINE_ID:-}" ]; then
        note="AUTO_PAUSE refused: JL_API_KEY and JL_MACHINE_ID must be set"
    elif [ ! -x "$VENV/bin/jl" ]; then
        note="AUTO_PAUSE refused: no $VENV/bin/jl (rerun setup with AUTO_PAUSE=1)"
    else
        note="pausing instance $JL_MACHINE_ID now (jl pause --yes)"
    fi
    cat <<EOF
================================================================================
 T12 '$TAG' IS DONE OR STOPPED.  THE INSTANCE KEEPS BILLING (416 INR/h) UNTIL PAUSED.
 outputs synced off the VM: $([ "$synced" = 1 ] && echo yes || echo NO)   [$note]
 If this VM is still up: PAUSE IT NOW, in the dashboard or from your laptop:
     jl download <id> $PWD/runs/t12/$TAG.tgz .      # if not synced
     jl pause <id> --yes
 (pause keeps /home; do not destroy until the outputs are checked.)
================================================================================
EOF
    case "$note" in pausing*) "$VENV/bin/jl" pause "$JL_MACHINE_ID" --yes || echo "[finish] jl pause FAILED: pause in the dashboard" ;; esac
}

case "${1:?usage: jarvis_t12.sh setup|upload-inputs|verify-inputs|verify-gpu|check|run|status|sync|pack}" in
upload-inputs)
    # On the laptop: scp the staged Ada-only inputs (artifacts/c3, runs/corpus) into the VM's
    # checkout, then verify them there.  Alternative: `jl upload <id> <stage>/artifacts <repo>/`.
    VM="${VM:?VM=user@host}" VM_REPO="${VM_REPO:?VM_REPO=/home/.../ANLP-Project}"
    case "${STAGE:-jarvis-stage}" in /*|[A-Za-z]:*) S="${STAGE:-jarvis-stage}" ;; *) S="$ORIG_PWD/${STAGE:-jarvis-stage}" ;; esac
    [ -d "$S/artifacts" ] && [ -d "$S/runs" ] || die "STAGE=$S needs artifacts/ and runs/"
    # shellcheck disable=SC2086
    scp -r $SSH_OPTS -P "${VM_PORT:-22}" "$S/artifacts" "$S/runs" "$VM:$VM_REPO/"
    # shellcheck disable=SC2086
    ssh $SSH_OPTS -p "${VM_PORT:-22}" "$VM" "cd '$VM_REPO' && bash scripts/jarvis_t12.sh verify-inputs"
    ;;
verify-inputs)
    verify_inputs "${2:-all}"
    ;;
setup)
    check_precision
    check_home
    check_checkout
    python3 -m venv "$VENV"
    "$PY" -m pip install -q --upgrade pip
    # torch >= 2.7 from the CUDA 12.8 index: the cu124 wheels have no sm_120 (Blackwell)
    # kernels, so setup would pass and the run would die at the first kernel.  The rest as
    # pinned in pyproject.toml [models] (transformers==4.46.3: do not move it).
    "$PY" -m pip install -q "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
    "$PY" -m pip install -q -e ".[models]"
    [ "$AUTO_PAUSE" != 1 ] || "$PY" -m pip install -q jarvislabs
    # Fail fast, before the 16 GB model download is paid for.
    bash "$0" verify-gpu
    mkdir -p "$TASKS" runs/t12
    # The task files at the pinned hub revision, unless uploaded already (Ada's gsm8k file is
    # the one in $INPUTS; `run` refuses a different one).
    for t in gsm8k mbpp; do
        [ -s "$TASKS/${t}_test.jsonl" ] || "$PY" scripts/prepare_tasks.py --tasks "$t" --out-dir "$TASKS"
    done
    # The pinned LLaDA-8B-Instruct revision (c3.traces.models.KNOWN_MODELS), ~16 GB, once.
    "$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
from c3.traces.models import KNOWN_MODELS
spec = KNOWN_MODELS["llada-8b-instruct"]
path = snapshot_download(spec["path"], revision=spec["revision"])
print("[setup] model at", path)
PYEOF
    df -h . | tail -n 1
    echo "[setup] done; HF_HOME=$HF_HOME. Next: upload-inputs (from the laptop), check, run."
    ;;
verify-gpu)
    need_venv
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true
    "$PY" - <<'PYEOF'
import sys, torch
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
cu = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
print("[gpu] torch", torch.__version__, "cuda", torch.version.cuda, "arches", torch.cuda.get_arch_list())
if v < (2, 7) or cu < (12, 8):
    sys.exit("[gpu] FATAL: need torch >= 2.7 built for CUDA >= 12.8 (Blackwell sm_120)")
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
    ;;
check)
    # Timing and sanity at Ada sanity run 38707's settings: fit-half GSM8K, lm-eval 5-shot,
    # top-k k=1 and threshold 0.9, fp16, the first CHECK_LIMIT fit-half problems (the same
    # first problems as Ada's, so they compare one to one).  Never touches the report half.
    need_venv
    CHECK_LIMIT="${CHECK_LIMIT:-16}"
    [ "$CHECK_LIMIT" -ge 1 ] && [ "$CHECK_LIMIT" -le 40 ] || die "CHECK_LIMIT must be 1..40"
    TAG="check-$(date +%Y%m%d-%H%M%S)" TASK=gsm8k HALF=fit GRID=prereg/t12_sanity.json
    EXTRA=(--limit "$CHECK_LIMIT" --prompt lmeval --fewshot 5)
    MAX_HOURS="${CHECK_MAX_HOURS:-0.5}"
    prepare_run light
    supervise
    "$PY" - "$OUT" "artifacts/split_manifest_gsm8k.json" <<'PYEOF'
import json, sys
from collections import defaultdict
from pathlib import Path
out, manifest = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
a = defaultdict(lambda: defaultdict(float))
for p in sorted(out.glob("t12.shard*.jsonl")):
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        assert int(manifest[str(r["doc_id"])]) == 0, f"report-half problem {r['doc_id']} in a check"
        c = a[r["config"]]
        c["n"] += 1
        if r.get("error"):
            c["err"] += 1
        else:
            c["fwd"] += r["n_forward"]; c["sec"] += r["seconds"]; c["tps"] += r["tokens_per_step"]
for k, c in sorted(a.items()):
    ok = max(c["n"] - c["err"], 1)
    print(f"[check] {k:28s} {int(c['n']):3d} problems {int(c['err'])} errors  "
          f"{c['sec'] / max(c['fwd'], 1):.3f} s/forward  {c['tps'] / ok:.2f} tok/step  "
          f"{c['sec'] / ok:.1f} s/problem")
print("[check] Ada (job 38707, 2x2080 Ti, 5-shot): 0.80 s/forward; published 79.3 (k=1) / 79.2 (tau=0.9)")
PYEOF
    "$PY" scripts/score_t12.py "$OUT" --task-jsonl "gsm8k=$TASKS/gsm8k_test.jsonl" --out "$OUT/score.json"
    echo "[check] a handful of problems is noisy for accuracy; compare s/forward, then go on or pause."
    ;;
calibrate)
    # Fit-half-only knob calibration (CAL_GRID, e.g. the t12_cal.json draft): GSM8K, the first
    # 32 fit-half problems, on GPU 0 and MBPP, the first 24, on GPU 1; zero-shot chat, fp16,
    # one shard each, 0.7 h hard cap (problems are the outer loop, so a stop leaves every
    # configuration on the same problems).  Then PICK_KNOBS on both outputs.  Never touches
    # the report half; no sync, no pause.
    need_venv
    CAL_GRID="${CAL_GRID:?set CAL_GRID=<calibration grid JSON>}"
    PICK_KNOBS="${PICK_KNOBS:-scripts/pick_knobs.py}"
    [ -f "$PICK_KNOBS" ] || die "PICK_KNOBS=$PICK_KNOBS not found (pick_knobs.py from the grid draft)"
    MAX_HOURS="${CAL_MAX_HOURS:-0.7}"
    awk -v h="$MAX_HOURS" 'BEGIN { exit !(h > 0 && h <= 0.7) }' || die "CAL_MAX_HOURS=$MAX_HOURS: the calibration cap is 0.7 h"
    detect_gpus
    [ "${#GPU_IDS[@]}" -ge 2 ] || die "calibrate needs 2 GPUs (one task each), have ${#GPU_IDS[@]}"
    GPU_IDS=("${GPU_IDS[@]:0:2}")
    CAL=1 CAL_TASKS=(gsm8k mbpp) OUT=runs/t12/cal
    build_args full gsm8k fit "$CAL_GRID"
    # shellcheck disable=SC2034  # read by launch_worker through a nameref
    CAL_ARGS_0=("${ARGS[@]}" --limit 32)
    build_args full mbpp fit "$CAL_GRID"
    # shellcheck disable=SC2034
    CAL_ARGS_1=("${ARGS[@]}" --limit 24)
    supervise
    "$PY" -I "$PICK_KNOBS" runs/t12/cal-gsm8k runs/t12/cal-mbpp
    echo "[calibrate] paste the picks into the grid (by speed only), then run."
    ;;
run)
    need_venv
    check_home
    TAG="${2:?tag}" TASK="${3:?task (gsm8k|mbpp)}" HALF="${4:?half (fit|report|all)}"
    GRID="${GRID:?set GRID=<grid JSON>, e.g. GRID=prereg/t12_grid.json}"
    shift 4
    EXTRA=("$@")
    prepare_run full
    mkdir -p "$OUT"
    if [ "${RUN_FG:-0}" = 1 ]; then
        supervise
        finish
        exit "$WORST"
    fi
    nohup bash "$0" _supervise "$TAG" "$TASK" "$HALF" "${EXTRA[@]}" >> "$OUT/supervisor.log" 2>&1 &
    echo "[run] supervisor pid $! on ${#GPU_IDS[@]} GPU(s) -> $OUT (tail -f $OUT/supervisor.log)"
    ;;
_supervise)
    TAG="$2" TASK="$3" HALF="$4"
    shift 4
    EXTRA=("$@")
    prepare_run full
    supervise
    finish
    exit "$WORST"
    ;;
status)
    need_venv
    OUT="runs/t12/${2:?tag}"
    tail -n 3 "$OUT/supervisor.log" 2>/dev/null || true
    for f in "$OUT"/worker*.log; do
        [ -e "$f" ] && { echo "== $f"; tail -n 2 "$f"; }
    done
    [ ! -s "$OUT/deadline" ] || echo "budget left: $(( ($(cat "$OUT/deadline") - $(date +%s)) / 60 )) min"
    "$PY" - "$OUT" <<'PYEOF'
import json, sys
from collections import defaultdict
from pathlib import Path
n, err, tps, secs = defaultdict(int), defaultdict(int), defaultdict(float), defaultdict(float)
for p in sorted(Path(sys.argv[1]).glob("t12.shard*.jsonl")):
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = r["config"]
        n[k] += 1
        if r.get("error"):
            err[k] += 1
        else:
            tps[k] += r["tokens_per_step"]
            secs[k] += r["seconds"]
for k in sorted(n):
    ok = n[k] - err[k]
    print(f"{k:40s} {n[k]:5d} records  {err[k]:3d} errors  "
          f"{tps[k] / max(ok, 1):6.2f} tok/step  {secs[k] / max(ok, 1):6.1f} s/problem")
print(f"total {sum(n.values())} records, {sum(secs.values()) / 3600:.2f} GPU-h of decoding")
PYEOF
    ;;
sync)
    TAG="${2:?tag}"
    sync_out "$TAG"
    ;;
pack)
    OUT="runs/t12/${2:?tag}"
    tar -czf "$OUT.tgz" -C runs/t12 "$(basename "$OUT")"
    echo "[pack] $OUT.tgz ($(du -h "$OUT.tgz" | cut -f1)); copy it off the instance"
    ;;
*)
    echo "unknown subcommand $1"; exit 2 ;;
esac
