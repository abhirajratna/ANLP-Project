#!/bin/bash
# T12 on a Jarvis Labs A100 (plan to the final, Parth 3-5): set up, run, watch, pack.
# Run from the repository checkout on the instance:
#
#   bash scripts/jarvis_t12.sh setup                       # venv, pinned packages, tasks, model
#   bash scripts/jarvis_t12.sh run sanity gsm8k fit prereg/t12_sanity.json --limit 200 --prompt lmeval
#   bash scripts/jarvis_t12.sh run sweep-gsm8k gsm8k report <grid> --calibrator artifacts/c3/llada-8b-instruct_fp16_c3a-gsm8k.json
#   bash scripts/jarvis_t12.sh status sweep-gsm8k          # records per configuration, speed so far
#   bash scripts/jarvis_t12.sh pack sweep-gsm8k            # runs/t12/sweep-gsm8k.tgz to copy off
#
# Runs go to runs/t12/<tag>/ and are resumable: rerunning the same `run` line after a
# stopped instance skips every (problem, configuration) already recorded (scripts/run_t12.py
# refuses to mix runs).  Keep runs/ on the instance's persistent storage, `pack` and copy
# the tarball off at least daily, and pause the instance whenever nothing is running: the
# credit is $100.  Precision is fp16, as the C3a calibrator's traces and the Ada overflow;
# an A100 runs fp16 as fast as bf16.  PC-Sampler runs position-only unless
# runs/corpus/reserve_docs.jsonl is present (copy it from Ada: /home2/$USER/c3-runs/corpus/).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${C3_VENV:-.venv-t12}"
PY="$VENV/bin/python"
TASKS=runs/corpus/tasks
export HF_HOME="${HF_HOME:-$PWD/runs/hf}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

need_venv() { [ -x "$PY" ] || { echo "run 'bash scripts/jarvis_t12.sh setup' first"; exit 1; }; }

case "${1:?usage: jarvis_t12.sh setup|run|status|pack}" in
setup)
    python3 -m venv "$VENV"
    "$PY" -m pip install -q --upgrade pip
    # torch from the CUDA 12.4 wheel index; the rest as pinned in pyproject.toml [models].
    "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
    "$PY" -m pip install -q -e ".[models]"
    "$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; \
print('[setup] torch', torch.__version__, torch.cuda.get_device_name(0))"
    mkdir -p "$TASKS" runs/t12
    "$PY" scripts/prepare_tasks.py --tasks gsm8k mbpp --out-dir "$TASKS"
    # The pinned LLaDA-8B-Instruct revision (c3.traces.models.KNOWN_MODELS), ~16 GB, once.
    "$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
from c3.traces.models import KNOWN_MODELS
spec = KNOWN_MODELS["llada-8b-instruct"]
path = snapshot_download(spec["path"], revision=spec["revision"])
print("[setup] model at", path)
PYEOF
    echo "[setup] done; HF_HOME=$HF_HOME"
    ;;
run)
    need_venv
    TAG="${2:?tag}" TASK="${3:?task (gsm8k|mbpp)}" HALF="${4:?half (fit|report|all)}"
    GRID="${5:?grid JSON}"
    shift 5
    OUT="runs/t12/$TAG"
    mkdir -p "$OUT"
    UNIGRAM=()
    [ -s runs/corpus/reserve_docs.jsonl ] && UNIGRAM=(--unigram-docs runs/corpus/reserve_docs.jsonl)
    nohup "$PY" scripts/run_t12.py --model llada-8b-instruct --precision fp16 --device cuda \
        --task "$TASK" --task-jsonl "$TASKS/${TASK}_test.jsonl" \
        --manifest "artifacts/split_manifest_${TASK}.json" --half "$HALF" --configs "$GRID" \
        "${UNIGRAM[@]}" --label "jarvis-$TAG" --out-dir "$OUT" "$@" >> "$OUT/run.log" 2>&1 &
    echo "[run] pid $! -> $OUT (tail -f $OUT/run.log)"
    ;;
status)
    need_venv
    OUT="runs/t12/${2:?tag}"
    tail -n 3 "$OUT/run.log" 2>/dev/null || true
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
pack)
    OUT="runs/t12/${2:?tag}"
    tar -czf "$OUT.tgz" -C runs/t12 "$(basename "$OUT")"
    echo "[pack] $OUT.tgz ($(du -h "$OUT.tgz" | cut -f1)); copy it off the instance"
    ;;
*)
    echo "unknown subcommand $1"; exit 2 ;;
esac
