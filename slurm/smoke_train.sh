#!/bin/bash
#SBATCH --job-name=c3-smoke
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:2
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069,gnode076
#SBATCH --cpus-per-task=16
#SBATCH --mem=48000M
#SBATCH --time=01:30:00
#SBATCH --output=/home2/%u/c3/logs/smoke-%j.out
# Smoke test for the 110M training loop: exercises the model, MDLM loss, AMP,
# DDP, validation, EMA and checkpoint/resume on synthetic document-aligned
# shards, so a defect surfaces in minutes rather than after the real corpus is
# built.  Submit it with sbatch, or run it inside an interactive 2-GPU allocation.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
cd "$REPO_DIR"
echo "NODE=$(hostname)"

SM="$C3_SCRATCH/smoke"
rm -rf "$SM"; mkdir -p "$SM"
C3_SMOKE_DIR="$SM" "$C3_PY" - <<'PY'
import json, os
import numpy as np
d = os.environ["C3_SMOKE_DIR"]
rng = np.random.default_rng(0)

def shard(prefix, k, n_docs):
    lens = rng.integers(40, 900, n_docs)
    toks = rng.integers(0, 50256, int(lens.sum())).astype(np.uint16)
    p, o = f"{d}/{prefix}_{k:04d}.bin", f"{d}/{prefix}_{k:04d}.offsets.npy"
    toks.tofile(p)
    np.save(o, np.r_[0, np.cumsum(lens)].astype(np.int64))
    return {"path": p, "offsets": o, "tokens": int(toks.size), "documents": int(n_docs)}

train = [shard("train", k, 4000) for k in range(2)]
val = [shard("val", 0, 200)]
json.dump({"format": "doc-offsets-v2", "total_tokens": sum(s["tokens"] for s in train),
           "documents": 8000, "val_documents": 200, "val_fraction": 0.0005,
           "calibration_documents_skipped": 0, "seq_len": 256,
           "shards": train, "val_shards": val}, open(f"{d}/index.json", "w"))
print("smoke shards ready")
PY

TORCHRUN="$(dirname "$C3_PY")/torchrun"
ARGS=(--index "$SM/index.json" --out-dir "$SM/ckpt" --max-tokens 2000000 --batch-size 16
      --grad-accum 1 --ckpt-every 10 --eval-every 10 --eval-batches 2 --warmup-steps 5)
# Full output of each run goes to logs/; only its tail is echoed here, and on a
# failure the tail is long enough to hold the traceback.
LOG="$REPO_DIR/logs/smoke-${SLURM_JOB_ID:-$$}"
echo "=== run 1: fresh start, 2 GPUs (killed after 60 s to force a resume) ==="
timeout 60 "$TORCHRUN" --nproc_per_node=2 --nnodes=1 scripts/train_110m.py "${ARGS[@]}" \
    > "$LOG-run1.log" 2>&1 || true
grep -E "^\[train\]|Error|error" "$LOG-run1.log" | tail -8 || true
echo "=== run 2: resume from checkpoint to completion ==="
if ! "$TORCHRUN" --nproc_per_node=2 --nnodes=1 scripts/train_110m.py "${ARGS[@]}" \
        > "$LOG-run2.log" 2>&1; then
    tail -n 60 "$LOG-run2.log"
    echo "=== run 2 FAILED; full output: $LOG-run2.log (run 1: $LOG-run1.log) ==="
    exit 1
fi
tail -n 12 "$LOG-run2.log"
"$C3_PY" scripts/check_g2.py --history "$SM/ckpt/history.jsonl" --out "$SM/g2.json" | tail -3
echo "=== smoke test done ==="
