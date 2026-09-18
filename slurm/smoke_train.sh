#!/bin/bash
# Smoke test for the 110M training loop: exercises the model, MDLM loss, AMP,
# DDP and checkpoint/resume on synthetic shards, so a defect surfaces in minutes
# rather than after the real corpus is built.
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
import numpy as np, json, os
d = os.environ["C3_SMOKE_DIR"]
rng = np.random.default_rng(0)
shards = []
for i in range(2):
    a = rng.integers(0, 50256, 2_000_000).astype(np.uint16)
    p = f"{d}/train_{i:04d}.bin"
    a.tofile(p)
    shards.append({"path": p, "tokens": int(a.size)})
json.dump({"total_tokens": 4_000_000, "documents": 1,
           "calibration_documents_skipped": 1, "seq_len": 256,
           "shards": shards}, open(f"{d}/index.json", "w"))
print("smoke shards ready")
PY

TORCHRUN="$(dirname "$C3_PY")/torchrun"
echo "=== run 1: fresh start, 2 GPUs ==="
"$TORCHRUN" --nproc_per_node=2 --nnodes=1 scripts/train_110m.py \
  --index "$SM/index.json" --out-dir "$SM/ckpt" \
  --max-tokens 1310720 --batch-size 16 --grad-accum 1 --ckpt-every 10 2>&1 | tail -14

echo "=== run 2: resume from checkpoint ==="
"$TORCHRUN" --nproc_per_node=2 --nnodes=1 scripts/train_110m.py \
  --index "$SM/index.json" --out-dir "$SM/ckpt" \
  --max-tokens 2621440 --batch-size 16 --grad-accum 1 --ckpt-every 10 2>&1 | tail -10
echo "=== smoke test done ==="
