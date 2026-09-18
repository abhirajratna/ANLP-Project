#!/bin/bash
# Throughput benchmark at production settings, to set the token budget from a
# measurement rather than an assumed TFLOP/s figure.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
NGPU="${BENCH_NGPU:-2}"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
cd "$REPO_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

SM="$C3_SCRATCH/bench"
mkdir -p "$SM"
if [ ! -f "$SM/index.json" ]; then
  C3_BENCH_DIR="$SM" "$C3_PY" - <<'PY'
import numpy as np, json, os
d = os.environ["C3_BENCH_DIR"]
a = np.random.default_rng(0).integers(0, 50256, 8_000_000).astype(np.uint16)
p = f"{d}/train_0000.bin"; a.tofile(p)
json.dump({"total_tokens": int(a.size), "documents": 1,
           "calibration_documents_skipped": 1, "seq_len": 256,
           "shards": [{"path": p, "tokens": int(a.size)}]}, open(f"{d}/index.json","w"))
PY
fi

# ckpt-every is set beyond the run length: the smoke test wrote a ~1.4 GB
# checkpoint every 10 steps, which measured disk I/O rather than training.
"$(dirname "$C3_PY")/torchrun" --nproc_per_node="$NGPU" --nnodes=1 scripts/train_110m.py \
  --index "$SM/index.json" --out-dir "$SM/ckpt" \
  --max-tokens "${BENCH_TOKENS:-13000000}" \
  --batch-size "${C3_BATCH:-32}" --grad-accum "${C3_ACCUM:-4}" \
  --ckpt-every 1000000 2>&1 | grep -E "step |fresh start|no bf16|GPU-hours|days" | tail -12
