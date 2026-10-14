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
if [ ! -f "$SM/index.json" ] || ! grep -q doc-offsets-v2 "$SM/index.json"; then
  C3_BENCH_DIR="$SM" "$C3_PY" - <<'PY'
import numpy as np, json, os
d = os.environ["C3_BENCH_DIR"]
rng = np.random.default_rng(0)
# Document-aligned shards; long documents so windows are full and throughput is
# not flattered by padding.
lens = rng.integers(2000, 4000, 3000)
a = rng.integers(0, 50256, int(lens.sum())).astype(np.uint16)
p, o = f"{d}/train_0000.bin", f"{d}/train_0000.offsets.npy"
a.tofile(p); np.save(o, np.r_[0, np.cumsum(lens)].astype(np.int64))
json.dump({"format": "doc-offsets-v2", "total_tokens": int(a.size), "documents": int(lens.size),
           "val_documents": 0, "val_fraction": 0.0, "calibration_documents_skipped": 0,
           "seq_len": 256, "shards": [{"path": p, "offsets": o, "tokens": int(a.size),
                                       "documents": int(lens.size)}], "val_shards": []},
          open(f"{d}/index.json", "w"))
PY
fi

# A benchmark always starts fresh: a finished checkpoint from an earlier bench
# would be resumed at its last step and measure nothing.
rm -rf "$SM/ckpt"

# ckpt-every is set beyond the run length: the smoke test wrote a ~1.4 GB
# checkpoint every 10 steps, which measured disk I/O rather than training.
"$(dirname "$C3_PY")/torchrun" --nproc_per_node="$NGPU" --nnodes=1 scripts/train_110m.py \
  --index "$SM/index.json" --out-dir "$SM/ckpt" \
  --max-tokens "${BENCH_TOKENS:-13000000}" \
  --batch-size "${C3_BATCH:-16}" --grad-accum "${C3_ACCUM:-8}" \
  --ckpt-every 1000000 --eval-every 0 2>&1 | grep -E "step |fresh start|no bf16|GPU-hours|days" | tail -12
