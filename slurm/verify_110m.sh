#!/bin/bash
# Confirm the trained 110M checkpoint works as an audit instrument: it should
# infill masked positions above chance and produce usable confidence log-odds.
# Then run the full sanity check (perplexity bound + samples) on the held-out
# calibration documents.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
RUN="${C3_RUN_NAME:-train110m-v2}"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
cd "$REPO_DIR"

# Always pull: a final.pt left on this node by an earlier run would otherwise be
# verified in place of the current one.
CK="$C3_SCRATCH/$RUN-final.pt"
rsync -a --partial-dir=.rsync-partial -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
    "$USER@ada:$BLOBS/$RUN/final.pt" "$CK"

C3_CKPT="$CK" C3_DOCS="$PERSIST/corpus/calibration_docs.jsonl" "$C3_PY" - <<'PY'
import os, sys, numpy as np, torch
sys.path.insert(0, ".")
from c3.traces.models import build_model
from c3.data.fineweb import tokenize_documents

m = build_model("local-110m", path=os.environ["C3_CKPT"], device="cuda")
docs = tokenize_documents(os.environ["C3_DOCS"], m.tokenizer, 256)[:40]
print(f"docs: {len(docs)}")

rng = np.random.default_rng(0)
for rate in (0.2, 0.5, 0.9):
    accs, confs = [], []
    for _, x0 in docs:
        mask = rng.random(x0.size) < rate
        if mask.sum() < 4:
            continue
        ids = x0.copy(); ids[mask] = m.mask_token_id
        out = m.score_masked(torch.as_tensor(ids[None], dtype=torch.long).cuda(), mask)
        accs.append((out.top1 == x0[mask]).mean()); confs.append(out.conf.mean())
    print(f"  mask rate {rate:.1f}: accuracy {np.mean(accs):.4f}  mean confidence {np.mean(confs):.4f}")
print("chance accuracy would be ~1/50257 = 0.00002")
PY

"$C3_PY" scripts/eval_110m.py --ckpt "$CK" --docs "$PERSIST/corpus/calibration_docs.jsonl" \
    --out "$PERSIST/eval_110m_$RUN.json"
