#!/bin/bash
#SBATCH --job-name=c3-verify
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home2/%u/c3/logs/verify-%j.out
# Step 5 of T3 (slurm/T3_RUNBOOK.md); submit with sbatch or run in a 1-GPU allocation.
# Confirms the trained 110M checkpoint works as an audit instrument (infills
# masked positions above chance), then writes the sanity checks into artifacts/:
# the held-out perplexity bound with the GPT-2 reference and samples, and gate G2.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
RUN="${C3_RUN_NAME:-train110m-v2}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4
source "$REPO_DIR/slurm/activate_env.sh"
export HF_HOME="$C3_SCRATCH/hf"   # GPT-2 reference download stays off /home2
cd "$REPO_DIR"

# Always pull: a final.pt left on this node by an earlier run would otherwise be
# verified in place of the current one.
CK="$C3_SCRATCH/$RUN-final.pt"
rsync -a --partial-dir=.rsync-partial -e "ssh $SSH_OPTS" "$USER@ada:$BLOBS/$RUN/final.pt" "$CK"
rsync -a -e "ssh $SSH_OPTS" "$USER@ada:$BLOBS/$RUN/history.jsonl" "$C3_SCRATCH/$RUN-history.jsonl"

C3_CKPT="$CK" C3_DOCS="$PERSIST/corpus/calibration_docs.jsonl" "$C3_PY" - <<'PY'
import os, sys, numpy as np, torch
sys.path.insert(0, ".")
from c3.traces.models import build_model
from c3.data.fineweb import tokenize_documents

m = build_model("local-110m", path=os.environ["C3_CKPT"], device="cuda")
if getattr(m.model.cfg, "causal", False):
    print("autoregressive reference: skipping the infilling check")
    sys.exit(0)
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
    --gpt2-reference --out "$REPO_DIR/artifacts/eval_110m_$RUN.json"

# Gate G2 on the validation curve the trainer logged.
"$C3_PY" scripts/check_g2.py --history "$C3_SCRATCH/$RUN-history.jsonl" \
    --out "$REPO_DIR/artifacts/gate_g2_$RUN.json"
cp "$C3_SCRATCH/$RUN-history.jsonl" "$REPO_DIR/artifacts/history_$RUN.jsonl"
echo "[verify] wrote artifacts/eval_110m_$RUN.json, gate_g2_$RUN.json, history_$RUN.jsonl -- commit them"
