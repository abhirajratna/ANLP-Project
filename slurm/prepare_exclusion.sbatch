#!/bin/bash
#SBATCH --job-name=c3-exclusion
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home2/%u/c3/logs/exclusion-%j.out
#
# Step 1 of T3 (see slurm/T3_RUNBOOK.md). Reads the local FineWeb copy, no GPU.
#
#  1. Samples 500 reserve held-out documents (v2 pilot + replacements) from the
#     same 200k-document pool as the calibration corpus, skipping its documents.
#  2. Builds the v2 exclusion manifest over the main 2,000 and the reserve:
#     FineWeb id + raw-text hash + first-256-GPT-2-token hash per document.
#
# The manifest is also copied into the repo's artifacts/ so it can be committed
# as the record of what was held out.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
CORPUS="$PERSIST/corpus"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
export HF_HOME="${C3_SCRATCH:-/ssd_scratch/$USER}/hf"
mkdir -p "$HF_HOME"
cd "$REPO_DIR"

[ -f "$CORPUS/calibration_docs.jsonl" ] || { echo "FATAL: $CORPUS/calibration_docs.jsonl missing"; exit 1; }

# The corpus must be exactly the committed held-out record (a corpus regenerated
# on another account could differ if the upstream stream order changed).
"$C3_PY" scripts/check_corpus.py --docs "$CORPUS/calibration_docs.jsonl" \
    --manifest artifacts/exclusion_manifest.json \
    || { echo "FATAL: corpus does not match artifacts/exclusion_manifest.json"; exit 1; }

if [ ! -f "$CORPUS/reserve_docs.jsonl" ]; then
    c3_stage_fineweb 2      # the same 200k-document pool as the corpus
    "$C3_PY" scripts/sample_reserve_docs.py \
        --calibration "$CORPUS/calibration_docs.jsonl" \
        --out "$CORPUS/reserve_docs.jsonl" \
        --n-documents "${C3_N_RESERVE:-500}"
else
    echo "[exclusion] reusing $CORPUS/reserve_docs.jsonl"
fi

"$C3_PY" scripts/build_exclusion.py \
    main="$CORPUS/calibration_docs.jsonl" \
    reserve="$CORPUS/reserve_docs.jsonl" \
    --out "$CORPUS/exclusion_v2.json"

cp "$CORPUS/exclusion_v2.json" "$REPO_DIR/artifacts/exclusion_v2.json"
echo "[exclusion] copied to $REPO_DIR/artifacts/exclusion_v2.json -- commit it"
