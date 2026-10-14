#!/bin/bash
#SBATCH --job-name=c3-corpus
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M
#SBATCH --time=06:00:00
#SBATCH --output=/home2/%u/c3/logs/corpus-%j.out
#
# Samples the 2,000-document held-out calibration corpus from the local FineWeb
# copy (slurm/fetch_fineweb.sbatch), then checks it is exactly the committed
# record, artifacts/exclusion_manifest.json. No GPU.
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"

export HF_HOME="${C3_SCRATCH:-/ssd_scratch/$USER}/hf"
mkdir -p "$HF_HOME" "$PERSIST/corpus"
cd "$REPO_DIR"
# The sample reads only the stream's first 200k qualifying documents; if the
# staged files ever held fewer, the sampler stops with an error.
c3_stage_fineweb 2

"$C3_PY" scripts/prepare_corpus.py \
    --n-documents "${C3_N_DOCS:-2000}" \
    --max-tokens 256 \
    --pool "${C3_POOL:-200000}" \
    --out-dir "$PERSIST/corpus"

"$C3_PY" scripts/check_corpus.py --docs "$PERSIST/corpus/calibration_docs.jsonl" \
    --manifest artifacts/exclusion_manifest.json
