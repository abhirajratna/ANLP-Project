#!/bin/bash
#SBATCH --job-name=c3-hf
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home2/%u/c3/logs/hf-%j.out
#
# Publish the 110M models and the C1 traces to the HuggingFace Hub (no GPU), private.
#
#     sbatch --export=ALL,C3_HF_ONLY=models slurm/hf_upload.sh
#     sbatch --export=ALL,C3_HF_ONLY=traces --dependency=afterany:<models job> slurm/hf_upload.sh
#
# The traces are ~20 GB: at Ada's ~50-120 kB/s, 2-3 days, longer than the 2-day
# walltime.  Queue a resume behind the traces job when you submit it, and another
# while the log lacks the final "[hf] ...c3-traces: N files" line:
#
#     sbatch --export=ALL,C3_HF_ONLY=traces --dependency=afterany:<traces job> slurm/hf_upload.sh
#
# A resume skips what is already on the Hub (on another node it re-pulls and re-hashes
# first, ~30 min); a job that finds everything committed just exits.
#
# Auth: the token in ~/.cache/huggingface/token on Ada (the user's own, mode 600).
# /share1 is not mounted on compute nodes, so every file is pulled to node SSD first,
# as in the other jobs.  Re-running is safe: model uploads replace the same files, and
# upload_large_folder resumes.  Ada's uplink to the Hub is slow (~50 kB/s measured on
# 15 Oct 2026) and Xet uploads died on a network error there (job 22429), so the
# job uses plain LFS unless C3_HF_XET=1.
#
# C3_HF_ONLY=models|traces runs one half.  Repos are created private; flip them in
# the org settings (or `HfApi().update_repo_settings(repo, private=False)`).
set -euo pipefail
[ "${C3_HF_XET:-0}" = 1 ] || export HF_HUB_DISABLE_XET=1
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
BLOBS="${C3_BLOBS:-/share1/$USER}"
ORG="${C3_HF_ORG:-ANLP-project}"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
cd "$REPO_DIR"
STAGE="$C3_SCRATCH/hf_stage"
mkdir -p "$STAGE"
pull() {  # pull <dest dir> [--rsync-option]... <path on /share1 or /home2>...   (missing sources are skipped)
    local dest="$1" opts=(); shift
    while [[ "${1:-}" == --* ]]; do opts+=("$1"); shift; done
    mkdir -p "$dest"
    rsync -a --ignore-missing-args "${opts[@]}" -e "ssh $C3_SSH_OPTS" \
        $(printf "$USER@$C3_LOGIN_HOST:%s " "$@") "$dest/"
}
"$C3_PY" -c "from huggingface_hub import whoami; print('[hf] authenticated as', whoami()['name'])"

model() {  # model <run dir under $BLOBS> <repo name> <eval json or ''> <card note>
    local d="$STAGE/models/$2"
    pull "$d" "$BLOBS/$1/final.pt" "$BLOBS/$1/train_summary.json" "$BLOBS/$1/history.jsonl"
    "$C3_PY" scripts/upload_hf.py --ckpt "$d/final.pt" --repo "$ORG/$2" \
        ${3:+--eval "$3"} --note "$4"
}

if [ "${C3_HF_ONLY:-models}" = models ]; then
    model v2/train110m-v3 c3-mdlm-110m artifacts/eval_110m_train110m-v3.json \
        "The plan-v2 110M (run \`train110m-v3\`, W&B \`train110m-v3-oct12\`): the model the C1 traces and the T9 probe use. Gate G2 passed (artifacts/gate_g2_train110m-v3.json)."
    model train110m c3-mdlm-110m-v1 artifacts/eval_110m_train110m.json \
        "The first 110M (v1, W&B \`train110m\`), used in the mid submission and the T9 dry run. Superseded by $ORG/c3-mdlm-110m."
    model probe/ft110m-v3 c3-ft110m-v3 "" \
        "T9 fine-tune (a): $ORG/c3-mdlm-110m fine-tuned for 100M tokens on the synthetic JSON probe corpus with a rho marker (W&B \`ft110m-v3\`)."
    model probe/ft110m-v3-visible c3-ft110m-v3-visible "" \
        "T9 fine-tune (b): as (a), but the rho marker is never masked (visible 8-token prefix), 300M tokens (W&B \`ft110m-v3-visible\`)."
    model probe/ft110m-v1-dryrun c3-ft110m-v1-dryrun "" \
        "T9 pipeline dry run: $ORG/c3-mdlm-110m-v1 fine-tuned for 100M tokens on the probe corpus (W&B \`ft110m-v1-dryrun\`)."
    rm -rf "$STAGE/models"
fi

if [ "${C3_HF_ONLY:-traces}" = traces ]; then
    T="$STAGE/c3-traces"
    pull "$T/main-110m-v3" "$BLOBS/traces/main-110m-v3/"
    # Parts, shard meta and manifest, as for main-110m-v3.  The merged copy next to them
    # (timesplit-110m-v3.parquet + .meta.json, 3.5 GB) duplicates the parts.
    pull "$T/timesplit-110m-v3" --exclude=timesplit-110m-v3.parquet \
        --exclude=timesplit-110m-v3.meta.json "$BLOBS/traces/timesplit-110m-v3/"
    pull "$T/t2floor-110m-v3-fp32" "$BLOBS/traces/t2floor-110m-v3-fp32/"
    pull "$T/t2pilot-llada-base-fp16" "$BLOBS/traces/t2pilot-llada-base-fp16/"
    pull "$T/stage0-llada-8b-4bit" "$C3_PERSIST/traces/"
    pull "$T/corpus" "$C3_PERSIST/corpus/"
    mkdir -p "$T/validation" && cp artifacts/traces_110m/*.json artifacts/c1/main-110m-v3.json \
        artifacts/c1/timesplit-110m-v3.json artifacts/t2_pilot/floor_local-110m-v3_fp32.json "$T/validation/"
    cat > "$T/README.md" <<EOF
---
license: odc-by
tags: [calibration, diffusion-language-model, masked-diffusion]
---

# C3 audit traces

Position-level traces from the C3 selection-conditional calibration audit
(ANLP Monsoon 2026, team DuckDuckDuckGoose): one row per (masked state, masked
position), schema v2 -- \`model variant revision precision dataset git_commit seed
mask_kind policy k target_rate mask_rate step n_masked doc_id state_id state_idx
doc_len span_start span_len position top1 conf ell p_true entropy margin topk_ids
topk_p delta_ell agree committed commit_rank\` (\`c3/traces/collect.py\`). Pools are
drawn at analysis time, so the audit can be re-run with other pool designs without
another forward pass.

| path | what |
|---|---|
| \`main-110m-v3/\` | $ORG/c3-mdlm-110m on 1,979 documents of the 2,000-document calibration set, fp32; 4 shards x 10 parts, \`*.meta.json\` per shard, and \`main-110m-v3.dataset.json\`, a zero-copy manifest of the parts (file, sha256, rows) that \`c3.assemble.load_traces\` reads directly: no merge step needed. Shards 0-1 and 2-3 were collected at two commits that differ only outside the trace code (validated with \`--allow-mixed-commits\`) |
| \`timesplit-110m-v3/\` | the same model on the 500-document time-split set (doc_ids 200000-200499), fp32, in the same layout: 1 shard x 10 parts, its \`*.meta.json\` and the manifest \`timesplit-110m-v3.dataset.json\` |
| \`t2floor-110m-v3-fp32/\` | non-determinism floor of $ORG/c3-mdlm-110m: 20 reserve documents, T2 pilot design, fp32; 5 bitwise-identical repeats, floor 0 |
| \`validation/\` | \`main-110m-v3.json\`, \`timesplit-110m-v3.json\`: \`scripts/validate_run.py\` on each manifest (every check passes); \`floor_local-110m-v3_fp32.json\`: the floor; \`validate_*.json\`, \`*.meta.json\`: metadata and row-count validation of the earlier merged copies against the closed-form design |
| \`t2pilot-llada-base-fp16/\` | T2 pilot: LLaDA-8B-Base fp16, 5 repeats (non-determinism floor) |
| \`stage0-llada-8b-4bit/\` | Stage 0 pilot: LLaDA-8B 4-bit, 20 documents |
| \`corpus/\` | calibration, reserve and time-split documents (FineWeb-Edu, ODC-By) and the exclusion manifests removed from the 110M training stream |

Code: https://github.com/abhirajratna/ANLP-Project
EOF
    "$C3_PY" - "$T" "$ORG/c3-traces" <<'PY'
import sys
from huggingface_hub import HfApi
folder, repo = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
# Not the default workers (cores - 2 = 38 on a gnode): each LFS file's part URLs expire
# 10 h after its upload starts (X-Amz-Expires=36000, job 22476), and 38 ~400 MB files
# sharing ~50-120 kB/s would each need 35-85 h and restart at every expiry.  3 in
# flight need ~7 h at 50 kB/s.
api.upload_large_folder(repo_id=repo, folder_path=folder, repo_type="dataset", private=True,
                        num_workers=3)
files = api.list_repo_files(repo, repo_type="dataset")
print(f"[hf] {repo}: {len(files)} files -> https://huggingface.co/datasets/{repo}")
PY
fi
echo "$(date +%F_%T) hf=${SLURM_JOB_ID:-?} models+traces -> https://huggingface.co/$ORG (private)" >> logs/t3_jobs.txt
