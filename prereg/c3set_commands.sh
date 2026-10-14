#!/bin/bash
# The set-level C3 final run on the report half (Phase 8 of the set-level plan;
# prereg/deviations.md items 7-13).  From the runner checkout, at the freeze tag:
#
#     git -C /home2/$USER/c3 checkout c3set-freeze-v4
#     C3_REPORT_HALF_OK=1 bash prereg/c3set_commands.sh
#
# Refuses unless C3_REPORT_HALF_OK=1 (set only after the team's sign-off, recorded as
# deviations.md item 13), HEAD is the tag, the tree is clean, and the calibrators and
# grid match the sha256 in prereg/c3set_frozen_v4.json.  Submits, chained:
#   1. trace analyses on the report half (B and A)          c3set_analysis.sbatch
#   2. infilling with the frozen grid and calibrators        infill_eval.sbatch (4 GPUs)
#   3. the matched-speed analysis                            c3set_analysis.sbatch infill
#   4. one W&B run with everything, config = the frozen JSON wandb_log.sbatch
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${C3_FREEZE_TAG:-c3set-freeze-v4}"
FROZEN=prereg/c3set_frozen_v4.json
[ "${C3_REPORT_HALF_OK:-}" = 1 ] || {
    echo "refusing: the report half is read once, after sign-off; set C3_REPORT_HALF_OK=1"; exit 1; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$TAG^{commit}" 2>/dev/null || echo none)" ] || {
    echo "refusing: HEAD is not $TAG"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "refusing: dirty tree"; exit 1; }
[ -r "$FROZEN" ] || { echo "refusing: $FROZEN missing"; exit 1; }
python3 - "$FROZEN" <<'PY'
import hashlib, json, sys
f = json.load(open(sys.argv[1]))
for item in (f["calibrators"]["token"], f["calibrators"]["set"], f["grid"]):
    got = hashlib.sha256(open(item["path"], "rb").read()).hexdigest()
    if got != item["sha256"]:
        sys.exit(f"refusing: {item['path']} sha256 {got} != frozen {item['sha256']}")
print("[final] frozen calibrators and grid verified")
PY
read -r GRID CAL NBOOT LABEL <<< "$(python3 -c "import json,sys,os; f=json.load(open(sys.argv[1])); print(f['grid']['path'], os.path.dirname(f['calibrators']['token']['path']), f['primary']['n_boot'], f['primary']['boot_label'])" "$FROZEN")"
REPO="$(pwd)"
R=/home2/$USER/c3-runs/c3set/report
E="ALL,C3_REPO=$REPO,C3_MODE=report,C3_REPORT_HALF_OK=1,C3_N_BOOT=$NBOOT"

A=$(sbatch --parsable --export="$E,C3_STEPS=fieldtest setcal" slurm/c3set_analysis.sbatch)
I=$(sbatch --parsable --export="$E,C3_HALF=report,C3_TAG=final,C3_GRID=$REPO/$GRID,C3_CAL_DIR=$REPO/$CAL" slurm/infill_eval.sbatch)
N=$(sbatch --parsable --dependency=afterok:$I --export="$E,C3_STEPS=infill,C3_INFILL_TAG=final,C3_INFILL_ARGS=--boot-label $LABEL" slurm/c3set_analysis.sbatch)
PROBE=$(ls artifacts/probe/results/c3set_ft110m-v4.json 2>/dev/null || true)
W=$(sbatch --parsable --dependency=afterok:$A:$N --export="ALL,C3_REPO=$REPO,C3_WANDB_ARGS=--name c3setlevel-v4-final --job-type final --tags final report --config $REPO/$FROZEN --infill $R/infill_final.json --fieldtest $R/fieldtest.json --setcal $R/setcal.json${PROBE:+ --probe $REPO/$PROBE}" slurm/wandb_log.sbatch)
echo "$(date +%F_%T) final: analysis=$A infill=$I matched=$N wandb=$W tag=$TAG" | tee -a logs/c3set_jobs.txt
