#!/bin/bash
# T12 REPORT phase on Ada: the fallback for the Jarvis sweep (scripts/jl_t12.sh launch report).
# Submits slurm/t12.sbatch once per task, as an array of shards, with the same settings the
# Jarvis job uses (scripts/jarvis_t12.sh do_sweep): report half, zero-shot chat prompt, fp16,
# the committed per-task grid prereg/t12_grid_<task>.json and the frozen C3a calibrator.
#
#   ssh ada; cd /home2/$USER/c3 && git fetch origin && git checkout --detach origin/ayush/t12-grids
#   bash slurm/t12_report_ada.sh gsm8k        # 12 shards x 2 GPUs (2080 Ti)
#   bash slurm/t12_report_ada.sh mbpp         # 5 shards
#   bash slurm/t12_report_ada.sh gsm8k 16     # more shards, if the queue has room
#
# Deviation 17: one machine per task.  Run a task here only if Jarvis is not running it, and
# claim it first: coord.py claim run:t12-report-<task>-ada.  A task's shards all run here.
# Records go to /home2/$USER/c3-runs/t12/report-<task>-ada-<sha7>/ (t12.sbatch, fsync'd per
# problem; a requeued or resubmitted shard resumes).  Score: scripts/score_t12.py on that dir.
#
# Refuses unless: the checkout is clean, contains the grid extension (4435c13), deviation 17
# (1b78470) and the grids, is on a branch on origin, and the grid, calibrator and corpus
# hashes match the pins (grids: bus 20261027T224410Z; the rest: scripts/jarvis_t12.inputs.sha256).
#
# Rough time, from Jarvis's calibration (77 forwards per record) and Ada's 2080 Ti speed
# (0.80 s/forward at the 5-shot prompt, ~0.28 s at the zero-shot one): GSM8K 659 x 21
# records ~ 83 h of 2-GPU job time, MBPP 250 x 21 ~ 31 h; so ~7 h per shard at the defaults.
set -euo pipefail
TASK="${1:?usage: t12_report_ada.sh gsm8k|mbpp [num_shards]}"
case "$TASK" in
    gsm8k) NS="${2:-12}"; GRID_SHA=c30115009c33c13e02cea612ebb70be20de7e8b6c71b39de2a2195a3bc9fb09e ;;
    mbpp)  NS="${2:-5}";  GRID_SHA=fe631dee91c0e305181228b545915f9339833995f55523e951eb3fc72436d0d5 ;;
    *) echo "task must be gsm8k or mbpp"; exit 1 ;;
esac
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
PERSIST="${C3_PERSIST:-/home2/$USER/c3-runs}"
die() { echo "[t12-ada] REFUSED: $*" >&2; exit 1; }
cd "$REPO_DIR"

[ -z "$(git status --porcelain --untracked-files=no)" ] || die "$REPO_DIR has uncommitted changes"
HEAD="$(git rev-parse HEAD)"
for c in 4435c13 1b78470; do
    git merge-base --is-ancestor "$c" HEAD 2>/dev/null || die "HEAD $HEAD lacks $c (grid extension / deviation 17)"
done
git fetch -q origin
[ -n "$(git branch -r --contains HEAD)" ] || die "HEAD $HEAD is on no branch on origin (push the grids first)"

GRID="prereg/t12_grid_${TASK}.json"
CAL="artifacts/c3/llada-8b-instruct_fp16_c3a-${TASK}.json"
[ "$(git show "HEAD:$GRID" | sha256sum | cut -d' ' -f1)" = "$GRID_SHA" ] || die "$GRID does not match the posted hash"
pin() { awk -v p="$1" '$2 == p { print $1 }' scripts/jarvis_t12.inputs.sha256; }
[ "$(git show "HEAD:$CAL" | sha256sum | cut -d' ' -f1)" = "$(pin "$CAL")" ] || die "$CAL does not match its pin"
for f in "artifacts/split_manifest_${TASK}.json"; do
    [ "$(git show "HEAD:$f" | sha256sum | cut -d' ' -f1)" = "$(pin "$f")" ] || die "$f does not match its pin"
done
# off-repo inputs that t12.sbatch reads from $PERSIST (tasks are built there if missing)
check_off() {  # check_off <pinned path> <path on Ada>
    [ -s "$2" ] || { [ "$1" = runs/corpus/reserve_docs.jsonl ] && die "$2 missing (copy it from HF t12-inputs/)"; return 0; }
    [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$(pin "$1")" ] || die "$2 does not match the pin of $1"
}
check_off runs/corpus/reserve_docs.jsonl "$PERSIST/corpus/reserve_docs.jsonl"
check_off "runs/corpus/tasks/${TASK}_test.jsonl" "$PERSIST/corpus/tasks/${TASK}_test.jsonl"

TAG="report-${TASK}-ada-${HEAD:0:7}"
mkdir -p "/home2/$USER/c3/logs"
echo "[t12-ada] $TASK report half, $NS shards, commit ${HEAD:0:7}, tag $TAG"
sbatch --array="0-$((NS - 1))" \
    --export="ALL,C3_TASK=$TASK,C3_HALF=report,C3_TAG=$TAG,C3_CONFIGS=$GRID,C3_PROMPT=chat,C3_CALIBRATOR=$CAL,C3_NUM_SHARDS=$NS" \
    slurm/t12.sbatch
