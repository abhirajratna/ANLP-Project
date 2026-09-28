#!/bin/bash
# Submits T3 as one chain of dependent jobs (slurm/T3_RUNBOOK.md):
#
#   env -> fineweb -> corpus -> exclusion -> shards -> probe -> train -> verify
#     `--> smoke ------------------------------------------^
#
# The train step is slurm/launch_train.sbatch: it probes a node's GPUs, then
# submits training pinned to that node and verify after it.
#
#   bash slurm/submit_t3.sh              # the whole pipeline
#   bash slurm/submit_t3.sh exclusion    # from a step on; earlier outputs must exist
#   C3_NGPU=4 C3_ACCUM=2 bash slurm/submit_t3.sh   # train on 4 GPUs, same batch
#
# A job whose prerequisite fails never starts (DependencyNeverSatisfied): fix the
# cause, `scancel` the stuck jobs, and resubmit from the failed step.
set -euo pipefail
cd "${C3_REPO:-/home2/$USER/c3}"
mkdir -p logs

STEPS=(env fineweb corpus exclusion shards smoke train verify)
declare -A SCRIPT=([env]=slurm/build_env.sbatch [fineweb]=slurm/fetch_fineweb.sbatch
    [corpus]=slurm/prepare_corpus.sbatch [exclusion]=slurm/prepare_exclusion.sbatch
    [shards]=slurm/build_shards.sbatch [smoke]=slurm/smoke_train.sh
    [train]=slurm/launch_train.sbatch [verify]=slurm/verify_110m.sh)
declare -A NEEDS=([env]="" [fineweb]=env [corpus]=fineweb [exclusion]=corpus
    [shards]=exclusion [smoke]=env [train]="shards smoke" [verify]=train)

FROM="${1:-env}"
if [ -z "${SCRIPT[$FROM]:-}" ]; then
    echo "unknown step '$FROM'; steps: ${STEPS[*]}"; exit 1
fi

declare -A ID
on=0
record=()
for s in "${STEPS[@]}"; do
    if [ "$s" = "$FROM" ]; then on=1; fi
    if [ "$on" = 0 ]; then continue; fi
    # launch_train.sbatch submits verify itself once training is placed.
    if [ "$s" = verify ] && [ -n "${ID[train]:-}" ]; then continue; fi
    # Only prerequisites submitted in this run; earlier steps are taken as done.
    deps=()
    for d in ${NEEDS[$s]}; do
        if [ -n "${ID[$d]:-}" ]; then deps+=("${ID[$d]}"); fi
    done
    args=(--parsable)
    if [ ${#deps[@]} -gt 0 ]; then args+=("--dependency=afterok:$(IFS=:; echo "${deps[*]}")"); fi
    if [ "$s" = train ] && [ -n "${C3_NGPU:-}" ]; then args+=("--gres=gpu:$C3_NGPU"); fi
    out=$(sbatch "${args[@]}" "${SCRIPT[$s]}")
    ID[$s]="${out%%;*}"
    record+=("$s=${ID[$s]}")
    printf '%-10s job %-10s %s\n' "$s" "${ID[$s]}" "${deps[*]:+(after ${deps[*]})}"
done
echo "$(date +%F_%T) ${record[*]}" >> logs/t3_jobs.txt
echo 'recorded in logs/t3_jobs.txt; watch with: squeue -u $USER'
