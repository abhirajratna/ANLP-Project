#!/bin/bash
# Run on a login gateway: waits until Slurm accepts this account's jobs again
# (after the October 2026 update the account's association was missing, and every
# sbatch failed), then submits the T3 training chain once and exits.
#
#   C3_NGPU=4 C3_ACCUM=4 nohup setsid bash slurm/autosubmit_t3.sh \
#       >> logs/autosubmit.log 2>&1 < /dev/null &
#   pkill -f autosubmit_t3.sh        # to stop it
#
# Each check is `sbatch --test-only` of the actual launch job, so it passes only
# when the real submission would.  Gives up after C3_RETRY_MAX_H hours.
set -uo pipefail
cd "${C3_REPO:-/home2/$USER/c3}"
INTERVAL="${C3_RETRY_SECS:-900}"
DEADLINE=$(( $(date +%s) + ${C3_RETRY_MAX_H:-336} * 3600 ))
GRES=()
if [ -n "${C3_NGPU:-}" ]; then GRES=(--gres="gpu:$C3_NGPU"); fi

echo "$(date +%F_%T) watching on $(hostname -s) (pid $$): C3_NGPU=${C3_NGPU:-2} C3_ACCUM=${C3_ACCUM:-8} run=${C3_RUN_NAME:-train110m-v4}"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if why=$(sbatch --test-only "${GRES[@]}" slurm/launch_train.sbatch 2>&1 | tail -1) \
            && ! grep -qiE "error|failure|invalid" <<<"$why"; then
        echo "$(date +%F_%T) Slurm accepts the job ($why); submitting"
        bash slurm/submit_t3.sh train
        exit $?
    fi
    echo "$(date +%F_%T) still refused: $why"
    sleep "$INTERVAL"
done
echo "$(date +%F_%T) gave up after ${C3_RETRY_MAX_H:-336} h"
exit 1
