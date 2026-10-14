#!/bin/bash
#SBATCH --job-name=c3-tests
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M
#SBATCH --time=02:00:00
#SBATCH --output=/home2/%u/c3/logs/tests-%j.out
#
# The test suite on a Linux node with the cluster environment -- the only place the
# tests that drive the job scripts through bash (test_slurm_login, test_prereg,
# test_c1_pipeline's merge) run as they do in production.
#
#   sbatch slurm/run_tests.sbatch                               # everything
#   sbatch --export=ALL,C3_PYTEST_ARGS="tests/test_slurm_login.py -q" slurm/run_tests.sbatch
set -euo pipefail
REPO_DIR="${C3_REPO:-/home2/$USER/c3}"
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
cd "$REPO_DIR"
echo "[tests] node=$(hostname) commit=$(git rev-parse HEAD)"
# shellcheck disable=SC2086
"$C3_PY" -m pytest ${C3_PYTEST_ARGS:--q tests}
