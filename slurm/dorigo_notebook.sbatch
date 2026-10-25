#!/bin/bash
#SBATCH --job-name=c3-dorigo
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=14
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home2/%u/c3/logs/dorigo-%j.out
#
# Runs notebooks/c3_dorigo_methods.ipynb (C3 x Dorigo winner's-curse modules A-F,
# E2 own/field split, H3 at equal confidence) headless on C1 traces held on Ada,
# and saves the executed notebook, CSVs and figures. CPU only, no GPU.
#
# From a checkout that has this file (main once PRs #18 and #21 merge; until then the
# branch ayush/dorigo-prereg-hold), on the Ada login node:
#
#   cd ~/c3 && git fetch origin && git checkout ayush/dorigo-prereg-hold   # or main after the merge
#   mkdir -p /home2/$USER/c3/logs
#   sbatch slurm/dorigo_notebook.sbatch                                     # 8B base runs, fit half only
#   C3_RUNS="c1-llada-base" sbatch slurm/dorigo_notebook.sbatch             # one run
#
# Ada's ~/c3 has no remote; from a laptop: `git push ada:c3 ayush/dorigo-prereg-hold`.
#
# Pre-registration. Every C1 arm is pre-registered and the report half stays
# unread until prereg-v1 is tagged. The default HALF_MODE=fit_only drops the
# report-half documents before anything is computed and splits the fit half
# again into inner fit / report halves; numbers from it are exploratory. Report-half
# numbers need both, set by the submitter (the job never sets them):
#
#   C3_REPORT_HALF_OK=1 HALF_MODE=report sbatch slurm/dorigo_notebook.sbatch
#
# and the job refuses unless the tag prereg-v1 exists in this checkout.
#
# Settings (environment, all optional):
#   C3_RUNS     run directories under TRACES_SRC (default: c1-llada-base c1-dream-base).
#               Also works for c1-llada-instruct-gsm8k, c1-llada-instruct-mbpp-v2,
#               c1-dream-instruct-gsm8k, c1-dream-instruct-mbpp, main-110m-v3, timesplit-110m-v3.
#   TRACES_SRC  where the runs live on /share1 (default /share1/$USER/traces), reached
#               over ssh from the node (/share1 is not mounted there)
#   HALF_MODE   fit_only (default) | report (see above)
#   N_BOOT      bootstrap resamples (default: the notebook's 10,000)
#   OUT         results (default /home2/$USER/c3-runs/dorigo/<HALF_MODE>-<job id>)
#
# What comes out, in $OUT: c3_dorigo_methods.executed.ipynb (open it in Jupyter or
# Colab: tables, figures, plain-language summary), c3_dorigo_out/ (summary*.csv,
# summary_e2.csv, summary_h3.csv, icc_influence.csv, *.png, run_meta.json).
# It worked when the log ends "[dorigo] done" and the notebook's summary says
# "Self-tests: 51/51 passed".
#
# Sizing. Each run is ~120M rows (1,979 documents; same schema as main-110m-v3).
# One mask kind is loaded at a time: ~6-11 GB peak on Colab for main-110m-v3, so 40G
# (= 14 CPUs under MaxMemPerCPU) leaves room. Colab took ~25-35 min per run plus
# ~6 min of self-tests; 12 h covers several runs. Staging copies ~20 GB per run to
# /ssd_scratch, in the same place as slurm/prereg_analysis.sbatch, so a node that
# already staged a run for that job skips the copy.
set -euo pipefail
die() { echo "[dorigo] refusing: $*" >&2; exit 1; }

REPO_DIR="${C3_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
NB="$REPO_DIR/notebooks/c3_dorigo_methods.ipynb"
[ -f "$NB" ] || die "$NB not found: submit from a checkout that has the notebook (ayush/dorigo-prereg-hold or main)"
TAG=prereg-v1
HALF_MODE="${HALF_MODE:-fit_only}"
case "$HALF_MODE" in
    fit_only) ;;
    report)
        [ "${C3_REPORT_HALF_OK:-}" = 1 ] || die "HALF_MODE=report needs C3_REPORT_HALF_OK=1 from the submitter"
        git -C "$REPO_DIR" rev-parse -q --verify "refs/tags/$TAG^{commit}" >/dev/null \
            || die "no tag $TAG in $REPO_DIR: the report half stays closed until it is tagged" ;;
    *) die "HALF_MODE must be fit_only or report, not $HALF_MODE" ;;
esac
HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)
DIRTY=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)
[ -z "$DIRTY" ] || echo "[dorigo] WARNING: uncommitted changes in $REPO_DIR (recorded commit is HEAD): $DIRTY"

# ---------------------------------------------------------------- environment
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
export C3_GIT_COMMIT="$HEAD"
cd "$REPO_DIR"

# The project env has no scikit-learn, matplotlib or Jupyter: add them in a venv on
# top of it (node-local, reused by later jobs on the node). Versions = the ones the
# notebook was tested with.
NBENV="$C3_SCRATCH/dorigo-nbenv"
NB_PKGS="scikit-learn==1.9.1 matplotlib==3.11.2 nbclient==0.11.0 nbformat==5.11.1 ipykernel==7.4.0"
exec 8>"$C3_SCRATCH/.dorigo-nbenv.lock"
flock 8
if ! "$NBENV/bin/python" -c "import sklearn, matplotlib, nbclient, ipykernel, pyarrow, scipy" 2>/dev/null; then
    echo "[dorigo] building $NBENV"
    rm -rf "$NBENV"
    "$C3_PY" -m venv --system-site-packages "$NBENV"
    export PIP_CACHE_DIR="$C3_SCRATCH/pip-cache"
    for attempt in 1 2 3 4 5; do
        "$NBENV/bin/python" -m pip install --retries 10 --timeout 60 $NB_PKGS && break
        [ "$attempt" = 5 ] && die "pip install failed 5 times"
        echo "[dorigo] pip failed (attempt $attempt/5); retrying in 60 s"; sleep 60
    done
fi
flock -u 8
PY="$NBENV/bin/python"
export PATH="$NBENV/bin:$PATH"
export MPLBACKEND=Agg

# ---------------------------------------------------------------- stage
TRACES_SRC="${TRACES_SRC:-/share1/$USER/traces}"
read -r -a RUNS <<<"${C3_RUNS:-c1-llada-base c1-dream-base}"
STAGE="$C3_SCRATCH/prereg/traces"
mkdir -p "$STAGE"
echo "[dorigo] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-none} HALF_MODE=$HALF_MODE"
echo "[dorigo] staging ${RUNS[*]} from $TRACES_SRC; /ssd_scratch free: $(df -h --output=avail "$C3_SCRATCH" | tail -1)"
for r in "${RUNS[@]}"; do
    t0=$SECONDS
    mkdir -p "$STAGE/$r"
    rsync -a -e "ssh $C3_SSH_OPTS" "$USER@$C3_LOGIN_HOST:$TRACES_SRC/$r/$r.dataset.json" "$STAGE/$r/" \
        || die "no $TRACES_SRC/$r/$r.dataset.json on $C3_LOGIN_HOST"
    "$C3_PY" -c 'import json, sys; print("\n".join(f["file"] for f in json.load(open(sys.argv[1]))["files"]))' \
        "$STAGE/$r/$r.dataset.json" > "$STAGE/.$r.files"
    rsync -a --files-from="$STAGE/.$r.files" -e "ssh $C3_SSH_OPTS" \
        "$USER@$C3_LOGIN_HOST:$TRACES_SRC/$r/" "$STAGE/$r/" || die "could not copy the parts of $r"
    n=$("$C3_PY" -c 'import sys; from c3.traces.shards import dataset_files; print(len(dataset_files(sys.argv[1], verify=True)))' \
        "$STAGE/$r/$r.dataset.json") || die "$r: a staged part does not match its manifest"
    echo "[dorigo] $r: $n parts, sha256 ok, $((SECONDS - t0)) s"
done

# ---------------------------------------------------------------- run
OUT="${OUT:-$C3_PERSIST/dorigo/$HALF_MODE-${SLURM_JOB_ID:-$$}}"
mkdir -p "$OUT"
# One notebook DATASETS entry per run: only the listed parts (the run directory may
# hold other files), and the run's fit/report manifest from artifacts/.
DATASETS=$("$PY" - "$STAGE" "$REPO_DIR" "${RUNS[@]}" <<'PY'
import json, sys
from pathlib import Path
stage, repo, runs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
def manifest(run):
    if "timesplit" in run:
        return "split_manifest_timesplit.json"
    for task in ("gsm8k", "mbpp"):
        if task in run:
            return f"split_manifest_{task}.json"
    return "split_manifest.json"
out = {}
for r in runs:
    view = stage / f".view-{r}"
    view.mkdir(exist_ok=True)
    for f in json.loads((stage / r / f"{r}.dataset.json").read_text())["files"]:
        link = view / f["file"].replace("/", "__")      # parts may sit in shard subfolders
        if not link.exists():
            link.symlink_to((stage / r / f["file"]).resolve())
    man = repo / "artifacts" / manifest(r)
    assert man.is_file(), man
    out[r] = {"path": str(view), "split": {"manifest": str(man)}}
print(json.dumps(out))
PY
)
echo "[dorigo] DATASETS=$DATASETS"
SETS=(--set USE_SYNTHETIC=False --set "HALF_MODE=\"$HALF_MODE\"" --set MOUNT_DRIVE=False
      --set "DATASETS=$DATASETS" --set "OUT_DIR=\"$OUT/c3_dorigo_out\"" --set CONFIRM_FULL_BOOT=True)
[ -n "${N_BOOT:-}" ] && SETS+=(--set "N_BOOT=$N_BOOT")
t0=$SECONDS
"$PY" notebooks/run_notebook.py "$NB" --out "$OUT/c3_dorigo_methods.executed.ipynb" \
    --cwd "$REPO_DIR" "${SETS[@]}" || die "a notebook cell failed: see $OUT/c3_dorigo_methods.executed.ipynb"
echo "$HEAD" > "$OUT/git_commit.txt"
echo "[dorigo] done in $((SECONDS - t0)) s -> $OUT (HALF_MODE=$HALF_MODE, commit ${HEAD:0:12})"
