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
# E2 own/field split, H3 at equal winner l) headless on C1 traces held on Ada,
# and saves the executed notebook, CSVs and figures. CPU only, no GPU.
#
# Submit from a clean checkout of main, in its own worktree (leave ~/c3 on its branch).
# On the Ada login node, if the clone has a remote:
#
#   git -C ~/c3 fetch origin --tags
#   git -C ~/c3 worktree add --detach ~/c3-dorigo origin/main      # later: git -C ~/c3-dorigo checkout --detach origin/main
#
# If it has none (abhiraj's ~/c3), push main and the tag from a laptop first (`ada`
# refuses ssh; use a gateway), then add the worktree from that branch:
#
#   git fetch origin --tags
#   git push <user>@ada-gw1:c3 origin/main:refs/heads/dorigo-main refs/tags/prereg-v1
#   ssh <user>@ada-gw1 'git -C ~/c3 worktree add --detach ~/c3-dorigo dorigo-main'
#
# Then:
#   mkdir -p /home2/$USER/c3/logs && cd ~/c3-dorigo
#   TRACES_SRC=/share1/abhiraj.ratna/traces sbatch slurm/dorigo_notebook.sbatch   # 8B base runs, fit half only
#
# Pre-registration. The default HALF_MODE=fit_only reads only fit-half documents
# (report-half rows are filtered out at read time) and splits them again into inner
# fit / report halves; its numbers are exploratory. Report-half numbers need both of
# these from the submitter (the job never sets them):
#
#   C3_REPORT_HALF_OK=1 HALF_MODE=report sbatch slurm/dorigo_notebook.sbatch
#
# and the job (and the notebook itself) refuses unless prereg-v1 is an ancestor of
# HEAD and the tree is clean. Even then the numbers are exploratory: the
# pre-registered ones come from prereg/analysis_commands.sh.
#
# Settings (environment, all optional):
#   C3_RUNS     run directories under TRACES_SRC (default: c1-llada-base c1-dream-base).
#               Also works for c1-llada-instruct-gsm8k, c1-llada-instruct-mbpp-v2,
#               c1-dream-instruct-gsm8k, c1-dream-instruct-mbpp, main-110m-v3, timesplit-110m-v3.
#   TRACES_SRC  where the runs live on /share1 (default /share1/$USER/traces), reached
#               over ssh from the node (/share1 is not mounted there)
#   HALF_MODE   fit_only (default) | report (see above)
#   N_BOOT      bootstrap resamples, an integer (default: the notebook's 10,000)
#   OUT         results (default /home2/$USER/c3-runs/dorigo/<HALF_MODE>-<job id>)
#
# What comes out, in $OUT: c3_dorigo_methods.executed.ipynb (open it in Jupyter or
# Colab: tables, inline figures, plain-language summary), c3_dorigo_out/ (summary*.csv,
# summary_e2.csv, summary_h3.csv, icc_influence.csv, *.png, run_meta.json).
# It worked when the log ends "[dorigo] done" and the notebook's summary says
# "Self-tests: N/N passed" (all passed).
#
# Environment. The c3 env (activate_env.sh) is itself a venv, and has numpy, scipy,
# pandas, pyarrow and matplotlib but not scikit-learn or Jupyter. The job makes a
# node-local venv whose site-packages points at c3-env's (a .pth file) and pip-installs
# only the missing packages there, constrained to c3-env's versions. The first job on
# a node pays the pip download (PyPI can be slow from compute nodes: tens of kB/s
# measured, slurm/CLUSTER.md; ~15 MB here); later jobs on that node reuse the venv.
#
# Sizing. Each run is ~120M rows (1,979 documents; same schema as main-110m-v3).
# One mask kind is loaded at a time: ~6-11 GB peak on Colab for main-110m-v3, so 40G
# (= 14 CPUs under MaxMemPerCPU) leaves room. Colab took ~25-35 min per run plus
# ~6 min of self-tests; 12 h covers several runs. Staging copies ~16 GB per run to
# /ssd_scratch, in the same place as slurm/prereg_analysis.sbatch, so a node that
# already staged a run skips the copy.
set -euo pipefail
die() { echo "[dorigo] refusing: $*" >&2; exit 1; }

REPO_DIR="${C3_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
NB="$REPO_DIR/notebooks/c3_dorigo_methods.ipynb"
[ -f "$NB" ] || die "$NB not found: submit from a checkout of main that has the notebook"
TAG=prereg-v1
HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)
DIRTY=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)
HALF_MODE="${HALF_MODE:-fit_only}"
case "$HALF_MODE" in
    fit_only)
        [ -z "$DIRTY" ] || echo "[dorigo] WARNING: uncommitted changes in $REPO_DIR (recorded commit is HEAD): $DIRTY" ;;
    report)
        [ "${C3_REPORT_HALF_OK:-}" = 1 ] || die "HALF_MODE=report needs C3_REPORT_HALF_OK=1 from the submitter"
        TAGGED=$(git -C "$REPO_DIR" rev-parse -q --verify "refs/tags/$TAG^{commit}") \
            || die "no tag $TAG in $REPO_DIR: the report half stays closed until it is tagged"
        git -C "$REPO_DIR" merge-base --is-ancestor "$TAGGED" HEAD \
            || die "$TAG (${TAGGED:0:12}) is not an ancestor of HEAD ${HEAD:0:12}"
        [ -z "$DIRTY" ] || die "$REPO_DIR has uncommitted changes: $DIRTY"
        echo "[dorigo] report half: $TAG = ${TAGGED:0:12} is an ancestor of HEAD ${HEAD:0:12}, clean tree" ;;
    *) die "HALF_MODE must be fit_only or report, not $HALF_MODE" ;;
esac
[[ -z "${N_BOOT:-}" || "$N_BOOT" =~ ^[1-9][0-9]*$ ]] || die "N_BOOT must be a positive integer, not '$N_BOOT'"

# ---------------------------------------------------------------- environment
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
export C3_GIT_COMMIT="$HEAD"
unset PYTHONPATH PYTHONHOME          # as build_env.sbatch: nothing outside the two venvs
cd "$REPO_DIR"

# A venv on top of c3-env: c3-env's site-packages via a .pth file, plus the packages it
# lacks. pip is constrained to c3-env's own versions, so numpy/scipy/pandas/pyarrow are
# never replaced. Built in a temporary directory and renamed into place, so a job never
# sees a half-built venv and a running kernel's venv is never deleted. The name carries
# a version: bump it when NB_PKGS changes.
NBENV="$C3_SCRATCH/dorigo-nbenv-v2"
NB_PKGS="scikit-learn==1.9.1 matplotlib==3.11.2 nbclient==0.11.0 nbformat==5.11.1 ipykernel==7.4.0"
C3_SITE=$("$C3_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
nbenv_ok() {
    [ -x "$1/bin/python" ] || return 1
    "$1/bin/python" - "$C3_SITE" <<'PY'
import sys
import numpy, scipy, pandas, pyarrow, sklearn, matplotlib, nbclient, nbformat, ipykernel  # noqa: F401
assert numpy.__file__.startswith(sys.argv[1]), f"numpy from {numpy.__file__}, not c3-env ({sys.argv[1]})"
assert pandas.__file__.startswith(sys.argv[1]), f"pandas from {pandas.__file__}, not c3-env"
PY
}
exec 8>"$C3_SCRATCH/.dorigo-nbenv.lock"
flock 8
if ! nbenv_ok "$NBENV" 2>/dev/null; then
    echo "[dorigo] building $NBENV on top of $C3_SITE"
    TMPENV="$NBENV.build.$$"
    # leftovers of earlier builds: failed builds, and old venvs no kernel uses any more (> 1 day)
    find "$C3_SCRATCH" -maxdepth 1 \( -name "$(basename "$NBENV").build.*" -o -name "$(basename "$NBENV").old.*" \) \
        -mmin +1440 -exec rm -rf {} + 2>/dev/null || true
    rm -rf "$TMPENV"
    "$C3_PY" -m venv "$TMPENV"
    echo "$C3_SITE" > "$("$TMPENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/_c3env.pth"
    # c3-env's versions as constraints (read by the new venv's pip, which sees c3-env through the .pth)
    "$TMPENV/bin/python" -m pip list --format=freeze 2>/dev/null | grep -E '^[A-Za-z0-9_.-]+==' \
        | grep -viE '^(pip|setuptools|wheel)==' > "$TMPENV/c3env-constraints.txt" || true
    [ -s "$TMPENV/c3env-constraints.txt" ] || die "no c3-env package list for the pip constraints"
    export PIP_CACHE_DIR="$C3_SCRATCH/pip-cache"
    for attempt in 1 2 3 4 5; do
        "$TMPENV/bin/python" -m pip install --retries 10 --timeout 60 \
            -c "$TMPENV/c3env-constraints.txt" $NB_PKGS && break
        [ "$attempt" = 5 ] && die "pip install failed 5 times"
        echo "[dorigo] pip failed (attempt $attempt/5); retrying in 60 s"; sleep 60
    done
    nbenv_ok "$TMPENV" || die "the new venv cannot import the notebook's packages from c3-env (see above)"
    [ -e "$NBENV" ] && mv "$NBENV" "$NBENV.old.$$"      # never deleted while a kernel may use it
    mv "$TMPENV" "$NBENV"
fi
flock -u 8
PY="$NBENV/bin/python"
"$PY" -c 'import numpy, pandas, pyarrow, sklearn; print("[dorigo] env:", numpy.__version__, pandas.__version__, pyarrow.__version__, sklearn.__version__, numpy.__file__)'

# ---------------------------------------------------------------- stage
TRACES_SRC="${TRACES_SRC:-/share1/$USER/traces}"
read -r -a RUNS <<<"${C3_RUNS:-c1-llada-base c1-dream-base}"
STAGE="$C3_SCRATCH/prereg/traces"        # shared with prereg_analysis.sbatch: parts only
mkdir -p "$STAGE"
JOBTMP=$(mktemp -d "$C3_SCRATCH/prereg/dorigo-job-${SLURM_JOB_ID:-$$}.XXXX")
trap 'rm -rf "$JOBTMP"' EXIT
echo "[dorigo] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-none} HALF_MODE=$HALF_MODE"
echo "[dorigo] staging ${RUNS[*]} from $TRACES_SRC; /ssd_scratch free: $(df -h --output=avail "$C3_SCRATCH" | tail -1)"
verify() {   # verify <manifest>: delete every listed part whose sha256 differs (rsync would keep it forever)
    "$C3_PY" - "$1" <<'PY'
import json, sys
from pathlib import Path
from c3.provenance import sha256_file
mp = Path(sys.argv[1]); bad = []
for f in json.loads(mp.read_text())["files"]:
    p = mp.parent / f["file"]
    if not p.is_file() or sha256_file(p) != f["sha256"]:
        bad.append(f["file"]); p.unlink(missing_ok=True)
print(f"{len(bad)} bad: {bad[:5]}" if bad else "ok")
sys.exit(1 if bad else 0)
PY
}
for r in "${RUNS[@]}"; do
    t0=$SECONDS
    mkdir -p "$STAGE/$r"
    rsync -a -e "ssh $C3_SSH_OPTS" "$USER@$C3_LOGIN_HOST:$TRACES_SRC/$r/$r.dataset.json" "$STAGE/$r/" \
        || die "no $TRACES_SRC/$r/$r.dataset.json on $C3_LOGIN_HOST"
    "$C3_PY" -c 'import json, sys; print("\n".join(f["file"] for f in json.load(open(sys.argv[1]))["files"]))' \
        "$STAGE/$r/$r.dataset.json" > "$JOBTMP/$r.files"
    for attempt in 1 2; do
        rsync -a --files-from="$JOBTMP/$r.files" -e "ssh $C3_SSH_OPTS" \
            "$USER@$C3_LOGIN_HOST:$TRACES_SRC/$r/" "$STAGE/$r/" || die "could not copy the parts of $r"
        if verify "$STAGE/$r/$r.dataset.json"; then break; fi
        [ "$attempt" = 2 ] && die "$r: parts still fail sha256 after a fresh copy"
        echo "[dorigo] $r: bad parts deleted, copying them again"
    done
    echo "[dorigo] $r: $(wc -l < "$JOBTMP/$r.files") parts, sha256 ok, $((SECONDS - t0)) s"
done

# ---------------------------------------------------------------- run
OUT="${OUT:-$C3_PERSIST/dorigo/$HALF_MODE-${SLURM_JOB_ID:-$$}}"
[[ "$OUT" =~ ^[A-Za-z0-9_./-]+$ ]] || die "OUT may hold only letters, digits and _ . / - (it is passed to the notebook as Python)"
mkdir -p "$OUT"
# One notebook DATASETS entry per run: a per-job view holding only the listed parts
# (the run directory may hold other files), and the run's fit/report manifest.
DATASETS=$("$PY" - "$STAGE" "$JOBTMP" "$REPO_DIR" "${RUNS[@]}" <<'PY'
import json, sys
from pathlib import Path
stage, jobtmp, repo, runs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
def manifest(run):
    if "timesplit" in run:
        return "split_manifest_timesplit.json"
    for task in ("gsm8k", "mbpp"):
        if task in run:
            return f"split_manifest_{task}.json"
    return "split_manifest.json"
out = {}
for r in runs:
    view = jobtmp / f"view-{r}"
    view.mkdir()
    for f in json.loads((stage / r / f"{r}.dataset.json").read_text())["files"]:
        (view / f["file"].replace("/", "__")).symlink_to((stage / r / f["file"]).resolve())
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
EXEC="$OUT/c3_dorigo_methods.executed.ipynb"
t0=$SECONDS
"$PY" notebooks/run_notebook.py "$NB" --out "$EXEC" --cwd "$REPO_DIR" "${SETS[@]}" \
    || die "a notebook cell failed: see $EXEC"
grep -q "python_executable[^,]*$NBENV/bin/python" "$EXEC" \
    || echo "[dorigo] WARNING: the kernel did not run $NBENV/bin/python (see VERSIONS in cell 0.2)"
echo "$HEAD" > "$OUT/git_commit.txt"
echo "[dorigo] done in $((SECONDS - t0)) s -> $OUT (HALF_MODE=$HALF_MODE, commit ${HEAD:0:12})"
