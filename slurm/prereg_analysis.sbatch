#!/bin/bash
#SBATCH --job-name=c3-prereg
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=40
#SBATCH --mem=120000M
#SBATCH --time=4-00:00:00
#SBATCH --output=/home2/%u/c3/logs/prereg-%j.out
#
# The pre-registered analysis (prereg/analysis_commands.sh) on the REPORT HALF, for the
# C1 runs this account holds.  Once prereg-v1 is tagged and the pre-registration is on
# OSF, from a clean checkout of the tag on Ada (any worktree of the account's clone):
#
#   git -C ~/c3 worktree add --detach ~/c3-prereg prereg-v1      # once
#   cd ~/c3-prereg && C3_REPORT_HALF_OK=1 sbatch slurm/prereg_analysis.sbatch
#
# ojas's account (the 110M arms, run main-110m-v3 and timesplit-110m-v3) needs nothing
# else.  abhiraj's (the 8B arms) too, with TRACES_SRC defaulting to his own /share1.
# Ada's ~/c3 has no remote, so bring the tag over from the laptop first:
# `git fetch origin --tags && git push ada:c3 prereg-v1`.  The job runs the commands of
# the checkout it was submitted from (SLURM_SUBMIT_DIR, or C3_REPO), so this file need
# not be in the tag: `sbatch <path to it>` from the tagged checkout works too.
#
# What the job does:
#  1. Refuses unless C3_REPORT_HALF_OK=1 came from the submitter (the job never sets
#     it), and unless the checkout is clean (tracked files) with HEAD at prereg-v1.
#  2. Stages each run's <run>.dataset.json, and the parts it lists, from
#     $TRACES_SRC/<run>/ (default /share1/$USER/traces; /share1 is not mounted on the
#     nodes, so through $C3_LOGIN_HOST) to node-local SSD, and checks every part
#     against the sha256 recorded at the merge.  By default the runs are those under
#     TRACES_SRC that analysis_commands.sh names: main-110m-v3 and timesplit-110m-v3 on
#     ojas.k, the nine 8B runs on abhiraj.ratna.  C3_RUNS="a b" stages exactly those.
#  3. Checks, offline, what the commands will need: the T8 tokenizers from the HF cache
#     (HF_HUB_OFFLINE=1 unless set), the corpus and task files against their recorded
#     sha256, the split and contamination manifests.  It warns if matplotlib is missing
#     (figures are skipped, nothing else changes: the env tarball has none; one rebuild
#     of it, by abhiraj, adds it, and each account needs that tarball).
#  4. Runs `bash prereg/analysis_commands.sh` from the checkout, with TRACES holding only
#     the runs it staged and verified.  The arms of runs not staged are skipped and named.
#
# OUT defaults to /home2/$USER/c3-runs/prereg-v1.  /home2 is mounted on the nodes, so each
# step's results land there as it finishes and outlive the job; /share1 would need a push
# at the end, and its ~15 GB free is for traces.  Once the other account's $OUT/c1/<label>
# directories are copied in, `SUMMARY_ONLY=1` with the same OUT runs the 7-arm Holm
# summary alone (no staging; a small job will do: --cpus-per-task=2 --mem=5G).
#
# The 8B arms (abhiraj's runs, group-readable under /share1/abhiraj.ratna/traces) take the
# same script from either account: TRACES_SRC=/share1/abhiraj.ratna/traces, and C3_RUNS to
# split them into one job per run (each job needs the whole QOS, so submit them one after
# the other, e.g. with --dependency=afterany; the steps already done are kept).  The T8
# precision step needs both of its runs in one job: C3_RUNS="c1-llada-base c1t8-llada-base-4bit".
#
# Sizing (nothing here read a report half).  Each command loads the WHOLE run, both halves:
# calibrators are fitted on the fit half and every number comes from the report half.
# Memory and time grew linearly with the rows loaded, measured on fit-half documents of
# main-110m-v3 at 9M rows (150 documents; jobs 36215, 36251) and 30M rows (500 documents,
# split 254/246; job 36422, 14 CPUs), and scale to the 119.6M rows (1,979 documents) of
# main-110m-v3, LLaDA-Base or Dream-Base, and the 30.2M rows of timesplit-110m-v3.
# At 120M rows:
#   - the main process peaks at ~90 GB in Eq. 2/3 (22.8 GB at 30M) and ~105 GB in T8
#     (26.2 GB at 30M); each Eq. 2/3 worker adds up to ~7.5 GB (1.9 GB at 30M), each
#     worker holds its own copy of one section.  120,000 MB is the QOS limit (u22 gives
#     3,000 MB per CPU, so this is also all 40 CPUs, and the job needs an empty node), so
#     N_JOBS defaults to 3 (~112 GB); N_JOBS=4 would reach the limit.  Memory, not CPUs,
#     caps the bootstrap.
#   - E1/E2 without Eq. 2/3 (10,000 resamples): ~45 min, 77 GB.  E3: 1-2 h, 45 GB.
#     T8: 4-6 h, 105 GB.  The 30M-row time-split run: ~15 min, 20 GB.
#   - Eq. 2/3: 118 s per resample at 30M rows with one worker (two resamples 2,408 s in
#     all, six 2,882 s), i.e. ~310-475 s at 120M (work grows with rows^0.7-1.0), summed
#     over the 7 mask kinds and 4 fits (extrapolated, not measured at 120M).  With 3
#     workers that is 29-45 h for the primary run (--eq-boot 1000) and ~7-10 h for each
#     knot run (--eq-boot 200), so a whole FineWeb arm (primary, two knot runs, the
#     additive and clean runs, E3, T8) takes ~50-76 h: it fits the 4-day limit, and two
#     such arms on one account do not fit one job.  Submit it again: a step that finished
#     is kept, but one that was killed starts over, so a single step (~45 h) must fit in a
#     job.  --skip-regression runs (clean, time-split, additive) cost ~45 min each.
# A 40-CPU, 120,000 MB, 4-day job is the most the account can ask for; no --constraint:
# nothing here uses a GPU, and 2080 Ti nodes are rarely free of other jobs' CPUs.
#
# E1/E2-only mode (C3_E1E2_ONLY=1, opt-in).  The Eq. 2/3 fixed-effects bootstrap inside a
# primary step is too slow to finish in one 96 h job, and the E1/E2 Holm-family numbers
# come only from the reweighting bootstrap, so they are identical with --skip-regression
# (scripts/analyze_traces.py).  This mode banks them first: the job runs only each arm's
# primary scripts/analyze_traces.py step, `--out $OUT/c1/<Label>`, with --skip-regression
# appended (so the .prereg-done stamp records it).  Every other step with an --out is
# skipped and named: the knots5/knots13/additive/clean/time-split runs, E3, T8, the
# summary.  OUT defaults to $C3_PERSIST/prereg-v1-e1e2 (never the full job's prereg-v1,
# so a running full job's outputs are not touched).  A primary label is a directory
# directly under c1/ whose name does not end in _knots5, _knots13, _additive, _clean or
# TimeSplit.  The Holm summary (scripts/summarize_c1.py, c1/summary) passes the filter
# unchanged.  Env interface of the mode:
#   C3_E1E2_ONLY=1   switch it on (unset or 0: the normal job, unchanged).
#   C3_REPO          clean checkout at prereg-v1; the script itself comes from main / the
#                    branch that has it.
#   OUT              default $C3_PERSIST/prereg-v1-e1e2; refused if it is the full
#                    $C3_PERSIST/prereg-v1 root.
#   C3_RUNS          one arm per job: c1-llada-base, c1-dream-base, c1-llada-instruct-gsm8k,
#                    c1-llada-instruct-mbpp-v2, c1-dream-instruct-gsm8k,
#                    c1-dream-instruct-mbpp (8B, TRACES_SRC=/share1/abhiraj.ratna/traces)
#                    and main-110m-v3 (110M, ojas's account) = the 7 Holm arms.
#   SUMMARY_ONLY=1  with C3_E1E2_ONLY=1 and OUT=<e1e2 root> holding all seven
#                    c1/<Label>/results.json: the Holm summary alone (small job, no staging).
#   Upload: HF repo ANLP-project/c3-traces-8b, under prereg-v1-e1e2/ (the c1/<Label> dirs).
# Sizing: ~77 GB and ~45 min per FineWeb arm.  The analysis code stays fixed
# at the tag: submit this script from main, with C3_REPO a clean checkout at prereg-v1
# (the tag and clean-tree gates, analysis_commands.sh and activate_env.sh all come from
# C3_REPO, not from the submit directory):
#
#   cd ~/c3-prereg   # clean, HEAD = prereg-v1
#   C3_REPO=$PWD C3_E1E2_ONLY=1 C3_REPORT_HALF_OK=1 C3_RUNS=c1-dream-base \
#       TRACES_SRC=/share1/abhiraj.ratna/traces C3_SCRATCH=/ssd_scratch/$USER/v2 HF_HUB_OFFLINE=0 \
#       sbatch --cpus-per-task=38 --mem=120000M ~/c3/slurm/prereg_analysis.sbatch
#
# Dry run of the runner itself, with a stub interpreter that reads no trace:
#
#   C3_PREREG_DRYRUN=1 PY=$PWD/tests/prereg_stub_py.sh C3_REPORT_HALF_OK=1 \
#       sbatch --cpus-per-task=2 --mem=5G --time=2:00:00 slurm/prereg_analysis.sbatch
#
# C3_PREREG_DRYRUN=1 skips the tag and clean-tree checks, and is refused unless PY is a
# stub.  Its OUT defaults to $C3_PERSIST/prereg-dryrun/<job id> (prereg-dryrun-e1e2/<job id>
# with C3_E1E2_ONLY=1).
set -euo pipefail
die() { echo "[prereg] refusing: $*" >&2; exit 1; }
REPO_DIR="${C3_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
TAG=prereg-v1
DRYRUN="${C3_PREREG_DRYRUN:-0}"
E1E2="${C3_E1E2_ONLY:-0}"

# ---------------------------------------------------------------- 1. gates
[ "${C3_REPORT_HALF_OK:-}" = 1 ] || die "C3_REPORT_HALF_OK=1 is not set in the submitting \
environment.  These commands read the report half: set it only once $TAG is signed off and on OSF."
[ -f "$REPO_DIR/prereg/analysis_commands.sh" ] \
    || die "$REPO_DIR is not a c3 checkout: submit from the repository root, or set C3_REPO"
HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)
TAGGED=$(git -C "$REPO_DIR" rev-parse -q --verify "refs/tags/$TAG^{commit}" || true)
DIRTY=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)
if [ "$DRYRUN" = 1 ]; then
    case "$(basename "${PY:-python}")" in
        python*) die "C3_PREREG_DRYRUN=1 is for a stub PY only (PY=${PY:-unset})" ;;
    esac
    echo "[prereg] DRY RUN at ${HEAD:0:12}: tag and clean-tree checks skipped; PY=$PY"
else
    [ -n "$TAGGED" ] || die "no tag $TAG in $REPO_DIR (from the laptop: git push ada:c3 $TAG)"
    [ "$HEAD" = "$TAGGED" ] || die "HEAD ${HEAD:0:12} is not $TAG (${TAGGED:0:12}): git checkout --detach $TAG"
    [ -z "$DIRTY" ] || die "$REPO_DIR has uncommitted changes: $DIRTY"
    echo "[prereg] $TAG = ${HEAD:0:12}, clean checkout $REPO_DIR"
fi

# ---------------------------------------------------------------- 2. environment
if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
        [ -r "$f" ] && . "$f" && break
    done
fi
module load u22/python/3.12.4
source "$REPO_DIR/slurm/activate_env.sh"
export C3_GIT_COMMIT="$HEAD"
cd "$REPO_DIR"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRACES_SRC="${TRACES_SRC:-/share1/$USER/traces}"
STAGE="$C3_SCRATCH/prereg/traces"                 # node-local copies, reused by later jobs
export TRACES="$C3_SCRATCH/prereg/view-${SLURM_JOB_ID:-$$}"    # only what this job verified
if [ "$E1E2" = 1 ]; then
    SFX=-e1e2
else
    SFX=
fi
if [ "$DRYRUN" = 1 ]; then
    export OUT="${OUT:-$C3_PERSIST/prereg-dryrun$SFX/${SLURM_JOB_ID:-$$}}"
else
    export OUT="${OUT:-$C3_PERSIST/prereg-v1$SFX}"
fi
[ "$E1E2" != 1 ] || [ "$(realpath -m "$OUT")" != "$(realpath -m "$C3_PERSIST/prereg-v1")" ] \
    || die "C3_E1E2_ONLY=1 would write into the full prereg-v1 root ($OUT); unset OUT"
[ "$E1E2" != 1 ] || echo "[prereg] C3_E1E2_ONLY=1: primary c1 steps only, --skip-regression, OUT=$OUT"
export PY="${PY:-$C3_PY}"
export CORPUS="${CORPUS:-$C3_PERSIST/corpus/calibration_docs.jsonl}"
export TASKS_DIR="${TASKS_DIR:-$C3_PERSIST/corpus/tasks}"
export N_JOBS="${N_JOBS:-3}"
mkdir -p "$STAGE" "$TRACES" "$OUT"
trap 'rm -rf "$TRACES"' EXIT
echo "[prereg] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-none} OUT=$OUT"

# ---------------------------------------------------------------- 3. stage
RUNS=()
if [ "${SUMMARY_ONLY:-0}" != 1 ]; then
    remote() { ssh $C3_SSH_OPTS "$USER@$C3_LOGIN_HOST" "$@"; }
    if [ -n "${C3_RUNS:-}" ]; then
        read -r -a RUNS <<<"$C3_RUNS"
    else
        listing=$(remote ls -1 "$TRACES_SRC") || die "cannot list $TRACES_SRC on $C3_LOGIN_HOST"
        named=$(grep -oE '[A-Za-z0-9]+(-[A-Za-z0-9]+)+' prereg/analysis_commands.sh | sort -u)
        for r in $listing; do
            grep -qxF -- "$r" <<<"$named" || continue
            rc=0; remote test -f "$TRACES_SRC/$r/$r.dataset.json" || rc=$?
            case $rc in
                0) RUNS+=("$r") ;;
                1) echo "[prereg] $TRACES_SRC/$r has no $r.dataset.json (not merged); not staged" ;;
                *) die "cannot reach $C3_LOGIN_HOST to check $r" ;;
            esac
        done
    fi
    [ ${#RUNS[@]} -gt 0 ] || die "no run of analysis_commands.sh under $TRACES_SRC"
    echo "[prereg] staging ${RUNS[*]} from $TRACES_SRC; /ssd_scratch free: $(df -h --output=avail "$C3_SCRATCH" | tail -1)"
    for r in "${RUNS[@]}"; do
        t0=$SECONDS
        mkdir -p "$STAGE/$r"
        rsync -a -e "ssh $C3_SSH_OPTS" "$USER@$C3_LOGIN_HOST:$TRACES_SRC/$r/$r.dataset.json" "$STAGE/$r/" \
            || die "no $TRACES_SRC/$r/$r.dataset.json"
        "$C3_PY" -c 'import json, sys; print("\n".join(f["file"] for f in json.load(open(sys.argv[1]))["files"]))' \
            "$STAGE/$r/$r.dataset.json" > "$STAGE/.$r.files"
        rsync -a --files-from="$STAGE/.$r.files" -e "ssh $C3_SSH_OPTS" \
            "$USER@$C3_LOGIN_HOST:$TRACES_SRC/$r/" "$STAGE/$r/" || die "could not pull the parts of $r"
        # Every part against the sha256 the merge recorded (scripts/merge_traces.py).
        n=$("$C3_PY" -c 'import sys; from c3.traces.shards import dataset_files; print(len(dataset_files(sys.argv[1], verify=True)))' \
            "$STAGE/$r/$r.dataset.json") || die "$r: a staged part does not match its manifest"
        ln -sfn "$STAGE/$r" "$TRACES/$r"
        echo "[prereg] $r: $n parts, sha256 ok, $((SECONDS - t0)) s"
    done
fi

# ---------------------------------------------------------------- 4. preflight
# What would otherwise fail hours in: T8 loads a tokenizer per arm (gpt2 for the 110M,
# as analysis_commands.sh passes it), reads the corpus or task file the traces were made
# from, and the time-split and contamination steps read their own manifests.
if "$C3_PY" -c 'import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot' 2>/dev/null; then
    echo "[prereg] matplotlib $("$C3_PY" -c 'import matplotlib; print(matplotlib.__version__)'): figures on"
else
    echo "[prereg] WARNING: matplotlib does not import; figures will be skipped (the env tarball lacks it; every number is still in results.json)"
fi
"$C3_PY" - "$TRACES" "$CORPUS" "$TASKS_DIR" "${RUNS[@]}" <<'PY' || die "preflight failed"
import json, sys
from pathlib import Path
from c3.provenance import sha256_file

traces, corpus, tasks_dir, runs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
need = {"artifacts/split_manifest.json", "artifacts/split_manifest_gsm8k.json",
        "artifacts/split_manifest_mbpp.json", "artifacts/split_manifest_timesplit.json",
        "artifacts/contamination_llada-8b-base.json", "artifacts/contamination_dream-7b-base.json"}
bad = [f"{p} (missing)" for p in sorted(need) if not Path(p).is_file()]
tasks = {t["task"]: t for t in json.loads(Path("artifacts/task_manifest.json").read_text())["tasks"]}
want = {}           # data file -> recorded sha256
toks = {}           # (path, revision) -> first run that needs it
for run in runs:
    man = json.loads((traces / run / f"{run}.dataset.json").read_text())
    data = man.get("dataset")
    if data == "timesplit":
        continue                                # time-split arms run E1/E2 only, no T8
    if data in tasks:
        want[tasks_dir / f"{data}_test.jsonl"] = tasks[data]["sha256"]
    else:                                       # FineWeb text (collect_traces.py's default label)
        want[corpus] = json.loads(Path("artifacts/split_manifest.meta.json").read_text())["corpus_sha256"]
    if "110m" in run:
        toks.setdefault(("gpt2", None), run)
    else:
        from c3.traces.models import KNOWN_MODELS
        spec = KNOWN_MODELS.get(man.get("model"))
        if spec is None:
            bad.append(f"{run}: model {man.get('model')!r} has no tokenizer in KNOWN_MODELS")
        else:
            toks.setdefault((spec["path"], spec.get("revision")), run)
for path, sha in want.items():
    if not path.is_file():
        bad.append(f"{path} (missing)")
    elif sha256_file(path) != sha:
        bad.append(f"{path} (sha256 differs from the recorded {sha[:12]})")
    else:
        print(f"[prereg] {path}: sha256 ok")
if toks:
    from transformers import AutoTokenizer
for (path, rev), run in toks.items():
    try:
        AutoTokenizer.from_pretrained(path, trust_remote_code=True, revision=rev)
        print(f"[prereg] tokenizer {path} (T8 of {run}): loads offline")
    except Exception as exc:                    # noqa: BLE001 -- report whatever it is
        bad.append(f"tokenizer {path}@{rev} for {run} does not load offline ({type(exc).__name__}); "
                   "fetch it once into the HF cache, or submit with HF_HUB_OFFLINE=0")
for b in bad:
    print(f"[prereg] PREFLIGHT: {b}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY

# ---------------------------------------------------------------- 5. the commands
# Resumable.  The commands call PY once per step; a step whose --out holds a stamp for
# this commit and these arguments was finished by an earlier job into the same OUT, and
# is kept rather than redone.  The steps are deterministic (seeded resamples), so a job
# that runs out of walltime is submitted again as it was.  The stamp is written only
# once the step has exited 0.  It leaves out --n-jobs: each Eq. 2/3 resample is seeded
# by its index and every worker runs one BLAS thread, so any N_JOBS >= 2 gives the same
# numbers, and a resubmission may change it.
prereg_py() {
    local out= prev= a key=() stamp t0=$SECONDS
    local args=("$@")
    if [ "${C3_E1E2_ONLY:-0}" = 1 ]; then
        for a in "$@"; do
            [ "$prev" = --out ] && out=$a
            prev=$a
        done
        if [ -n "$out" ]; then
            local rel=${out#"$OUT"/} label
            label=${rel#c1/}
            case "$rel" in
                c1/*/*|c1/*_knots5|c1/*_knots13|c1/*_additive|c1/*_clean|c1/*TimeSplit) label= ;;
                c1/*) ;;
                *) label= ;;
            esac
            if [ "$1" = scripts/summarize_c1.py ] && [ "$rel" = c1/summary ]; then
                :                       # the Holm summary: runs as is, takes no --skip-regression
            elif [ "$rel" = "$out" ] || [ -z "$label" ] || [ "$1" != scripts/analyze_traces.py ]; then
                echo "[prereg] C3_E1E2_ONLY: skipped $out"
                return 0
            else
                case " ${args[*]} " in *" --skip-regression "*) ;; *) args+=(--skip-regression) ;; esac
            fi
        fi
        out= prev=
    fi
    set -- "${args[@]}"
    for a in "$@"; do
        [ "$prev" = --out ] && out=$a
        [ "$a" = --n-jobs ] || [ "$prev" = --n-jobs ] || key+=("${a//$TRACES/\$TRACES}")
        prev=$a
    done
    stamp="$C3_GIT_COMMIT ${key[*]}"
    if [ -n "$out" ] && [ "$(cat "$out/.prereg-done" 2>/dev/null)" = "$stamp" ]; then
        echo "[prereg] kept $out: finished by an earlier job at this commit"
        return 0
    fi
    "$PREREG_PY" "$@" || return
    if [ -n "$out" ]; then
        printf '%s\n' "$stamp" > "$out/.prereg-done"
        echo "[prereg] step done in $((SECONDS - t0)) s: $out"
    fi
}
export -f prereg_py
export PREREG_PY="$PY" PY=prereg_py
echo "[prereg] TRACES=$TRACES (${RUNS[*]:-no runs}) OUT=$OUT N_JOBS=$N_JOBS PY=$PREREG_PY"
t0=$SECONDS
bash prereg/analysis_commands.sh
echo "[prereg] analysis_commands.sh finished in $((SECONDS - t0)) s -> $OUT"
