#!/bin/bash
# T12 on Jarvis Labs through the official `jl` CLI, driven from the laptop (Git Bash on Windows
# or any Unix shell).  Phases with a laptop step in between: the final grids, the matching rule
# and the calibrators must be on origin before the first report-half generation.
#
# ONCE (the dedicated key is the only new thing; read "Pausing" below before the first launch)
#   uv tool install jarvislabs                           # jl 0.2.x
#   jl setup                                             # your own token, for jl itself
#   jl ssh-key add ~/.ssh/id_ed25519.pub --name laptop   # jl run/upload/exec need an SSH key
#   cp .gitattributes .git/info/attributes               # in the MAIN checkout, once: every worktree
#                                                        # then checks *.sh out with LF endings
#   # Jarvis dashboard > Settings > API keys: create a DEDICATED key for this sweep (name it
#   # t12-sweep).  In the shell you launch from:
#   read -rs JL_API_KEY; export JL_API_KEY                # hidden prompt: no literal key in the shell
#                                                        # history; launch refuses without it; never put
#                                                        # it in a file, a commit or a message
#
# THE SEQUENCE (every `launch` asks for `yes`; YES=1 skips that)
#   1. bash scripts/jl_t12.sh prepare smoke <commit>   && bash scripts/jl_t12.sh launch smoke
#        1 x RTX-PRO6000, hard 20 min.  Rehearses setup, verify-gpu, the model and 4 fit-half
#        problems -- and the WATCHDOG: while it runs, close the laptop terminal (kill the launch)
#        and confirm in the dashboard that the instance pauses itself after the job ends + the
#        20 min grace (T12_FETCH_GRACE_MIN=2 shortens it for the rehearsal), or at the deadline +
#        20 min.  Then `bash scripts/jl_t12.sh fetch smoke` (resumes with 1 GPU, fetches, pauses).
#   2. bash scripts/jl_t12.sh prepare calib <commit>   && bash scripts/jl_t12.sh launch calib
#        2 x RTX-PRO6000 in IN1, <= 2.5 h.  Fetches and pauses by itself ->
#        runs/t12/jl-calib-<sha7>/cal-grids/t12_grid_{gsm8k,mbpp}.json
#        (ON=<smoke instance id> reuses the smoke instance's venv and model: JL_NUM_GPUS stays 2,
#        the instance is resumed with 2 GPUs.)
#   3. review the grids, then push them:
#        cp runs/t12/jl-calib-<sha7>/cal-grids/t12_grid_*.json prereg/
#        git add prereg && git commit && git push
#   4. bash scripts/jl_t12.sh prepare report <pushed-commit> && bash scripts/jl_t12.sh launch report
#        refuses unless the commit is an ancestor of the branch on origin (git ls-remote) with both
#        grids.  <= 9.0 h.  -> runs/t12/jl-report-<sha7>/sweep-{gsm8k,mbpp}/ (then score_t12.py)
#   5. If the report stopped by its budget (exit 3), a link dropped for good, or an instance died:
#        a. SAME instance, first choice (its /home holds the venv, the model and the partial JSONLs):
#             NEW_BUDGET=1 bash scripts/jl_t12.sh prepare report <same-commit>
#             ON=<instance id> bash scripts/jl_t12.sh launch report
#           (resumes it with JL_NUM_GPUS GPUs; the id after a resume may differ, it is re-read.
#           Without NEW_BUDGET=1 the instance's own budget file is honoured, and a spent budget
#           refuses to run.)
#        b. FRESH instance, fallback: pack the partial results fetched earlier into the upload.
#             RESUME_FROM=runs/t12/jl-report-<sha7> bash scripts/jl_t12.sh prepare report <same-commit>
#             bash scripts/jl_t12.sh launch report
#           (needs the same commit and the same NGPU as the partial run: checked at prepare.)
#   6. When the outputs are checked: `jl destroy <id>` by hand (nothing here destroys an
#      instance), and REVOKE the dedicated API key in the Jarvis dashboard.
#
# IF THE LAPTOP CRASHES OR LOSES THE NETWORK DURING `launch`, before the watchdog is armed (the
# 1-3 minutes between `jl create` returning and the guard watchdog reporting "armed" cannot be
# closed), run `jl list` and pause anything named t12-* by hand: `jl pause <id> --yes`.
#
#   bash scripts/jl_t12.sh status|watch|fetch|pause <phase>    # see below
#
# How it works.  `launch` (1) `jl create ... --json --yes` (or, with ON=<id>, `jl resume`), (2) uploads
# /home/.t12.env (the watchdog's secret) and the watchdog script and starts a "guard" watchdog,
# (3) `jl run <upload>.sh --on <id>`: copies ONE file to /home and runs it detached with bash,
# streaming the log.  A directory target would be synced with rsync, which Git Bash does not have,
# so `prepare` packs everything into one self-extracting file: a bootstrap, then (base64 after a
# marker) a tar of a `git bundle` of the commit (LF files; a Windows checkout's CRLF would change
# the frozen manifests' sha256), the Ada-only inputs from runs/jarvis-stage/ (sha256-checked here
# against scripts/jarvis_t12.inputs.sha256 at that commit), job.env (the knobs below) and, with
# RESUME_FROM, the partial shards.  The bootstrap checks git, starts the job's watchdog (it refuses
# to run the job if no watchdog can be armed), clones the bundle to /home/ANLP-Project and execs
# `scripts/jarvis_t12.sh job <phase>`, which has its own global deadline and ALWAYS ends.
#
# Pausing.  jl's ssh drops the log stream after a ~45 s outage and `jl run` then applies no
# lifecycle action, so a laptop-side pause can never be the only one.  Every instance therefore
# carries its own watchdog (scripts/t12_watchdog.sh, log /home/t12-watchdog.log):
#   * the job ends  -> T12_FETCH_GRACE_MIN (default 20) minutes for the laptop to fetch, then PAUSE;
#   * always        -> PAUSE at the global deadline + 20 min, even if the job is still running;
#   * the job's own supervisor restarts a watchdog that has died (ensure_watchdog, jarvis_t12.sh);
#   * resumes (fetch, ON=) start a fresh watchdog at once; a fetch in progress touches /home/.t12-hold
#     every 4 min so the pause waits for it (never past the hard limit).
# The pause is a `POST misc/pause?machine_id=` to the instance's regional Jarvis backend with the
# key in /home/.t12.env (looked up by NAME, the id changes on a resume), then `jl pause` as the fallback
# (jarvislabs is pip-installed with --target into /home/.t12-jl when the instance is armed).  The
# deadline is an absolute time on the instance's clock (/home/.t12-deadline, written when armed).
# THE KEY: `launch` writes JL_API_KEY (from YOUR environment, nothing else) into a chmod-600 temp
# file outside the repo, uploads it with `jl upload` and deletes the local copy at once (also on
# Ctrl+C).  It lives on the instance's /home until you destroy the instance.  Therefore: use a
# DEDICATED key (not your main one), export it only in the launching shell, and revoke it in the
# dashboard after the sweep.  Jarvis also pauses all instances when the balance runs out.
# After a successful fetch the laptop pauses at once; the watchdog's later pause is then a no-op
# (it finds the instance Paused, or is gone with the container).  If the link drops, `launch`
# no longer exits: it falls through to `watch` (polls every 2 min, fetches when the job has ended,
# pauses, and force-pauses 40 min after the job's budget).  `watch` survives an outage of the laptop's
# link or of the Jarvis API: a failed `jl list`/`jl status` is "unknown", never "nothing to pause";
# it retries every minute until the budget + 1 h, and if it ever exits non-zero while an instance
# might exist it prints a banner with the id/name and the exact `jl list` / `jl pause <id> --yes`.  `pause <phase>` pauses by hand (refuses
# while the job runs, FORCE=1 overrides); `status <phase>` shows the instance, the run, the log and the
# watchdog's log.  Fetching resumes a paused instance with ONE GPU (JL_FETCH_GPUS) so that it
# is not stranded when IN1 has no 2 free, and pauses it again.  Nothing here ever destroys an
# instance.  Every jl call has a timeout.
#
# Budget.  Defaults: calib 2.5 h, report 9.0 h, smoke 20 min.  The clock starts when `launch`
# starts (provisioning counts); the job and the watchdog get what is left.  Worst case at 207.36 INR
# per GPU-hour (2 GPU = 414.72 INR/h), every phase running to its deadline and then waiting out the
# watchdog's 20 min margin / fetch grace:
#     smoke   1 GPU x (0.34 + 0.33) h             =  138 INR
#     calib   2 GPU x (2.5  + 0.33) h             = 1175 INR
#     report  2 GPU x (9.0  + 0.33) h             = 3871 INR      total 5184 INR
#     + a 1-GPU fetch resume lasts <= 20 min (grace) = 69 INR each, 3 of them          5391 INR
# (storage while paused not included).  Before creating anything, `launch` reads the balance (+ grants)
# from `jl status --json` and the price from `jl gpus --json` (price_per_hour is PER GPU; the
# row for a container) and refuses if balance < price x GPUs x (budget + 0.5 h).
# T12_SKIP_WALLET_CHECK=1 overrides it, loudly.
#
# Knobs, baked into the upload by `prepare`: BUDGET_HOURS, NEW_BUDGET, CAL_GRID, CAL_MAX_HOURS,
# CHECK_LIMIT, CHECK_MAX_HOURS, CHECK_MAX_SPF, MAX_RESTARTS, RUN_ARGS, PRECISION, NGPU, ALLOW_BF16
# (see scripts/jarvis_t12.sh), RESUME_FROM (a directory, see 5b).  At `launch`: JL_GPU (RTX-PRO6000),
# JL_NUM_GPUS (2; smoke 1; must equal the baked NGPU / RESUME_FROM's), JL_REGION (IN1), JL_STORAGE (100), JL_TEMPLATE (pytorch), JL_FETCH_GPUS (1),
# T12_FETCH_GRACE_MIN (20), T12_SKIP_WALLET_CHECK, ON=<machine id> (use an existing instance, resumed if
# paused), YES=1, STAGE (the directory with artifacts/ and runs/ from Ada; default runs/jarvis-stage).
set +x
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
UP=runs/jl-upload
JL_GPU="${JL_GPU:-RTX-PRO6000}"
JL_NUM_GPUS="${JL_NUM_GPUS:-}"
JL_REGION="${JL_REGION:-IN1}"
JL_STORAGE="${JL_STORAGE:-100}"
JL_TEMPLATE="${JL_TEMPLATE:-pytorch}"
JL_FETCH_GPUS="${JL_FETCH_GPUS:-1}"
NAP="${T12_NAP:-20}"
WATCH_SLEEP="${T12_WATCH_SLEEP:-120}"
export PATH="$PATH:$HOME/.local/bin"
# Git Bash would rewrite the instance paths we pass to jl (/home/... -> C:/Program Files/Git/home/...)
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"
KNOBS=(BUDGET_HOURS NEW_BUDGET CAL_GRID CAL_MAX_HOURS CHECK_LIMIT CHECK_MAX_HOURS CHECK_MAX_SPF
       MAX_RESTARTS RUN_ARGS PRECISION NGPU ALLOW_BF16 POLL_SECONDS STOP_GRACE)

die() { echo "[jl_t12] FATAL: $*" >&2; exit 1; }
info() { echo "[jl_t12] $*"; }
PYJ="$(command -v python || command -v python3 || true)"
if [ -z "$PYJ" ] || ! "$PYJ" -c 'import json' 2>/dev/null; then die "no working python on PATH (needed to read jl's JSON)"; fi

check_phase() {
    case "${1:-}" in
        calib|smoke|report|report-gsm8k|report-mbpp) ;;
        *) die "phase: calib | smoke | report | report-gsm8k | report-mbpp" ;;
    esac
}
is_report() { case "$1" in report*) return 0 ;; *) return 1 ;; esac; }
# (scripts/jarvis_t12.sh phase_hours says the same; a test compares them)
default_hours() { case "$1" in calib) echo 2.5 ;; smoke) echo 0.34 ;; *) echo 9.0 ;; esac; }
default_gpus() { case "$1" in smoke) echo 1 ;; *) echo 2 ;; esac; }

# ---- prepare: the upload -------------------------------------------------------------------

# The staging directory of the Ada-only inputs (a worktree has no runs/ of its own).
find_stage() {
    local c
    for c in "${STAGE:-}" "$ROOT/runs/jarvis-stage" "$(git rev-parse --git-common-dir)/../runs/jarvis-stage"; do
        [ -n "$c" ] && [ -d "$c/artifacts" ] && { (cd "$c" && pwd); return 0; }
    done
    return 1
}

# The self-extracting bootstrap (placeholders @PHASE@ and @COMMIT@).  T12_BASE, T12_DRYRUN and
# T12_SECRET exist for `prepare`'s own rehearsal on this machine (T12_DRYRUN skips the watchdog).
write_bootstrap() {
    cat > "$1" <<'EOF'
#!/bin/bash
# T12 upload for `jl run`, generated by scripts/jl_t12.sh prepare (phase @PHASE@, commit @COMMIT@).
# jl copies this single file to the instance and runs it with bash.  After the marker line
# below comes a base64 tar: repo.bundle (git bundle of the commit), inputs/ (Ada-only files),
# job.env (the knobs), resume/ (partial shards, optional).  The checkout is made from the
# bundle, so its files are LF.
set -euo pipefail
PHASE=@PHASE@
COMMIT=@COMMIT@
if [ -n "${T12_BASE:-}" ]; then BASE="$T12_BASE"; elif [ "$(id -u)" = 0 ]; then BASE=/home; else BASE="$HOME"; fi
SECRET="${T12_SECRET:-/home/.t12.env}"
WD="${T12_WATCHDOG:-/home/.t12-watchdog.sh}"
# 1. NO JOB WITHOUT A WATCHDOG.  The secret and the watchdog script were uploaded by the laptop's
#    `launch`; the watchdog outlives this job, the ssh session and the laptop.
if [ "${T12_DRYRUN:-0}" != 1 ]; then
    [ -f "$SECRET" ] || { echo "[bootstrap] FATAL: no $SECRET: refusing to start a job that nothing would pause" >&2; exit 1; }
    [ -f "$WD" ] || { echo "[bootstrap] FATAL: no $WD: refusing to start a job that nothing would pause" >&2; exit 1; }
    chmod 600 "$SECRET"
    nonce="$$.$(date +%s)"
    T12_WD_NONCE="$nonce" setsid nohup bash "$WD" job "$$" "$PHASE" >/dev/null 2>&1 </dev/null &
    armed=0
    for _ in $(seq 1 "${T12_ARM_WAIT:-30}"); do
        if tail -n 5 "${T12_WD_LOG:-/home/t12-watchdog.log}" 2>/dev/null | grep -q "nonce $nonce;"; then armed=1; break; fi
        sleep 1
    done
    [ "$armed" = 1 ] || { echo "[bootstrap] FATAL: the watchdog did not arm (see /home/t12-watchdog.log); no job" >&2; exit 1; }
    echo "[bootstrap] watchdog armed for pid $$"
    export T12_WATCHED_PID=$$      # the job restarts the watchdog for this pid if it dies
fi
REPO="$BASE/ANLP-Project"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
IN="$BASE/c3-t12-upload-$PHASE"
rm -rf "$IN"
mkdir -p "$IN"
start="$(awk '/^__T12_PAYLOAD__$/ { print NR + 1; exit }' "$SELF")"
tail -n +"$start" "$SELF" | base64 -d | tar -xzf - -C "$IN"
# 2. git (a resumed container loses what apt installed outside /home)
if ! command -v git >/dev/null 2>&1; then
    echo "[bootstrap] git is missing: installing"
    (apt-get update -qq && apt-get install -y -qq git) >/dev/null 2>&1 || true
fi
command -v git >/dev/null 2>&1 || { echo "[bootstrap] FATAL: git is not installed and apt-get could not install it" >&2; exit 1; }
git config --global --get-all safe.directory 2>/dev/null | grep -qx "$REPO" || git config --global --add safe.directory "$REPO"   # repo dir owned by another uid on Jarvis containers
[ -d "$REPO/.git" ] || git init -q "$REPO"
git -C "$REPO" config core.autocrlf false
git -C "$REPO" fetch -q "$IN/repo.bundle" "+refs/jl-t12/commit:refs/jl-t12/commit"
[ "$(git -C "$REPO" rev-parse refs/jl-t12/commit)" = "$COMMIT" ] || { echo "bundle does not hold $COMMIT" >&2; exit 1; }
git -C "$REPO" checkout -q -f --detach "$COMMIT"
[ ! -d "$IN/inputs" ] || cp -a "$IN/inputs/." "$REPO/"
# 3. partial shards of an earlier run (RESUME_FROM): only where the instance has none
for src in "$IN"/resume/sweep-*/t12.shard*of*.jsonl; do
    [ -e "$src" ] || continue
    d="$REPO/runs/t12/$(basename "$(dirname "$src")")"
    b="$(basename "$src")"
    if [ -e "$d/$b" ]; then echo "[bootstrap] keeping $d/$b (the instance has it already)"; continue; fi
    mkdir -p "$d"
    cp -p "$src" "$d/$b"
    cp -p "${src%.jsonl}.meta.json" "$d/${b%.jsonl}.meta.json"
    echo "[bootstrap] restored $d/$b ($(grep -c . "$src" || true) records)"
done
set -a
# shellcheck disable=SC1091
. "$IN/job.env"
set +a
if [ "${T12_DRYRUN:-0}" != 1 ]; then
    # the budget left (provisioning counted); read in a subshell so the key never enters the job
    T12_BUDGET_REMAINING_S="$(set -a; . "$SECRET" >/dev/null 2>&1; printf '%s' "${T12_BUDGET_REMAINING_S:-}")"
    export T12_BUDGET_REMAINING_S
    # the absolute deadline the laptop wrote when it armed the instance (this machine's clock)
    DLF="${T12_DEADLINE_FILE:-/home/.t12-deadline}"
    if [ -s "$DLF" ]; then T12_DEADLINE_EPOCH="$(cat "$DLF")"; export T12_DEADLINE_EPOCH; fi
fi
cd "$REPO"
echo "[bootstrap] $PHASE at $(git rev-parse HEAD) in $REPO"
if [ "${T12_DRYRUN:-0}" = 1 ]; then
    n="$(git ls-files --eol | grep -c 'w/crlf' || true)"
    echo "[dryrun] CRLF files in the checkout: $n"
    [ "$n" = 0 ]
    bash scripts/jarvis_t12.sh verify-inputs gsm8k
    exit 0
fi
exec bash scripts/jarvis_t12.sh job "$PHASE"
exit 1
__T12_PAYLOAD__
EOF
    # a CRLF work tree (core.autocrlf on Windows) would put CRs into the heredoc above
    sed -i "s/\r$//; s/@PHASE@/$2/g; s/@COMMIT@/$3/g" "$1"
}

# resume_pack SRC DEST FULL NGPU: copy sweep-*/t12.shard*of<N>.{jsonl,meta.json} of an earlier
# run into DEST; refuse unless every shard was made at commit FULL with NGPU shards.
resume_pack() {
    "$PYJ" - "$1" "$2" "$3" "$4" <<'PYEOF'
import json, shutil, sys
from pathlib import Path
src, dest, commit, ngpu = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
n = 0
for meta in sorted(src.glob("sweep-*/t12.shard*of*.meta.json")):
    jl = meta.with_name(meta.name.replace(".meta.json", ".jsonl"))
    m = json.loads(meta.read_text())
    got = m["starts"][0]["git_commit"]
    if got != commit:
        sys.exit(f"RESUME_FROM: {meta} was made at commit {got}, this upload is {commit}: the same commit is required")
    if int(m["num_shards"]) != ngpu or f"of{ngpu}." not in meta.name:
        sys.exit(f"RESUME_FROM: {meta} has {m['num_shards']} shard(s), this run has NGPU={ngpu}: the same number of GPUs is required")
    if not jl.exists():
        sys.exit(f"RESUME_FROM: {jl} is missing next to {meta.name}")
    out = dest / meta.parent.name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta, out / meta.name)
    shutil.copy2(jl, out / jl.name)
    n += 1
    print(f"[resume] {meta.parent.name}/{jl.name}: {sum(1 for _ in jl.open())} records")
if n == 0:
    sys.exit(f"RESUME_FROM: no sweep-*/t12.shard*of*.meta.json under {src}")
PYEOF
}

cmd_prepare() {
    local phase="${1:-}" commit="${2:-}" full short dir stage want path got src name k t ngpu
    check_phase "$phase"
    [ -n "$commit" ] || die "usage: prepare <phase> <commit>"
    full="$(git rev-parse --verify "$commit^{commit}")" || die "no such commit: $commit"
    short="${full:0:7}"
    for path in scripts/jarvis_t12.sh scripts/jarvis_t12.inputs.sha256 scripts/run_t12.py scripts/t12_watchdog.sh; do
        git cat-file -e "$full:$path" 2>/dev/null || die "$path is not in $short"
    done
    if [ "$phase" = calib ] || [ "$phase" = smoke ]; then
        for path in prereg/t12_cal.json scripts/pick_knobs.py prereg/t12_sanity.json; do
            git cat-file -e "$full:$path" 2>/dev/null || die "$path is not in $short"
        done
    fi
    if is_report "$phase"; then
        for t in gsm8k mbpp; do
            git cat-file -e "$full:prereg/t12_grid_$t.json" 2>/dev/null || die "prereg/t12_grid_$t.json is not in $short: commit the calibrated grids first"
        done
    fi
    if [ "$phase" = smoke ] && [ -n "${BUDGET_HOURS:-}" ]; then
        awk -v h="$BUDGET_HOURS" 'BEGIN { exit !(h > 0 && h <= 0.34) }' || die "smoke has a hard 20 min cap: BUDGET_HOURS=$BUDGET_HOURS is more than 0.34"
    fi
    ngpu="${NGPU:-${JL_NUM_GPUS:-$(default_gpus "$phase")}}"
    stage="$(find_stage)" || die "no staging directory with artifacts/ and runs/ (STAGE=...; the Ada inputs copied to runs/jarvis-stage)"
    dir="$UP/$phase-$short"
    rm -rf "$dir"
    mkdir -p "$dir/payload/inputs"

    # The inputs: what the commit tracks is checked against its blob (LF), the rest against the
    # staging copy, which goes into the upload.  A task file missing from both is built on the
    # instance from the pinned hub revision and checked there against the same list.
    info "inputs from $stage, checked against scripts/jarvis_t12.inputs.sha256 at $short"
    while read -r want path; do
        want="${want%$'\r'}"
        path="${path%$'\r'}"
        case "$want" in ''|'#'*) continue ;; esac
        if git cat-file -e "$full:$path" 2>/dev/null; then
            got="$(git show "$full:$path" | sha256sum | cut -d' ' -f1)"
            src="in $short"
        elif [ -f "$stage/$path" ]; then
            got="$(sha256sum "$stage/$path" | cut -d' ' -f1)"
            src="staged"
        else
            case "$path" in
                */tasks/*) echo "[inputs] $path: built on the instance (pinned hub revision), checked there"; continue ;;
                *) die "$path is neither in $short nor in $stage" ;;
            esac
        fi
        [ "$got" = "$want" ] || die "$path ($src): sha256 $got, want $want"
        echo "[inputs] ok $path ($src)"
        if [ "$src" = staged ]; then
            mkdir -p "$dir/payload/inputs/$(dirname "$path")"
            cp "$stage/$path" "$dir/payload/inputs/$path"
        fi
    done < <(git show "$full:scripts/jarvis_t12.inputs.sha256")

    git update-ref refs/jl-t12/commit "$full"
    # shellcheck disable=SC2064
    trap "git update-ref -d refs/jl-t12/commit" EXIT
    git bundle create "$dir/payload/repo.bundle" refs/jl-t12/commit >/dev/null 2>&1 || die "git bundle failed"
    git update-ref -d refs/jl-t12/commit
    trap - EXIT

    # the watchdog of this commit (launch uploads it; the bootstrap refuses to run without it)
    git show "$full:scripts/t12_watchdog.sh" | sed 's/\r$//' > "$dir/t12_watchdog.sh"
    bash -n "$dir/t12_watchdog.sh" || die "scripts/t12_watchdog.sh has a syntax error at $short"

    if [ -n "${RESUME_FROM:-}" ]; then
        is_report "$phase" || die "RESUME_FROM is for a report phase (it packs sweep-* shards)"
        [ -d "$RESUME_FROM" ] || die "RESUME_FROM=$RESUME_FROM is not a directory (runs/t12/jl-report-<sha7> of an earlier fetch)"
        mkdir -p "$dir/payload/resume"
        resume_pack "$RESUME_FROM" "$dir/payload/resume" "$full" "$ngpu" || die "RESUME_FROM refused (see above)"
        echo "$ngpu" > "$dir/RESUME_NGPU"      # launch refuses another JL_NUM_GPUS before creating anything
    fi

    : > "$dir/payload/job.env"
    for k in "${KNOBS[@]}"; do
        [ -z "${!k:-}" ] || printf 'export %s=%q\n' "$k" "${!k}" >> "$dir/payload/job.env"
    done
    if [ "$phase" = smoke ] && [ -z "${NGPU:-}" ]; then echo 'export NGPU=1' >> "$dir/payload/job.env"; fi
    echo "$full" > "$dir/COMMIT"
    echo "${BUDGET_HOURS:-$(default_hours "$phase")}" > "$dir/HOURS"
    name="t12-$phase-$short.sh"
    write_bootstrap "$dir/$name" "$phase" "$full"
    bash -n "$dir/$name" || die "the generated bootstrap has a syntax error"
    tar -czf "$dir/payload.tgz" -C "$dir/payload" .
    base64 "$dir/payload.tgz" >> "$dir/$name"
    echo "$name" > "$dir/NAME"
    {
        echo "phase $phase  commit $full  prepared $(date -u +%FT%TZ)"
        echo "upload $name  sha256 $(sha256sum "$dir/$name" | cut -d' ' -f1)  $(du -h "$dir/$name" | cut -f1)"
        echo "bundle sha256 $(sha256sum "$dir/payload/repo.bundle" | cut -d' ' -f1)"
        [ -z "${RESUME_FROM:-}" ] || echo "resume from $RESUME_FROM"
        cat "$dir/payload/job.env"
    } > "$dir/MANIFEST.txt"
    echo "$dir" > "$UP/.last-$phase"

    # Rehearsal: unpack the upload exactly as the instance will, check the checkout is LF and
    # the GSM8K inputs verify; nothing leaves this machine.
    local tmp
    tmp="$(mktemp -d)"
    # (native git on Windows needs the /tmp path conversion that is off for jl)
    if (cd "$dir" && env -u MSYS_NO_PATHCONV -u MSYS2_ARG_CONV_EXCL T12_BASE="$tmp" T12_DRYRUN=1 bash "$name") > "$dir/rehearsal.log" 2>&1; then
        info "rehearsal ok ($(grep -c . "$dir/rehearsal.log") lines, $dir/rehearsal.log)"
    else
        tail -n 15 "$dir/rehearsal.log" >&2
        rm -rf "$tmp"
        die "the rehearsal of the upload failed"
    fi
    rm -rf "$tmp"
    cat "$dir/MANIFEST.txt"
    info "prepared $dir/$name; next: bash scripts/jl_t12.sh launch $phase"
}

# ---- jl helpers ----------------------------------------------------------------------------

need_tools() {
    command -v jl >/dev/null 2>&1 || die "jl is not installed: uv tool install jarvislabs (then put ~/.local/bin on PATH)"
    command -v ssh >/dev/null 2>&1 || die "ssh must be on PATH (jl uses it)"
    command -v scp >/dev/null 2>&1 || die "scp must be on PATH (jl uses it)"
}

need_jl() {
    need_tools
    timeout 120 jl status --json >/dev/null 2>&1 || die "jl is not logged in (run: jl setup), JL_API_KEY is wrong, or the API is unreachable"
}

need_key() {
    [ -n "${JL_API_KEY:-}" ] || die "JL_API_KEY is not set.  The instance pauses ITSELF through the Jarvis API, which needs a key on the instance.
    Create a DEDICATED key (Jarvis dashboard > Settings > API keys, e.g. named t12-sweep), then in THIS shell:
        export JL_API_KEY=<that key>
    and revoke it in the dashboard when the sweep is done.  Without it no job is started: nothing could pause it if the laptop link drops."
}

# Phase directory of the last `prepare`.
phase_dir() {
    [ -f "$UP/.last-$1" ] || die "no upload for '$1': run prepare first"
    cat "$UP/.last-$1"
}

# "machine_id status" of the newest instance with this name; nothing when the list is read and
# has none; "? unknown" when `jl list` FAILED (API down, link down): never "no instance".
find_instance() {
    local out
    out="$(timeout 120 jl list --json 2>/dev/null)" || { echo "? unknown"; return 0; }
    printf '%s' "$out" | "$PYJ" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    print("? unknown"); sys.exit(0)
for i in (d.get("instances", []) if isinstance(d, dict) else d):
    if i.get("name") == sys.argv[1]:
        print(i["machine_id"], i["status"])
' "$1" | tr -d '\r' | tail -n 1
}

# "name status num_gpus" of the instance with this machine id, or nothing.
instance_by_id() {
    local out
    out="$(timeout 120 jl list --json 2>/dev/null)" || { echo "? unknown ?"; return 0; }
    printf '%s' "$out" | "$PYJ" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    print("? unknown ?"); sys.exit(0)
for i in (d.get("instances", []) if isinstance(d, dict) else d):
    if str(i.get("machine_id")) == sys.argv[1]:
        print(i.get("name"), i.get("status"), i.get("num_gpus"))
' "$1" | tr -d '\r' | tail -n 1
}

# The latest local run record of an instance ("?" when `jl run list` failed).
latest_run_id() {
    local out
    out="$(timeout 120 jl run list --json --machine "$1" --limit 1 2>/dev/null)" || { echo "?"; return 0; }
    printf '%s' "$out" | "$PYJ" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    print("?"); sys.exit(0)
if d:
    print(d[0]["run_id"])
' | tr -d '\r'
}

# running | succeeded | failed | unknown | none (no run was ever started on the instance)
run_state() {
    local id="$1" rid="$2"
    [ -n "$rid" ] || { echo none; return 0; }
    [ "$rid" != "?" ] || { echo unknown; return 0; }
    { timeout 180 jl run status "$rid" --json 2>/dev/null || true; } | "$PYJ" -c '
import json, sys
try:
    print(json.load(sys.stdin).get("state", "unknown"))
except ValueError:
    print("unknown")
' | tr -d '\r'
}

# gpu_row GPU REGION: "free price" of the container row, "noregion regions", "absent types" or "?".
# jl gpus lists a row per (GPU type, region, workload): `workload_type` is "container", "vm" or
# null (both); a VM row must not give the price or the free count of a container.  price_per_hour
# is the price of ONE GPU (jl deploy multiplies it by gpus_per_worker).
gpu_row() {
    { timeout 120 jl gpus --json 2>/dev/null || true; } | "$PYJ" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    print("?"); sys.exit(0)
rows = d.get("gpus", d) if isinstance(d, dict) else d
rows = [r for r in rows if r.get("workload_type") in ("container", None)]
want, reg = sys.argv[1].lower(), sys.argv[2].upper()
same = [r for r in rows if str(r.get("gpu_type", "")).lower() == want]
hit = [r for r in same if str(r.get("region", "")).upper() == reg]
if hit:
    print(hit[0].get("effective_num_free_devices", hit[0].get("num_free_devices")), hit[0].get("price_per_hour"))
elif same:
    print("noregion", ",".join(sorted({str(r.get("region")) for r in same})))
else:
    print("absent", ",".join(sorted({str(r.get("gpu_type")) for r in rows})))
' "$1" "$2" | tr -d '\r'
}

# wallet_check HOURS NGPUS PRICE_PER_GPU_HOUR: refuse unless balance (+ grants) covers
# price x GPUs x (HOURS + 0.5 h).
wallet_check() {
    local hours="$1" n="$2" price="$3" out verdict have need cur
    if [ "${T12_SKIP_WALLET_CHECK:-0}" = 1 ]; then
        echo "################################################################################" >&2
        echo "# WARNING: T12_SKIP_WALLET_CHECK=1: the wallet is NOT checked.  If the balance runs out" >&2
        echo "# Jarvis pauses the instance in the middle of the job.                                 #" >&2
        echo "################################################################################" >&2
        return 0
    fi
    out="$({ timeout 120 jl status --json 2>/dev/null || true; } | "$PYJ" -c '
import json, sys
hours, n = float(sys.argv[1]), int(sys.argv[2])
try:
    price = float(sys.argv[3])
    d = json.load(sys.stdin)
    b = d["balance"]
    have = float(b["balance"]) + float(b.get("grants") or 0)
except (ValueError, KeyError, TypeError):
    print("unreadable"); sys.exit(0)
need = price * n * (hours + 0.5)
print("ok" if have >= need else "short", "%.2f" % have, "%.2f" % need, d.get("currency", ""))
' "$hours" "$n" "$price" | tr -d '\r')"
    read -r verdict have need cur <<< "$out"
    case "$verdict" in
        ok) info "wallet: $have $cur (balance + grants) covers $need $cur = $price per GPU-hour x $n GPU x ($hours h + 0.5 h)" ;;
        short) die "the wallet is short: balance + grants $have $cur < $need $cur = $price per GPU-hour x $n GPU x ($hours h + 0.5 h).  Top up, shorten BUDGET_HOURS, or (at your risk) T12_SKIP_WALLET_CHECK=1" ;;
        *) die "could not read the balance (jl status --json) or the GPU price (jl gpus --json: no price for $JL_GPU in $JL_REGION); T12_SKIP_WALLET_CHECK=1 overrides at your risk" ;;
    esac
}

# origin_has_commit FULL: FULL is an ancestor of a branch that is on origin NOW (git ls-remote),
# not merely of a stale remote-tracking ref.  Only the one branch is fetched, if needed: never a
# bare `git fetch` (origin has `Abhiraj` and `abhiraj/*`, which collide on a case-insensitive
# file system).  Sets ORIGIN_BRANCH and ORIGIN_SHA.
origin_has_commit() {
    local full="$1" cands b sha cur seen=" "
    cur="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    cands="${T12_ORIGIN_BRANCH:-}"$'\n'"$cur"$'\n'"$(git branch -r --contains "$full" 2>/dev/null | sed 's/^[[:space:]*]*//' | grep '^origin/' | grep -v -e '->' | sed 's#^origin/##' || true)"
    while IFS= read -r b; do
        case "$b" in ''|HEAD) continue ;; esac
        case "$seen" in *" $b "*) continue ;; esac
        seen="$seen$b "
        sha="$(timeout 60 git ls-remote origin "refs/heads/$b" 2>/dev/null | head -n 1 | cut -f1 || true)"
        [ -n "$sha" ] || continue
        git cat-file -e "$sha^{commit}" 2>/dev/null || timeout 300 git fetch -q origin "refs/heads/$b" 2>/dev/null || continue
        if git merge-base --is-ancestor "$full" "$sha" 2>/dev/null; then
            ORIGIN_BRANCH="$b"
            ORIGIN_SHA="$sha"
            return 0
        fi
    done <<< "$cands"
    return 1
}

pause_instance() {
    local id="$1" name="$2" t ok=0 now
    for t in 1 2 3; do
        if timeout 180 jl pause "$id" --yes; then ok=1; break; fi
        info "jl pause failed (try $t/3)"
        sleep "$NAP"
    done
    now="$(find_instance "$name" | cut -d' ' -f2)"
    if [ "$ok" = 1 ] || [ "$now" = Paused ]; then
        info "PAUSED instance $id ($name); /home is kept (a small storage cost continues). Destroy only after the outputs are checked."
        return 0
    fi
    cat >&2 <<EOF
================================================================================
 PAUSE FAILED: instance $id ($name) KEEPS BILLING.  Pause it NOW:
     jl pause $id --yes        (or the dashboard, https://jarvislabs.ai/)
 (its own watchdog, /home/t12-watchdog.log, will also try)
================================================================================
EOF
    return 1
}

# ---- the instance's watchdog ---------------------------------------------------------------

SECRET_DIR=""
cleanup_secret() {
    if [ -n "${SECRET_DIR:-}" ]; then rm -rf "$SECRET_DIR"; fi
    SECRET_DIR=""
}

# write_secret FILE NAME ID REMAINING_S: the watchdog's env file.  The key goes through the
# printf builtin only: never a command line, never echoed.
write_secret() {
    { set +x; } 2>/dev/null
    {
        printf 'JL_API_KEY=%q\n' "$JL_API_KEY"
        printf 'T12_INSTANCE_NAME=%q\n' "$2"
        printf 'T12_MACHINE_ID=%q\n' "$3"
        printf 'T12_BUDGET_REMAINING_S=%q\n' "$4"
        printf 'T12_FETCH_GRACE_MIN=%q\n' "${T12_FETCH_GRACE_MIN:-20}"
    } > "$1"
}

# guard_cmd NONCE GRACE_S SET_ID DEADLINE_IN_S: the command (run on the instance through `jl exec`)
# that (re)starts the watchdog in "fetch" mode (no job): a guard for the minutes before the job's
# bootstrap starts its own, and for a resume that only fetches.  The FIRST thing it does is chmod
# the secret 600 (scp does not keep the mode).  With DEADLINE_IN_S it writes the absolute deadline
# (the instance's clock) and installs jarvislabs for the fallback pause; with SET_ID it rewrites
# T12_MACHINE_ID in the secret (the id changes on a resume).
guard_cmd() {
    local c="umask 077; chmod 600 /home/.t12.env"
    [ -z "$4" ] || c="$c; echo \$((\$(date +%s) + $4)) > /home/.t12-deadline"
    [ -z "$3" ] || c="$c; sed -i 's/^T12_MACHINE_ID=.*/T12_MACHINE_ID=$3/' /home/.t12.env"
    c="$c; if [ -f /home/.t12-watchdog.sh.new ]; then mv -f /home/.t12-watchdog.sh.new /home/.t12-watchdog.sh; fi; chmod 700 /home/.t12-watchdog.sh"
    [ -z "$2" ] || c="$c; T12_FETCH_GRACE_S=$2; export T12_FETCH_GRACE_S"
    c="$c; T12_WD_NONCE=$1; export T12_WD_NONCE"
    [ -z "$4" ] || c="$c; (nohup python3 -m pip install -q --target /home/.t12-jl jarvislabs >/home/.t12-jl-install.log 2>&1 </dev/null &)"
    c="$c; (setsid nohup bash /home/.t12-watchdog.sh fetch </dev/null >/dev/null 2>&1 &); sleep 3; tail -n 3 /home/t12-watchdog.log"
    printf '%s' "$c"
}

# start_guard ID [GRACE_S [SET_ID [DEADLINE_AT]]]: true when THIS start of the watchdog reports
# "armed" (it echoes a fresh nonce: an old log line cannot pass; 3 tries, ssh may need a moment
# after a create or a resume).  GRACE_S replaces the grace of the minutes before the job's own
# watchdog takes over.  DEADLINE_AT is the laptop's epoch of the budget's end; the remaining
# seconds are computed again at every try.
start_guard() {
    local out t cmd nonce rem=""
    nonce="$(date +%s)$RANDOM"
    for t in 1 2 3; do
        [ -z "${4:-}" ] || rem=$(($4 - $(date +%s)))
        cmd="$(guard_cmd "$nonce" "${2:-}" "${3:-}" "$rem")"
        out="$(timeout 120 jl exec "$1" -- sh -c "$cmd" 2>&1 || true)"
        if printf '%s\n' "$out" | grep -q "nonce $nonce;"; then return 0; fi
        sleep "$NAP"
    done
    info "the watchdog did not report 'armed' on instance $1: $(printf '%s' "$out" | tail -n 2 | tr '\n' ' ')"
    return 1
}

# upload_try ID SRC DEST LOGDIR: jl upload, 3 tries.
upload_try() {
    local t
    for t in 1 2 3; do
        if timeout 300 jl upload "$1" "$2" "$3" >> "$4/launch.log" 2>&1; then return 0; fi
        sleep "$NAP"
    done
    return 1
}

# arm_instance DIR ID NAME T0 HOURS: upload the secret (written with umask 077 in a private temp
# directory OUTSIDE the repo, deleted at once) and the watchdog IN PARALLEL, then start the guard
# watchdog: this is the first thing done to an instance, right after it exists.
arm_instance() {
    local dir="$1" id="$2" name="$3" t0="$4" hours="$5" rem sfile sarg wsarg p1 p2 ok=1 dl
    dl=$((t0 + $(awk -v h="$hours" 'BEGIN { printf "%d", h * 3600 }')))
    rem=$((dl - $(date +%s)))
    [ "$rem" -ge 300 ] || die "the budget ($hours h) is used up by provisioning ($rem s left)"
    SECRET_DIR="$(umask 077; mktemp -d)" || die "mktemp -d failed"
    sfile="$SECRET_DIR/secret.env"
    (umask 077; write_secret "$sfile" "$name" "$id" "$rem")
    sarg="$sfile"
    wsarg="$(cd "$dir" && pwd)/t12_watchdog.sh"
    if command -v cygpath >/dev/null 2>&1; then sarg="$(cygpath -m "$sfile")"; wsarg="$(cygpath -m "$wsarg")"; fi
    upload_try "$id" "$sarg" /home/.t12.env "$dir" & p1=$!
    upload_try "$id" "$wsarg" /home/.t12-watchdog.sh.new "$dir" & p2=$!
    wait "$p1" || ok=0
    wait "$p2" || ok=0
    cleanup_secret
    [ "$ok" = 1 ] || die "could not upload the watchdog and its secret to instance $id: nothing would pause it; no job started (the instance will be paused)"
    start_guard "$id" 900 "" "$dl" || die "the watchdog is not running on instance $id; no job started (the instance will be paused)"
    info "watchdog armed on instance $id (pauses it ${T12_FETCH_GRACE_MIN:-20} min after the job, and at the budget + 20 min)"
}

# ---- fetch and finalize --------------------------------------------------------------------

# fetch_outputs PHASE ID RID: runs/t12-out/<phase>.tgz and the job log, into runs/t12/jl-<phase>-<sha7>/
# While it runs, a background loop touches /home/.t12-hold every 4 min (it counts as fresh for 10):
# the watchdog's pause waits for a fetch in progress, and for nothing longer.  A background job
# ignores Ctrl+C, so the loop checks every few seconds that its parent still exists, stops after
# T12_HOLD_MAX_S (1 h) whatever happens, and is killed by stop_holder (normal end, INT/TERM/HUP).
HOLDER_PID=""
stop_holder() {
    [ -z "${HOLDER_PID:-}" ] || kill "$HOLDER_PID" 2>/dev/null || true
    HOLDER_PID=""
}

# hold_loop ID PARENT_PID
hold_loop() {
    local end waited
    end=$(($(date +%s) + ${T12_HOLD_MAX_S:-3600}))
    while kill -0 "$2" 2>/dev/null && [ "$(date +%s)" -lt "$end" ]; do
        timeout 60 jl exec "$1" -- touch /home/.t12-hold >/dev/null 2>&1 || true
        waited=0
        while [ "$waited" -lt "${T12_HOLD_EVERY_S:-240}" ] && kill -0 "$2" 2>/dev/null; do
            sleep "${T12_HOLD_CHECK_S:-5}"
            waited=$((waited + ${T12_HOLD_CHECK_S:-5}))
        done
    done
}

fetch_outputs() {
    local parent=$BASHPID rc=0
    ( hold_loop "$2" "$parent" ) &
    HOLDER_PID=$!
    trap 'stop_holder; exit 130' INT TERM HUP
    fetch_outputs_now "$@" || rc=$?
    stop_holder
    trap - INT TERM HUP
    return "$rc"
}

fetch_outputs_now() {
    local phase="$1" id="$2" rid="$3" dir short out remote want got
    dir="$(phase_dir "$phase")"
    short="$(cut -c1-7 "$dir/COMMIT")"
    out="runs/t12/jl-$phase-$short"
    mkdir -p "$out"
    remote="$(timeout 120 jl exec "$id" -- sh -lc "ls -1 /home/ANLP-Project/runs/t12-out/$phase.tgz /home/*/ANLP-Project/runs/t12-out/$phase.tgz 2>/dev/null | head -n 1" | tr -d '\r' | tail -n 1)" || true
    [ -n "$remote" ] || { info "no runs/t12-out/$phase.tgz on instance $id"; return 1; }
    want="$(timeout 120 jl exec "$id" -- sh -lc "cut -d' ' -f1 $remote.sha256" | tr -d '\r' | tail -n 1)" || true
    (cd "$out" && timeout 1800 jl download "$id" "$remote" "$phase.tgz") || { info "jl download failed"; return 1; }
    got="$(sha256sum "$out/$phase.tgz" | cut -d' ' -f1)"
    [ -z "$want" ] || [ "$got" = "$want" ] || { info "the tarball's sha256 differs from the instance's ($got vs $want)"; return 1; }
    tar -xzf "$out/$phase.tgz" -C "$out" || { info "could not unpack $out/$phase.tgz"; return 1; }
    [ -z "$rid" ] || { timeout 600 jl run logs "$rid" > "$out/job.log" 2>/dev/null || true; }
    timeout 120 jl exec "$id" -- sh -lc "cp /home/t12-watchdog.log /home/ANLP-Project/runs/t12-out/ 2>/dev/null; true" >/dev/null 2>&1 || true
    info "FETCHED into $out ($(du -h "$out/$phase.tgz" | cut -f1), sha256 ${got:0:12}); $(cat "$out/.job-$phase.status" 2>/dev/null || echo 'no status file')"
    return 0
}

# finalize PHASE [resume]: after the job: fetch, then pause.  Returns 2 while the job still runs, the
# instance is changing state, or the instance list cannot be read (LOOKUP_DOWN=1): "unknown" is
# never taken for "nothing to pause".
finalize() {
    local phase="$1" resume="${2:-}" dir name id status rid state t
    dir="$(phase_dir "$phase")"
    name="$(cat "$dir/launch.name" 2>/dev/null)" || die "'$phase' was not launched from this checkout"
    read -r id status < <(find_instance "$name") || true
    if [ "${id:-}" = "?" ]; then
        # `jl list` failed: the link or the API is down.  That is NOT "no instance": keep trying.
        LOOKUP_DOWN=1
        info "cannot read the instance list (link or API down?): $name may still be billing; will keep trying"
        return 2
    fi
    LOOKUP_DOWN=0
    [ -n "${id:-}" ] || { info "no instance named $name in a good \`jl list\`: nothing to pause"; return 0; }
    echo "$id" > "$dir/launch.id"
    if [ "$status" = Paused ]; then
        [ "$resume" = resume ] || { info "instance $id is paused; \`fetch $phase\` resumes it, fetches and pauses it again"; return 0; }
        # ONE GPU is enough to read /home, and it cannot be stranded when the region has no 2 free
        info "resuming the paused instance $id with $JL_FETCH_GPUS GPU to fetch the outputs (billed while it runs)"
        # a resume that times out may still have worked on the server: remember that a guard
        # watchdog is owed (below, before anything else), and keep watching instead of exiting
        : > "$dir/resume.pending"
        timeout 900 jl resume "$id" --num-gpus "$JL_FETCH_GPUS" --yes || { LOOKUP_DOWN=1; info "jl resume failed or timed out: it may have worked; will check and keep trying"; return 2; }
        read -r id status < <(find_instance "$name") || true
        if [ "${id:-}" = "?" ]; then LOOKUP_DOWN=1; info "resumed, but cannot read the instance list: will keep trying"; return 2; fi
        [ "${status:-}" = Running ] || { LOOKUP_DOWN=1; info "instance $name is ${status:-missing} after resume: will keep watching"; return 2; }
    elif [ "$status" = Running ]; then
        :
    else
        case "$status" in
            Destroyed|Destroying|Deleted|Terminated) info "instance $id is $status"; return 0 ;;
            *) info "instance $id is $status (not Paused, not Running): will keep watching"; return 2 ;;
        esac
    fi
    echo "$id" > "$dir/launch.id"
    if [ -f "$dir/resume.pending" ]; then
        # a resumed container has no processes: the guard watchdog (with the new machine id in
        # its secret) comes before anything else
        if start_guard "$id" "" "$id"; then rm -f "$dir/resume.pending"
        else info "WARNING: no watchdog on the resumed instance $id: fetch fast, the pause below is the only one"; fi
    fi
    state=unknown
    for t in 1 2 3; do
        rid="$(latest_run_id "$id")"
        state="$(run_state "$id" "$rid")"
        [ "$state" != unknown ] && break
        sleep "$NAP"
    done
    case "$state" in
        running|unknown)
            cat >&2 <<EOF
================================================================================
 JOB '$phase' IS STILL RUNNING or its state is unknown (state: $state) on instance $id: IT KEEPS BILLING.
 The job ends by itself at its deadline, and the instance's own watchdog pauses it after the
 job + the fetch grace (and at the deadline + 20 min), even if this laptop is gone.
 Re-attach:   bash scripts/jl_t12.sh watch $phase     (done automatically by \`launch\`)
 or look:   bash scripts/jl_t12.sh status $phase       pause by hand:   bash scripts/jl_t12.sh pause $phase
================================================================================
EOF
            return 2 ;;
    esac
    # `jl run status` can report a finished state when it cannot reach the machine (seen on 551207:
    # "Connection failed", then a mid-run pause).  Only the job's own end marker says it is over.
    if ! { timeout 120 jl exec "$id" -- sh -c "test -s /home/ANLP-Project/runs/t12/.job-$phase.status && echo T12_JOB_ENDED" 2>/dev/null || true; } | grep -q T12_JOB_ENDED; then
        info "jl reports the run as '$state', but instance $id has no runs/t12/.job-$phase.status (or cannot be read): treating it as still running"
        return 2
    fi
    fetch_outputs "$phase" "$id" "$rid" || info "FETCH FAILED: the outputs stay on the instance (/home/ANLP-Project/runs/t12-out); pausing anyway, \`fetch $phase\` retries"
    pause_instance "$id" "$name" || { LOOKUP_DOWN=1; return 2; }
    rm -f "$dir/resume.pending"
}

# ---- launch --------------------------------------------------------------------------------

# budget_on_instance ID PHASE: seconds left on the instance's own budget file (empty: no file).
budget_on_instance() {
    timeout 120 jl exec "$1" -- sh -c "f=/home/ANLP-Project/runs/t12/.deadline-$2; [ -s \$f ] && echo \$((\$(cat \$f) - \$(date +%s)))" 2>/dev/null | tr -d '\r' | grep -E '^-?[0-9]+$' | tail -n 1 || true
}

cmd_launch() {
    local phase="${1:-}" dir full short name hours row price free t ans id status rid state ngpus T0 inst rest left baked
    check_phase "$phase"
    dir="$(phase_dir "$phase")"
    full="$(cat "$dir/COMMIT")"
    short="${full:0:7}"
    hours="$(cat "$dir/HOURS")"
    [ -f "$dir/t12_watchdog.sh" ] || die "$dir has no t12_watchdog.sh: run prepare again"
    ngpus="${JL_NUM_GPUS:-$(default_gpus "$phase")}"
    JL_NUM_GPUS="$ngpus"
    need_key
    need_jl
    baked="$(sed -n 's/^export NGPU=//p' "$dir/payload/job.env" 2>/dev/null | tail -n 1 | tr -d '\r')"
    if [ -n "$baked" ] && [ "$baked" != "$ngpus" ]; then
        die "the upload has NGPU=$baked baked in but this launch has JL_NUM_GPUS=$ngpus: set JL_NUM_GPUS=$baked or prepare again"
    fi
    if [ -f "$dir/RESUME_NGPU" ] && [ "$(tr -d '\r\n' < "$dir/RESUME_NGPU")" != "$ngpus" ]; then
        die "RESUME_FROM was packed for $(cat "$dir/RESUME_NGPU") shards but JL_NUM_GPUS=$ngpus: they must be equal (checked before anything is created)"
    fi
    [ ! -f "$dir/launch.name" ] || info "note: $phase was launched before from this upload${ON:+ (on instance $ON)}"
    if is_report "$phase"; then
        origin_has_commit "$full" \
            || die "commit $short is not an ancestor of any branch that is on origin now (git ls-remote): git push it first.  The grids must be pushed before the first report-half generation"
        info "commit $short is on origin (branch $ORIGIN_BRANCH at ${ORIGIN_SHA:0:7})"
        for t in gsm8k mbpp; do
            git show "$full:prereg/t12_grid_$t.json" 2>/dev/null | "$PYJ" -c 'import json, sys; assert json.load(sys.stdin)["configs"]' \
                || die "prereg/t12_grid_$t.json is missing or has no configs at $short"
        done
        info "commit $short has both per-task grids"
    fi
    if [ "$(timeout 120 jl ssh-key list --json 2>/dev/null | "$PYJ" -c 'import json, sys; d = json.load(sys.stdin); print(len(d.get("ssh_keys", d) if isinstance(d, dict) else d))' 2>/dev/null | tr -d '\r' || echo ?)" = 0 ]; then
        die "no SSH key on the Jarvis account: jl ssh-key add ~/.ssh/id_ed25519.pub --name laptop (jl run needs one)"
    fi
    row="$(gpu_row "$JL_GPU" "$JL_REGION")"
    price=""
    case "$row" in
        "?") info "warning: could not read \`jl gpus --json\`; jl will say if the GPU is unavailable" ;;
        noregion*) info "warning: $JL_GPU is listed only in ${row#noregion }, not in $JL_REGION; jl will say if it cannot be created" ;;
        absent*) die "GPU type $JL_GPU is not offered in $JL_REGION (offered: ${row#absent }); see \`jl gpus\`, set JL_GPU/JL_REGION" ;;
        *) free="${row%% *}"
           price="${row#* }"
           info "$JL_GPU container in $JL_REGION: $free free, price $price per GPU per hour = $(awk -v p="$price" -v n="$ngpus" 'BEGIN { printf "%.2f", p * n }') per hour for $ngpus GPU"
           if [ -z "${ON:-}" ]; then
               [ "$free" -ge "$ngpus" ] 2>/dev/null || die "only $free $JL_GPU free in $JL_REGION, need $ngpus"
           fi ;;
    esac
    wallet_check "$hours" "$ngpus" "$price"
    if [ -n "${ON:-}" ]; then
        inst="$(instance_by_id "$ON")"
        [ -n "$inst" ] || die "no instance with machine id $ON on this account (jl list)"
        read -r name status rest <<< "$inst"
        [ "$name" != "?" ] || die "cannot read the instance list (jl list failed): try again"
        case "$status" in Running|Paused) ;; *) die "instance $ON is $status: only a Running or Paused instance can be used" ;; esac
    else
        name="t12-$phase-$short-$(date +%m%d%H%M)"
        status=new
    fi
    cat <<EOF
================================================================================
 LAUNCH '$phase' (commit $short) -- THIS BILLS: $ngpus x $JL_GPU in $JL_REGION, up to $hours h (provisioning included)
 instance name: $name   $([ -n "${ON:-}" ] && echo "existing instance $ON ($status; resumed with $ngpus GPU if paused)")
 It is paused by the instance's own watchdog ${T12_FETCH_GRACE_MIN:-20} min after the job (or at the budget + 20 min)
 even if this terminal dies.  The dedicated JL_API_KEY goes to the instance's /home/.t12.env:
 revoke it in the dashboard after the sweep.
 IF THIS LAPTOP CRASHES OR LOSES THE NETWORK in the next minutes (before "watchdog armed"), run
 \`jl list\` and pause anything named t12-* by hand: \`jl pause <id> --yes\`.
================================================================================
EOF
    if [ "${YES:-0}" != 1 ]; then
        read -r -p "Type yes to ${ON:+resume/}create the instance and start: " ans
        [ "$ans" = yes ] || die "not confirmed"
    fi
    echo "$name" > "$dir/launch.name"
    rm -f "$dir/resume.pending"          # (a leftover would make `watch` replace the job's watchdog)
    T0="$(date +%s)"
    echo "$T0" > "$dir/launch.epoch"
    LAUNCH_PHASE="$phase"
    JOB_RC=0
    trap 'launch_exit' EXIT
    trap 'echo "[jl_t12] interrupted"; stop_holder; exit 130' INT TERM HUP

    # 1. the instance, 2. the watchdog (secret + script + guard) AT ONCE, 3. the checks, 4. the job
    if [ -n "${ON:-}" ]; then
        id="$ON"
        if [ "$status" = Paused ]; then
            info "resuming the paused instance $id with $ngpus GPU (its /home keeps the venv, the model and the partial results)"
            timeout 900 jl resume "$id" --num-gpus "$ngpus" --yes 2>&1 | tee -a "$dir/launch.log" || true
            read -r id status < <(find_instance "$name") || true
            [ "${id:-}" != "?" ] || die "instance $name was resumed (or not) but the instance list cannot be read: see \`jl list\`; \`watch\` keeps trying"
            [ "${status:-}" = Running ] || die "instance $name is ${status:-missing} after jl resume"
            info "instance $name is Running as machine $id"
        else
            rid="$(latest_run_id "$id")"
            state="$(run_state "$id" "$rid")"
            [ "$state" != running ] && [ "$state" != unknown ] || die "a run (${rid:-?}) is running on instance $id, or its state cannot be read: one job per instance"
        fi
    else
        info "creating the instance"
        local out
        out="$(timeout 900 jl create --gpu "$JL_GPU" --num-gpus "$ngpus" --region "$JL_REGION" --storage "$JL_STORAGE" \
                 --template "$JL_TEMPLATE" --name "$name" --json --yes 2>>"$dir/launch.log")" || true
        id="$(printf '%s' "$out" | "$PYJ" -c 'import json, sys; print(json.load(sys.stdin)["machine_id"])' 2>/dev/null | tr -d '\r' || true)"
        if [ -z "$id" ]; then
            read -r id status < <(find_instance "$name") || true
            [ "${id:-}" != "?" ] || die "jl create gave no id and the instance list cannot be read: $name MAY EXIST and bill: see \`jl list\` (\`watch\` keeps trying)"
            [ -n "${id:-}" ] || die "jl create failed and no instance named $name exists (see $dir/launch.log)"
            info "jl create reported no id but instance $id ($name) exists: $status"
            [ "${status:-}" = Running ] || die "instance $id ($name) is ${status:-?}: check \`jl list\`; it will be paused when Running"
        fi
        info "instance $id ($name) is Running"
    fi
    echo "$id" > "$dir/launch.id"
    arm_instance "$dir" "$id" "$name" "$T0" "$hours"
    if [ -n "${ON:-}" ]; then
        inst="$(instance_by_id "$id")"
        read -r name status rest <<< "$inst"
        case "${rest:-}" in
            ''|None|null) ;;
            *) [ "$rest" = "$ngpus" ] || die "instance $id has $rest GPU, this launch wants $ngpus (JL_NUM_GPUS); it will be paused again" ;;
        esac
        left="$(budget_on_instance "$id" "$phase")"
        if [ -n "$left" ]; then
            info "the instance's budget file for '$phase': $((left / 60)) min left"
            if [ "$left" -le 0 ] && ! grep -q '^export NEW_BUDGET=1$' "$dir/payload/job.env"; then
                die "the budget on instance $id is spent: \`NEW_BUDGET=1 prepare $phase $short\` and launch again (the instance will be paused)"
            fi
        fi
    fi
    left="$(awk -v t0="$T0" -v now="$(date +%s)" -v h="$hours" 'BEGIN { printf "%d", h * 3600 - (now - t0) }')"
    info "starting the job; Ctrl+C or a dropped link only detaches this terminal"
    (cd "$dir" && timeout -k 30 "$((left + 3600))" jl run "$(cat NAME)" --on "$id" --yes) 2>&1 | tee -a "$dir/launch.log" || JOB_RC="${PIPESTATUS[0]}"
    info "jl run returned $JOB_RC"
    return 0
}

# The safety net of `launch`: runs on every way out of it.  A job that is still running (link
# dropped, Ctrl+C) is not abandoned: `watch` takes over (the instance's watchdog covers the case
# that this laptop dies too).
launch_exit() {
    local rc=$?
    trap - EXIT INT TERM
    cleanup_secret
    [ "${JOB_RC:-0}" = 0 ] || rc="$JOB_RC"
    info "launch is over: now fetching and pausing (\`watch\`); Ctrl+C again to leave, the instance's watchdog pauses it"
    # `watch` runs in a subshell: a `die` inside it (or Ctrl+C) ends only the subshell, and the
    # banner below says so.  (An EXIT trap set from inside an EXIT trap does not run.)
    trap ':' INT
    if (cmd_watch "$LAUNCH_PHASE"); then :; else rc=$?; billing_banner "$LAUNCH_PHASE"; fi
    exit "$rc"
}

# ---- status, watch, fetch, pause -----------------------------------------------------------

# billing_banner PHASE: said whenever watch/launch leaves with an instance that may not be paused.
billing_banner() {
    local dir name id
    dir="$(cat "$UP/.last-$1" 2>/dev/null || true)"
    name="$(cat "$dir/launch.name" 2>/dev/null || echo "t12-$1-*")"
    id="$(cat "$dir/launch.id" 2>/dev/null || true)"
    cat >&2 <<EOF
================================================================================
 INSTANCE $name (machine id ${id:-unknown}) MAY STILL BE BILLING.
 This script could not confirm that it is paused.  Do this NOW:
     jl list                                # find the t12-* instance and its status
     jl pause ${id:-<id>} --yes             # (the id changes on a resume: take it from jl list)
 or use the dashboard, https://jarvislabs.ai/.  The instance's own watchdog
 (/home/t12-watchdog.log) is also trying, if it was armed.
================================================================================
EOF
}

# run_watch PHASE: `watch` as a command: a banner on any non-zero way out.
run_watch() {
    local p="${1:-}" rc=0
    check_phase "$p"
    trap ':' INT
    (cmd_watch "$p") || rc=$?
    [ "$rc" = 0 ] || billing_banner "$p"
    return "$rc"
}

instance_of() {
    local dir name
    dir="$(phase_dir "$1")"
    name="$(cat "$dir/launch.name" 2>/dev/null)" || die "'$1' was not launched from this checkout"
    echo "$name"
}

cmd_status() {
    local phase="${1:-}" name id status rid
    check_phase "$phase"; need_jl
    name="$(instance_of "$phase")"
    read -r id status < <(find_instance "$name") || true
    [ "${id:-}" != "?" ] || die "cannot read the instance list (jl list failed)"
    [ -n "${id:-}" ] || die "no instance named $name"
    rid="$(latest_run_id "$id")"
    info "instance $id ($name): $status; run ${rid:-none}: $(run_state "$id" "$rid")"
    [ "$status" = Running ] || return 0
    { [ -z "$rid" ] || [ "$rid" = "?" ]; } || timeout 120 jl run logs "$rid" --tail 15 || true
    timeout 120 jl exec "$id" -- sh -lc 'tail -n 5 /home/t12-watchdog.log' || true
    # shellcheck disable=SC2016  # the loop runs on the instance
    timeout 180 jl exec "$id" -- sh -lc 'for d in /home/ANLP-Project /home/*/ANLP-Project; do [ -d "$d" ] && cd "$d" && exec bash scripts/jarvis_t12.sh status; done' || true
}

cmd_watch() {
    local phase="${1:-}" dir hours start now rc giveup t_watch
    check_phase "$phase"; need_tools
    dir="$(phase_dir "$phase")"
    hours="$(cat "$dir/HOURS")"
    start="$(cat "$dir/launch.epoch" 2>/dev/null || date +%s)"
    t_watch="$(date +%s)"
    # never give up before the budget + 1 h: an outage is what `watch` is for
    giveup="$(awk -v s="$start" -v h="$hours" 'BEGIN { printf "%d", s + h * 3600 + 3600 }')"
    [ -z "${T12_WATCH_GIVEUP_S:-}" ] || giveup=$((t_watch + T12_WATCH_GIVEUP_S))
    LOOKUP_DOWN=0
    while :; do
        rc=0
        finalize "$phase" resume || rc=$?
        [ "$rc" = 2 ] || return "$rc"
        now="$(date +%s)"
        if [ "$now" -ge "$giveup" ]; then
            info "giving up watching '$phase' after the budget + 1 h: the instance state could not be settled"
            return 1
        fi
        if [ "$LOOKUP_DOWN" = 0 ] && [ "$now" -ge "$(awk -v s="$start" -v h="$hours" 'BEGIN { printf "%d", s + h * 3600 + 2400 }')" ]; then
            info "40 min past the job's budget and it still reports running: force-pausing"
            if (FORCE=1 cmd_pause "$phase"); then return 0; fi
            info "the force-pause did not work: will try again"
        fi
        if [ "$LOOKUP_DOWN" = 1 ]; then sleep "${T12_RETRY_S:-60}"; else sleep "$WATCH_SLEEP"; fi
    done
}

cmd_fetch() {
    local phase="${1:-}"
    check_phase "$phase"; need_jl
    finalize "$phase" resume
}

cmd_pause() {
    local phase="${1:-}" name id status rid state
    check_phase "$phase"; need_jl
    name="$(instance_of "$phase")"
    read -r id status < <(find_instance "$name") || true
    [ "${id:-}" != "?" ] || die "cannot read the instance list (jl list failed): try again, or pause by hand (jl list; jl pause <id> --yes)"
    [ -n "${id:-}" ] || die "no instance named $name"
    case "$status" in
        Running) ;;
        Paused|Destroyed|Destroying|Deleted|Terminated) info "instance $id is $status"; return 0 ;;
        *) info "instance $id is $status: not paused yet"; return 2 ;;
    esac
    rid="$(latest_run_id "$id")"
    state="$(run_state "$id" "$rid")"
    if [ "$state" = running ] && [ "${FORCE:-0}" != 1 ]; then
        die "the job is still running on $id; FORCE=1 pauses it anyway (outputs written so far stay on /home)"
    fi
    pause_instance "$id" "$name"
}

main() {
    case "${1:-}" in
        prepare) shift; cmd_prepare "$@" ;;
        launch) shift; cmd_launch "$@" ;;
        status) shift; cmd_status "$@" ;;
        watch) shift; run_watch "$@" ;;
        fetch) shift; cmd_fetch "$@" ;;
        pause) shift; cmd_pause "$@" ;;
        *) echo "usage: jl_t12.sh prepare <phase> <commit> | launch|status|watch|fetch|pause <phase>   (phase: calib|smoke|report|report-gsm8k|report-mbpp)"; exit 2 ;;
    esac
}

# (a test sources this file to call the functions above)
if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
