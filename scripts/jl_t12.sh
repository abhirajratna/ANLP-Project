#!/bin/bash
# T12 on Jarvis Labs through the official `jl` CLI, driven from the laptop (Git Bash on Windows
# or any Unix shell).  Two paid phases with a laptop step in between: the final grids, the
# matching rule and the calibrators must be on origin before the first report-half generation.
#
#   uv tool install jarvislabs          # once (done on this laptop: jl 0.2.x)
#   jl setup                            # once: asks for your API token (jarvislabs.ai > Settings)
#   jl ssh-key add ~/.ssh/id_ed25519.pub --name laptop   # once; jl run needs an SSH key on the account
#
#   bash scripts/jl_t12.sh prepare smoke <commit> && JL_GPU=L4 JL_NUM_GPUS=1 JL_REGION=IN2 \
#       bash scripts/jl_t12.sh launch smoke      # optional, ~10 min on a cheap GPU: proves the chain
#
#   bash scripts/jl_t12.sh prepare calib <commit>   # offline: bundle + inputs + job into one file
#   bash scripts/jl_t12.sh launch calib             # creates the 2-GPU instance, runs, fetches, pauses
#   # -> runs/t12/jl-calib-<sha7>/cal-grids/t12_grid_{gsm8k,mbpp}.json; review them, then
#   cp runs/t12/jl-calib-<sha7>/cal-grids/t12_grid_*.json prereg/ && git add prereg && git commit && git push
#
#   bash scripts/jl_t12.sh prepare report <pushed-commit>
#   bash scripts/jl_t12.sh launch report            # refuses unless the commit is on origin with both grids
#   # -> runs/t12/jl-report-<sha7>/sweep-{gsm8k,mbpp}/  (then scripts/score_t12.py)
#
#   bash scripts/jl_t12.sh status|watch|fetch|pause <phase>    # see below
#
# How it works.  `jl run <file>.sh --gpu RTX-PRO6000 --num-gpus 2 --region IN1 --keep` creates a
# fresh instance, scp's that ONE file to /home and runs it detached with bash, streaming the log.
# A directory target would be synced with rsync, which Git Bash does not have, so `prepare` packs
# everything into one self-extracting file: a bootstrap, then (base64 after a marker) a tar of
# a `git bundle` of the commit (LF files; a Windows checkout's CRLF would change the frozen
# manifests' sha256), the Ada-only inputs from runs/jarvis-stage/ (sha256-checked here against
# scripts/jarvis_t12.inputs.sha256 at that commit) and job.env (the knobs below).  The bootstrap
# clones the bundle to /home/ANLP-Project and execs `scripts/jarvis_t12.sh job <phase>`, which
# has its own global deadline and ALWAYS ends.
#
# Pausing.  jl's own `--pause` acts only when the jl process on this laptop is still attached
# when the run ends (Ctrl+C, a closed terminal or a dropped link detaches it and the instance
# keeps billing), and it pauses BEFORE the outputs are fetched, so `jl download` (running
# instances only) would then need a billed `jl resume`.  So `launch` uses `--keep` and does the
# same itself, in this order: wait for the job -> fetch the outputs -> pause, always (even if
# the fetch failed: /home survives a pause; the fetch status is printed).  JL_LIFECYCLE=pause
# uses jl's `--pause` instead (`fetch` then resumes, fetches and pauses again).  If the launch is
# interrupted: `watch <phase>` re-attaches (polls every 2 min, pauses when the job has ended,
# and force-pauses 45 min after the job's budget); `pause <phase>` pauses by hand (refuses while
# the job runs, FORCE=1 overrides); `status <phase>` shows the instance, the run and the log.
# Nothing here ever destroys an instance.  Every jl call has a timeout.
#
# On Windows with core.autocrlf the checkout of this file is CRLF and bash chokes on it: once,
# `sed -i 's/\r$//' scripts/jl_t12.sh scripts/jarvis_t12.sh` (the upload comes from git blobs, LF).
#
# Knobs, baked into the upload by `prepare`: BUDGET_HOURS (calib 6, report 11, smoke 1),
# NEW_BUDGET, CAL_GRID, CAL_MAX_HOURS, CHECK_LIMIT, CHECK_MAX_HOURS, CHECK_MAX_SPF, MAX_RESTARTS,
# RUN_ARGS, PRECISION, NGPU, ALLOW_BF16 (see scripts/jarvis_t12.sh).  At `launch`: JL_GPU
# (RTX-PRO6000), JL_NUM_GPUS (2), JL_REGION (IN1), JL_STORAGE (100), JL_TEMPLATE (pytorch),
# JL_LIFECYCLE (keep|pause), ON=<machine id> (run on an existing running instance instead of
# creating one, e.g. to resume a sweep: resume it with `jl resume <id>`, `prepare` with
# NEW_BUDGET=1, `ON=<id> launch report`), YES=1 (skip the confirmation), STAGE (the directory
# with artifacts/ and runs/ from Ada; default runs/jarvis-stage of this checkout).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
UP=runs/jl-upload
JL_GPU="${JL_GPU:-RTX-PRO6000}"
JL_NUM_GPUS="${JL_NUM_GPUS:-2}"
JL_REGION="${JL_REGION:-IN1}"
JL_STORAGE="${JL_STORAGE:-100}"
JL_TEMPLATE="${JL_TEMPLATE:-pytorch}"
JL_LIFECYCLE="${JL_LIFECYCLE:-keep}"
export PATH="$HOME/.local/bin:$PATH"
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
default_hours() { case "$1" in calib) echo 6 ;; smoke) echo 1 ;; *) echo 11 ;; esac; }

# ---- prepare: the upload -------------------------------------------------------------------

# The staging directory of the Ada-only inputs (a worktree has no runs/ of its own).
find_stage() {
    local c
    for c in "${STAGE:-}" "$ROOT/runs/jarvis-stage" "$(git rev-parse --git-common-dir)/../runs/jarvis-stage"; do
        [ -n "$c" ] && [ -d "$c/artifacts" ] && { (cd "$c" && pwd); return 0; }
    done
    return 1
}

# The self-extracting bootstrap (placeholders @PHASE@ and @COMMIT@).  T12_BASE and T12_DRYRUN
# exist for `prepare`'s own rehearsal on this machine.
write_bootstrap() {
    cat > "$1" <<'EOF'
#!/bin/bash
# T12 upload for `jl run`, generated by scripts/jl_t12.sh prepare (phase @PHASE@, commit @COMMIT@).
# jl copies this single file to the instance and runs it with bash.  After the marker line
# below comes a base64 tar: repo.bundle (git bundle of the commit), inputs/ (Ada-only files),
# job.env (the knobs).  The checkout is made from the bundle, so its files are LF.
set -euo pipefail
PHASE=@PHASE@
COMMIT=@COMMIT@
if [ -n "${T12_BASE:-}" ]; then BASE="$T12_BASE"; elif [ "$(id -u)" = 0 ]; then BASE=/home; else BASE="$HOME"; fi
REPO="$BASE/ANLP-Project"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
IN="$BASE/c3-t12-upload-$PHASE"
rm -rf "$IN"
mkdir -p "$IN"
start="$(awk '/^__T12_PAYLOAD__$/ { print NR + 1; exit }' "$SELF")"
tail -n +"$start" "$SELF" | base64 -d | tar -xzf - -C "$IN"
[ -d "$REPO/.git" ] || git init -q "$REPO"
git -C "$REPO" config core.autocrlf false
git -C "$REPO" fetch -q "$IN/repo.bundle" "+refs/jl-t12/commit:refs/jl-t12/commit"
[ "$(git -C "$REPO" rev-parse refs/jl-t12/commit)" = "$COMMIT" ] || { echo "bundle does not hold $COMMIT" >&2; exit 1; }
git -C "$REPO" checkout -q -f --detach "$COMMIT"
[ ! -d "$IN/inputs" ] || cp -a "$IN/inputs/." "$REPO/"
set -a
# shellcheck disable=SC1091
. "$IN/job.env"
set +a
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

cmd_prepare() {
    local phase="${1:-}" commit="${2:-}" full short dir stage want path got src name k t
    check_phase "$phase"
    [ -n "$commit" ] || die "usage: prepare <phase> <commit>"
    full="$(git rev-parse --verify "$commit^{commit}")" || die "no such commit: $commit"
    short="${full:0:7}"
    for path in scripts/jarvis_t12.sh scripts/jarvis_t12.inputs.sha256 scripts/run_t12.py; do
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

    : > "$dir/payload/job.env"
    for k in "${KNOBS[@]}"; do
        [ -z "${!k:-}" ] || printf 'export %s=%q\n' "$k" "${!k}" >> "$dir/payload/job.env"
    done
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
        cat "$dir/payload/job.env"
    } > "$dir/MANIFEST.txt"
    echo "$dir" > "$UP/.last-$phase"

    # Rehearsal: unpack the upload exactly as the instance will, check the checkout is LF and
    # the GSM8K inputs verify; nothing leaves this machine.
    local tmp
    tmp="$(mktemp -d)"
    if (cd "$dir" && T12_BASE="$tmp" T12_DRYRUN=1 bash "$name") > "$dir/rehearsal.log" 2>&1; then
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

need_jl() {
    command -v jl >/dev/null 2>&1 || die "jl is not installed: uv tool install jarvislabs (then put ~/.local/bin on PATH)"
    command -v ssh >/dev/null 2>&1 || die "ssh must be on PATH (jl uses it)"
    command -v scp >/dev/null 2>&1 || die "scp must be on PATH (jl uses it)"
    timeout 120 jl status --json >/dev/null 2>&1 || die "jl is not logged in (run: jl setup) or the API is unreachable"
}

# Phase directory of the last `prepare`.
phase_dir() {
    [ -f "$UP/.last-$1" ] || die "no upload for '$1': run prepare first"
    cat "$UP/.last-$1"
}

# "machine_id status" of the newest instance with this name, or nothing.
find_instance() {
    { timeout 120 jl list --json 2>/dev/null || true; } | "$PYJ" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
for i in (d.get("instances", []) if isinstance(d, dict) else d):
    if i.get("name") == sys.argv[1]:
        print(i["machine_id"], i["status"])
' "$1" | tail -n 1
}

# The latest local run record of an instance.
latest_run_id() {
    { timeout 120 jl run list --json --machine "$1" --limit 1 2>/dev/null || true; } | "$PYJ" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
if d:
    print(d[0]["run_id"])
'
}

# running | succeeded | failed | unknown | none (no run was ever started on the instance)
run_state() {
    local id="$1" rid="$2"
    [ -n "$rid" ] || { echo none; return 0; }
    { timeout 180 jl run status "$rid" --json 2>/dev/null || true; } | "$PYJ" -c '
import json, sys
try:
    print(json.load(sys.stdin).get("state", "unknown"))
except ValueError:
    print("unknown")
'
}

pause_instance() {
    local id="$1" name="$2" t ok=0 now
    for t in 1 2 3; do
        if timeout 180 jl pause "$id" --yes; then ok=1; break; fi
        info "jl pause failed (try $t/3)"
        sleep 20
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
================================================================================
EOF
    return 1
}

# fetch_outputs PHASE ID RID: runs/t12-out/<phase>.tgz and the job log, into runs/t12/jl-<phase>-<sha7>/
fetch_outputs() {
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
    info "FETCHED into $out ($(du -h "$out/$phase.tgz" | cut -f1), sha256 ${got:0:12}); $(cat "$out/.job-$phase.status" 2>/dev/null || echo 'no status file')"
    return 0
}

# finalize PHASE [resume]: after the job: fetch, then pause.  Returns 2 while the job still runs.
finalize() {
    local phase="$1" resume="${2:-}" dir name id status rid state t
    dir="$(phase_dir "$phase")"
    name="$(cat "$dir/launch.name" 2>/dev/null)" || die "'$phase' was not launched from this checkout"
    read -r id status < <(find_instance "$name") || true
    [ -n "${id:-}" ] || { info "no instance named $name: nothing to pause"; return 0; }
    if [ "$status" = Paused ]; then
        [ "$resume" = resume ] || { info "instance $id is paused; \`fetch $phase\` resumes it, fetches and pauses it again"; return 0; }
        info "resuming the paused instance $id to fetch the outputs (billed while it runs)"
        timeout 900 jl resume "$id" --yes || die "jl resume failed"
        read -r id status < <(find_instance "$name") || true
        [ "${status:-}" = Running ] || die "instance $name is ${status:-missing} after resume"
    elif [ "$status" != Running ]; then
        info "instance $id is $status"; return 0
    fi
    rid="$(latest_run_id "$id")"
    for t in 1 2 3; do
        state="$(run_state "$id" "$rid")"
        [ "$state" != unknown ] && break
        sleep 20
    done
    case "$state" in
        running|unknown)
            cat >&2 <<EOF
================================================================================
 JOB '$phase' IS STILL RUNNING (state: $state) on instance $id: IT KEEPS BILLING.
 The job ends by itself at its deadline.  Re-attach:   bash scripts/jl_t12.sh watch $phase
 or look:   bash scripts/jl_t12.sh status $phase       pause by hand:   bash scripts/jl_t12.sh pause $phase
================================================================================
EOF
            return 2 ;;
    esac
    fetch_outputs "$phase" "$id" "$rid" || info "FETCH FAILED: the outputs stay on the instance (/home/ANLP-Project/runs/t12-out); pausing anyway, \`fetch $phase\` retries"
    pause_instance "$id" "$name"
}

# ---- launch --------------------------------------------------------------------------------

cmd_launch() {
    local phase="${1:-}" dir full short name hours row t ans
    check_phase "$phase"
    dir="$(phase_dir "$phase")"
    full="$(cat "$dir/COMMIT")"
    short="${full:0:7}"
    hours="$(cat "$dir/HOURS")"
    need_jl
    [ ! -f "$dir/launch.name" ] || info "note: $phase was launched before from this upload; a new instance is created"
    if is_report "$phase"; then
        git branch -r --contains "$full" 2>/dev/null | grep -Eq '^[[:space:]]*origin/' \
            || die "commit $short is not on any origin branch (git push it first; \`git branch -r --contains $short\` is empty): the grids must be pushed before the first report-half generation"
        for t in gsm8k mbpp; do
            git show "$full:prereg/t12_grid_$t.json" 2>/dev/null | "$PYJ" -c 'import json, sys; assert json.load(sys.stdin)["configs"]' \
                || die "prereg/t12_grid_$t.json is missing or has no configs at $short"
        done
        info "commit $short is on origin with both per-task grids"
    fi
    if [ "$(timeout 120 jl ssh-key list --json 2>/dev/null | "$PYJ" -c 'import json, sys; d = json.load(sys.stdin); print(len(d.get("ssh_keys", d) if isinstance(d, dict) else d))' 2>/dev/null || echo ?)" = 0 ]; then
        die "no SSH key on the Jarvis account: jl ssh-key add ~/.ssh/id_ed25519.pub --name laptop (jl run needs one)"
    fi
    if [ -z "${ON:-}" ]; then
        row="$({ timeout 120 jl gpus --json 2>/dev/null || true; } | "$PYJ" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    print("?"); sys.exit(0)
rows = d.get("gpus", d) if isinstance(d, dict) else d
want, reg = sys.argv[1].lower(), sys.argv[2].upper()
for r in rows:
    if str(r.get("gpu_type", "")).lower() == want and str(r.get("region", "")).upper() == reg:
        print(r.get("effective_num_free_devices", r.get("num_free_devices")), r.get("price_per_hour"))
        break
else:
    print("absent", ",".join(sorted({str(r.get("gpu_type")) for r in rows})))
' "$JL_GPU" "$JL_REGION")"
        case "$row" in
            "?") info "warning: could not read \`jl gpus --json\`; jl run will say if the GPU is unavailable" ;;
            absent*) die "GPU type $JL_GPU is not offered in $JL_REGION (offered: ${row#absent }); see \`jl gpus\`, set JL_GPU/JL_REGION" ;;
            *) info "$JL_GPU in $JL_REGION: ${row%% *} free, price ${row#* } per hour (per GPU?)"
               [ "${row%% *}" -ge "$JL_NUM_GPUS" ] 2>/dev/null || die "only ${row%% *} $JL_GPU free in $JL_REGION, need $JL_NUM_GPUS" ;;
        esac
    fi
    if [ -n "${ON:-}" ]; then
        name="$({ timeout 120 jl list --json 2>/dev/null || true; } | "$PYJ" -c '
import json, sys
d = json.load(sys.stdin)
for i in (d.get("instances", []) if isinstance(d, dict) else d):
    if str(i.get("machine_id")) == sys.argv[1] and i.get("status") == "Running":
        print(i["name"])
' "$ON")"
        [ -n "$name" ] || die "instance $ON is not a Running instance of this account (jl resume $ON first)"
    else
        name="t12-$phase-$short-$(date +%m%d%H%M)"
    fi
    cat <<EOF
================================================================================
 LAUNCH '$phase' (commit $short) -- THIS BILLS: $JL_NUM_GPUS x $JL_GPU in $JL_REGION, up to $hours h (the job's own deadline)
 instance name: $name   lifecycle: $JL_LIFECYCLE   $([ -n "${ON:-}" ] && echo "on existing instance $ON")
 Keep this terminal open until it prints PAUSED (or run \`watch $phase\` afterwards).
================================================================================
EOF
    if [ "${YES:-0}" != 1 ]; then
        read -r -p "Type yes to create the instance and start: " ans
        [ "$ans" = yes ] || die "not confirmed"
    fi
    echo "$name" > "$dir/launch.name"
    date +%s > "$dir/launch.epoch"
    local args=("$(cat "$dir/NAME")")
    if [ -n "${ON:-}" ]; then
        args+=(--on "$ON")
    else
        args+=(--gpu "$JL_GPU" --num-gpus "$JL_NUM_GPUS" --region "$JL_REGION" --storage "$JL_STORAGE"
               --template "$JL_TEMPLATE" --name "$name")
        if [ "$JL_LIFECYCLE" = pause ]; then args+=(--pause); else args+=(--keep); fi
    fi
    LAUNCH_PHASE="$phase"
    trap 'launch_exit' EXIT
    trap 'echo "[jl_t12] interrupted"; exit 130' INT TERM
    local rc=0
    (cd "$dir" && jl run "${args[@]}" --yes) 2>&1 | tee -a "$dir/launch.log" || rc="${PIPESTATUS[0]}"
    info "jl run returned $rc"
    return 0
}

# The safety net of `launch`: runs on every way out of it.
launch_exit() {
    local rc=$?
    trap - EXIT INT TERM
    finalize "$LAUNCH_PHASE" || rc=$?
    exit "$rc"
}

# ---- status, watch, fetch, pause -----------------------------------------------------------

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
    [ -n "${id:-}" ] || die "no instance named $name"
    rid="$(latest_run_id "$id")"
    info "instance $id ($name): $status; run ${rid:-none}: $(run_state "$id" "$rid")"
    [ "$status" = Running ] || return 0
    [ -z "$rid" ] || timeout 120 jl run logs "$rid" --tail 15 || true
    # shellcheck disable=SC2016  # the loop runs on the instance
    timeout 180 jl exec "$id" -- sh -lc 'for d in /home/ANLP-Project /home/*/ANLP-Project; do [ -d "$d" ] && cd "$d" && exec bash scripts/jarvis_t12.sh status; done' || true
}

cmd_watch() {
    local phase="${1:-}" dir hours start now rc
    check_phase "$phase"; need_jl
    dir="$(phase_dir "$phase")"
    hours="$(cat "$dir/HOURS")"
    start="$(cat "$dir/launch.epoch" 2>/dev/null || date +%s)"
    while :; do
        rc=0
        finalize "$phase" resume || rc=$?
        [ "$rc" = 2 ] || return "$rc"
        now="$(date +%s)"
        if [ "$now" -ge "$(awk -v s="$start" -v h="$hours" 'BEGIN { printf "%d", s + h * 3600 + 2700 }')" ]; then
            info "45 min past the job's budget and it still reports running: force-pausing"
            FORCE=1 cmd_pause "$phase"
            return $?
        fi
        sleep 120
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
    [ -n "${id:-}" ] || die "no instance named $name"
    [ "$status" = Running ] || { info "instance $id is $status"; return 0; }
    rid="$(latest_run_id "$id")"
    state="$(run_state "$id" "$rid")"
    if [ "$state" = running ] && [ "${FORCE:-0}" != 1 ]; then
        die "the job is still running on $id; FORCE=1 pauses it anyway (outputs written so far stay on /home)"
    fi
    pause_instance "$id" "$name"
}

case "${1:-}" in
    prepare) shift; cmd_prepare "$@" ;;
    launch) shift; cmd_launch "$@" ;;
    status) shift; cmd_status "$@" ;;
    watch) shift; cmd_watch "$@" ;;
    fetch) shift; cmd_fetch "$@" ;;
    pause) shift; cmd_pause "$@" ;;
    *) echo "usage: jl_t12.sh prepare <phase> <commit> | launch|status|watch|fetch|pause <phase>   (phase: calib|smoke|report|report-gsm8k|report-mbpp)"; exit 2 ;;
esac
