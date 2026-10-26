#!/bin/bash
# The T12 instance's own pause switch.  Runs ON a Jarvis Labs instance, detached from every job
# and from every ssh session, so that a dropped laptop link, a closed terminal or a crashed
# `jl` can never leave the instance billing.  Started by scripts/jl_t12.sh (see there):
#
#   t12_watchdog.sh job PID PHASE   PID is the job's process (the upload's bootstrap, which execs
#                                   into the job): when it has exited, wait the fetch grace so the
#                                   laptop can download the outputs, then pause this instance.
#                                   Whatever the job does, pause at the global deadline + 30 min.
#   t12_watchdog.sh fetch           no job: pause after the grace.  Started right after a create,
#                                   a resume for fetching and a resume for a continued run, so
#                                   that there is never a minute without a watchdog.
#
# Every start replaces the previous watchdog of the instance (its pid is in /home/.t12-watchdog.pid
# and is checked against /proc before it is killed).  Everything it does goes to
# /home/t12-watchdog.log.
#
# The secret.  /home/.t12.env (chmod 600, uploaded by the laptop; /home survives a pause) holds
# JL_API_KEY -- a DEDICATED Jarvis API key, to be revoked after the sweep -- plus T12_INSTANCE_NAME,
# T12_MACHINE_ID, T12_BUDGET_REMAINING_S (seconds of budget left when the file was written) and
# T12_FETCH_GRACE_MIN (default 20).  The key is only ever sent as a header read from a chmod-600
# file (curl -H @file); it is never printed, logged or put on a command line.
#
# Pausing.  The machine id changes on a resume, so the instance is looked up BY NAME at the moment
# of the pause (GET users/fetch).  Then `POST <region backend>/misc/pause?machine_id=<id>` with the
# Bearer key -- exactly what `jl pause` sends (jarvislabs/instances.py) but with nothing to install.
# If that does not work three times, `jl pause <id> --yes` from a small venv (installed on demand).
# A pause that was accepted but did not stop this process within 90 s is retried.  An instance
# that is already paused (or being paused) is left alone: pausing twice is harmless.
#
# Test knobs (tests/test_t12_watchdog.py): T12_WD_ENV, T12_WD_LOG, T12_WD_PIDFILE, T12_WD_HOLD,
# T12_WD_POLL (15 s), T12_WD_MARGIN_S (1800), T12_FETCH_GRACE_S (overrides the minutes),
# T12_WD_HOLD_FRESH_S (600), T12_WD_CONFIRM_S (90), T12_WD_RETRY_S (5), T12_WD_SLOW_S (300), T12_WD_PY, T12_WD_TMP.
set +x
set -u
umask 077
MODE="${1:-}"
JOBPID="${2:-}"
PHASE="${3:-}"
case "$MODE" in
    job) [ -n "$JOBPID" ] || { echo "usage: $0 job PID PHASE | fetch" >&2; exit 2; } ;;
    fetch) ;;
    *) echo "usage: $0 job PID PHASE | fetch" >&2; exit 2 ;;
esac
ENVF="${T12_WD_ENV:-/home/.t12.env}"
LOG="${T12_WD_LOG:-/home/t12-watchdog.log}"
PIDF="${T12_WD_PIDFILE:-/home/.t12-watchdog.pid}"
HOLD="${T12_WD_HOLD:-/home/.t12-hold}"
POLL="${T12_WD_POLL:-15}"
MARGIN="${T12_WD_MARGIN_S:-1800}"
HOLD_FRESH="${T12_WD_HOLD_FRESH_S:-600}"
CONFIRM="${T12_WD_CONFIRM_S:-90}"
TMPD="${T12_WD_TMP:-/home}"

# sleep that a TERM interrupts at once (a plain `sleep` would delay the trap)
nap() { sleep "$1" & wait $! 2>/dev/null || true; }

log() { printf '%s [watchdog %s %s] %s\n' "$(date -u +%FT%TZ)" "$$" "$MODE" "$*" >> "$LOG" 2>/dev/null || true; }

PYB="${T12_WD_PY:-$(command -v python3 || command -v python || true)}"
[ -f "$ENVF" ] || { log "FATAL: no $ENVF: this instance cannot be paused by the watchdog"; exit 1; }
[ -n "$PYB" ] || { log "FATAL: no python3 to read the API's JSON"; exit 1; }
set -a
# shellcheck disable=SC1090
. "$ENVF"
set +a
[ -n "${JL_API_KEY:-}" ] || { log "FATAL: no JL_API_KEY in $ENVF"; exit 1; }
NAME="${T12_INSTANCE_NAME:-}"
[ -n "$NAME" ] || { log "FATAL: no T12_INSTANCE_NAME in $ENVF"; exit 1; }
REMAIN="${T12_BUDGET_REMAINING_S:-0}"
if [ -n "${T12_FETCH_GRACE_S:-}" ]; then GRACE="$T12_FETCH_GRACE_S"
else GRACE="$(awk -v m="${T12_FETCH_GRACE_MIN:-20}" 'BEGIN { printf "%d", m * 60 }')"; fi

# ---- the previous watchdog of this instance goes away; this one's pid is recorded -------------
if [ -s "$PIDF" ]; then
    old="$(cat "$PIDF" 2>/dev/null || true)"
    if [ -n "$old" ] && [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null \
        && tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null | grep -q 't12.watchdog'; then
        kill "$old" 2>/dev/null || true
        log "replaced the previous watchdog (pid $old)"
    fi
fi
echo "$$" > "$PIDF"
HDR="$(mktemp "$TMPD/.t12-hdr.XXXXXX")"
printf 'Authorization: Bearer %s\n' "$JL_API_KEY" > "$HDR"
# shellcheck disable=SC2329  # runs from the EXIT trap
cleanup() {
    rm -f "$HDR"
    [ "$(cat "$PIDF" 2>/dev/null || true)" != "$$" ] || rm -f "$PIDF"
}
trap cleanup EXIT
trap 'log "terminated"; exit 0' TERM INT

# ---- the API ------------------------------------------------------------------------------------
BASE_N="https://backendn.jarvislabs.net"   # IN2 (jl's default region: where `users/fetch` lists)
BASE_C="https://backendc.jarvislabs.net"   # IN1
BASE_E="https://backendeu.jarvislabs.net"  # EU1

base_for() {
    case "$1" in
        IN1|india-chennai-01) echo "$BASE_C" ;;
        IN2|india-noida-01) echo "$BASE_N" ;;
        EU1|europe-01) echo "$BASE_E" ;;
        *) echo "" ;;
    esac
}

# http METHOD URL: the body, then a last line "HTTP <code>".
http() {
    curl -sS --max-time 60 -X "$1" -H "@$HDR" -w '\nHTTP %{http_code}\n' "$2" 2>> "$LOG"
}

# body_if_2xx: stdin is http()'s output; prints the body when the code is 2xx.
body_if_2xx() {
    local out code
    out="$(cat)"
    code="$(printf '%s\n' "$out" | tail -n 1 | awk '{ print $2 }')"
    case "$code" in
        2??) printf '%s\n' "$out" | sed '$d' ;;
        *) return 1 ;;
    esac
}

# find_self: "machine_id status region" of the newest instance named $NAME (empty if none).
find_self() {
    local base body row
    for base in "$BASE_N" "$BASE_C" "$BASE_E"; do
        body="$(http GET "$base/users/fetch" | body_if_2xx)" || continue
        row="$(printf '%s' "$body" | "$PYB" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
rows = d.get("instances", []) if isinstance(d, dict) else d
hit = [i for i in rows if (i.get("name") or i.get("instance_name")) == sys.argv[1]]
if hit:
    i = hit[-1]
    print(i.get("machine_id"), i.get("status") or "?", i.get("region") or "?")
' "$NAME" | tr -d '\r' | tail -n 1)"
        [ -z "$row" ] || { echo "$row"; return 0; }
    done
    return 1
}

# post_pause ID REGION: success when the backend says so (HTTP 2xx and not success=false).
post_pause() {
    local id="$1" region="$2" base bases out
    base="$(base_for "$region")"
    if [ -n "$base" ]; then bases="$base"; else bases="$BASE_N $BASE_C $BASE_E"; fi
    for base in $bases; do
        out="$(http POST "$base/misc/pause?machine_id=$id" | body_if_2xx)" || { log "POST $base/misc/pause?machine_id=$id was refused or failed"; continue; }
        if printf '%s' "$out" | "$PYB" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(0)            # a 2xx without JSON: take it, the status check decides
v = d.get("success", d.get("sucess", True)) if isinstance(d, dict) else True
sys.exit(0 if (str(v).lower() == "true" or v is True) else 1)'; then
            log "pause accepted for machine $id by $base"
            return 0
        fi
        log "pause for machine $id was declined by $base: $(printf '%s' "$out" | head -c 200)"
    done
    return 1
}

# jl_pause ID: the fallback, `jl pause` from a venv installed on demand (JL_API_KEY is exported).
jl_pause() {
    local id="$1" jl
    jl="$(command -v jl || true)"
    if [ -z "$jl" ]; then
        jl="$TMPD/.t12-jl/bin/jl"
        if [ ! -x "$jl" ]; then
            log "installing jarvislabs into $TMPD/.t12-jl for the fallback pause"
            { "$PYB" -m venv "$TMPD/.t12-jl" && "$TMPD/.t12-jl/bin/pip" install -q jarvislabs; } >> "$LOG" 2>&1 || { log "could not install jl"; return 1; }
        fi
    fi
    timeout 120 "$jl" pause "$id" --yes >> "$LOG" 2>&1
}

# pause_self: look this instance up by name and pause it; loop until it is paused (or we are
# killed because it is).  After 8 rounds it keeps trying every 5 minutes: a watchdog that gives
# up is worse than one that nags a refusing API.
pause_self() {
    local round=0 row id status region fails=0 retry="${T12_WD_RETRY_S:-5}"
    while :; do
        round=$((round + 1))
        if [ "$round" -gt 8 ]; then
            log "round $round: still not paused; next try in 5 min"
            nap "${T12_WD_SLOW_S:-300}"
        fi
        row="$(find_self)" || row=""
        if [ -n "$row" ]; then
            read -r id status region <<< "$row"
            case "$(printf '%s' "$status" | tr '[:upper:]' '[:lower:]')" in
                paused|pausing|stopped|stopping|destroyed|destroying|deleted)
                    log "instance $id ($NAME) is $status: nothing to do"
                    return 0 ;;
            esac
        else
            log "could not look up '$NAME' by name; using the recorded id ${T12_MACHINE_ID:-?}"
            id="${T12_MACHINE_ID:-}"; status="?"; region="?"
        fi
        [ -n "$id" ] || { log "no machine id to pause"; nap $((retry * 4)); continue; }
        log "round $round: pausing machine $id ($NAME, status $status, region $region)"
        if [ "$fails" -lt 3 ]; then
            if post_pause "$id" "$region"; then
                nap "$CONFIRM"
                log "still running ${CONFIRM}s after an accepted pause: retrying"
            else
                fails=$((fails + 1))
            fi
        else
            if jl_pause "$id"; then
                nap "$CONFIRM"
                log "still running ${CONFIRM}s after jl pause: retrying"
            else
                log "jl pause failed too"
            fi
            fails=0
        fi
        nap "$retry"
    done
}

hold_fresh() {
    local m
    [ -e "$HOLD" ] || return 1
    m="$(stat -c %Y "$HOLD" 2>/dev/null || echo 0)"
    [ $(($(date +%s) - m)) -lt "$HOLD_FRESH" ]
}

# ---- the clock ------------------------------------------------------------------------------------
start="$(date +%s)"
if [ "$MODE" = job ]; then
    hard=$((start + REMAIN + MARGIN))
else
    hard=$((start + 3 * 3600))
fi
log "armed: $NAME (machine ${T12_MACHINE_ID:-?}), phase ${PHASE:-none}, job pid ${JOBPID:-none}; grace ${GRACE}s; pause unconditionally at $(date -u -d "@$hard" +%FT%TZ) (in $((hard - start))s)"

if [ "$MODE" = job ]; then
    while kill -0 "$JOBPID" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$hard" ]; then
            log "GLOBAL DEADLINE + $((MARGIN / 60)) min reached with the job (pid $JOBPID) still running: pausing"
            pause_self
            exit $?
        fi
        nap "$POLL"
    done
    log "the job (pid $JOBPID) has exited; the laptop has ${GRACE}s to fetch the outputs"
fi
end_at=$(($(date +%s) + GRACE))
while :; do
    now="$(date +%s)"
    if [ "$now" -ge "$hard" ]; then
        log "hard limit reached: pausing"
        break
    fi
    if [ "$now" -ge "$end_at" ] && ! hold_fresh; then
        log "grace over (no fresh $HOLD): pausing"
        break
    fi
    nap "$POLL"
done
pause_self
exit $?
