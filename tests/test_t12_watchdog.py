"""scripts/t12_watchdog.sh: the on-instance pause switch, run against a stubbed `curl` / `jl`.

The decisions under test: the fetch grace after the job exits, the unconditional pause at the
global deadline + margin, the hold file, the lookup of the instance BY NAME (the machine id
changes on a resume), the fallback to `jl pause`, and that the key is never printed or logged.
Everything runs in seconds through the T12_WD_* knobs; no network, no Jarvis account.
"""
import json
import sys
import time

import pytest

from shell_helpers import ROOT, needs_bash, posix, run_bash, script_ok, stub

pytestmark = needs_bash

KEY = "KEY-do-not-print-0123456789"
WD = posix(ROOT / "scripts" / "t12_watchdog.sh")

# curl: GET users/fetch -> instances.json; POST misc/pause?machine_id=ID -> pauses the instance
# named in instances.json when ID is its id (else 400), unless fail_pause exists (500).  It
# logs "<epoch> METHOD URL [auth-ok]" and never the key itself.
CURL = r'''
set -u
D="$STUB_DIR"
method=GET; url=""; hdr=""
while [ $# -gt 0 ]; do
    case "$1" in
        -X) method="$2"; shift 2 ;;
        -H) hdr="${2#@}"; shift 2 ;;
        --max-time|-w) shift 2 ;;
        -sS|-s|-S) shift ;;
        *) url="$1"; shift ;;
    esac
done
auth=no
grep -q "^Authorization: Bearer $EXPECT_KEY\$" "$hdr" 2>/dev/null && auth=ok
echo "$(date +%s) $method $url auth-$auth" >> "$D/curl.log"
if [ "$auth" != ok ]; then printf '{"error":"bad key"}\nHTTP 401\n'; exit 0; fi
case "$method $url" in
    "GET "*/users/fetch)
        cat "$D/instances.json"; printf '\nHTTP 200\n' ;;
    "POST "*/misc/pause?machine_id=*)
        id="${url##*machine_id=}"
        if [ -e "$D/fail_pause" ]; then printf '{"success":false}\nHTTP 500\n'; exit 0; fi
        if grep -q "\"machine_id\": $id[,}]" "$D/instances.json"; then
            sed -i 's/"status": "\(Running\|Pausing\)"/"status": "Paused"/' "$D/instances.json"
            echo "$(date +%s) PAUSED $id" >> "$D/curl.log"
            printf '{"success":"true"}\nHTTP 200\n'
        else
            printf '{"success":false,"message":"no such machine"}\nHTTP 400\n'
        fi ;;
    *) printf '{}\nHTTP 404\n' ;;
esac
'''

JL = r'''
# `jl pause ID --yes`: needs the key in the environment
[ "${JL_API_KEY:-}" = "$EXPECT_KEY" ] || { echo "jl: no key" >&2; exit 1; }
echo "$(date +%s) jl $*" >> "$STUB_DIR/curl.log"
if [ "$1" = pause ]; then
    sed -i 's/"status": "Running"/"status": "Paused"/' "$STUB_DIR/instances.json"
    echo "$(date +%s) PAUSED-BY-JL $2" >> "$STUB_DIR/curl.log"
fi
'''


def make_world(tmp_path, *, status="Running", real_id=777, region="IN1"):
    d = tmp_path / "stub"
    stub(d, "curl", CURL)
    stub(d, "jl", JL)
    (d / "instances.json").write_text(json.dumps({"success": "true", "instances": [
        {"machine_id": 5, "name": "someone-else", "status": "Running", "region": "IN2"},
        {"machine_id": real_id, "name": "t12-calib-abc1234-10261200", "status": status,
         "region": region}]}, indent=1))
    env_file = tmp_path / "t12.env"
    env_file.write_text(f"JL_API_KEY={KEY}\nT12_INSTANCE_NAME=t12-calib-abc1234-10261200\n"
                        f"T12_MACHINE_ID=111\nT12_BUDGET_REMAINING_S=100\n")
    return d, env_file


def env_for(tmp_path, d, env_file, **extra):
    e = {"STUB_DIR": posix(d), "EXPECT_KEY": KEY, "T12_WD_ENV": posix(env_file),
         "T12_WD_LOG": posix(tmp_path / "wd.log"), "T12_WD_PIDFILE": posix(tmp_path / "wd.pid"),
         "T12_WD_HOLD": posix(tmp_path / "hold"), "T12_WD_TMP": posix(tmp_path),
         "T12_WD_PY": posix(sys.executable), "T12_WD_SLOW_S": "1", "T12_WD_POLL": "1", "T12_WD_CONFIRM_S": "1", "T12_WD_RETRY_S": "1",
         "T12_FETCH_GRACE_S": "3", "T12_WD_MARGIN_S": "0"}
    e.update({k: str(v) for k, v in extra.items()})
    if "T12_BUDGET_REMAINING_S" in e:      # the env file is sourced last: it must carry the value
        with open(env_file, "a") as fh:
            fh.write(f"T12_BUDGET_REMAINING_S={e.pop('T12_BUDGET_REMAINING_S')}\n")
    return e


def curl_log(d):
    f = d / "curl.log"
    return f.read_text().splitlines() if f.exists() else []


def posts(d):
    return [ln for ln in curl_log(d) if " POST " in ln]


@pytest.fixture(autouse=True)
def _lf():
    script_ok("t12_watchdog.sh")


def test_grace_after_the_job_exits_then_pause_by_name(tmp_path):
    d, ef = make_world(tmp_path)
    t0 = int(time.time())
    # the job lives 3 s and records when it exited; the grace is 3 s
    script = (f'(sleep 3; date +%s > "{posix(tmp_path)}/job_exit") & JOB=$!\n'
              f'bash {WD} job $JOB calib\n')
    r = run_bash(script, env_for(tmp_path, d, ef, T12_BUDGET_REMAINING_S=100), stubs=d, timeout=60)
    assert r.returncode == 0, r.stderr
    exit_t = int((tmp_path / "job_exit").read_text())
    log = curl_log(d)
    paused = [ln for ln in log if "PAUSED" in ln]
    assert paused and paused[0].endswith("PAUSED 777"), log
    pause_t = int(paused[0].split()[0])
    assert pause_t - exit_t >= 3 - 1                      # not before the grace has run out
    assert pause_t - t0 < 25
    # the instance was found by NAME: the recorded id 111 was never used, the region's backend was
    assert not any("machine_id=111" in ln for ln in log)
    assert any(" POST https://backendc.jarvislabs.net/misc/pause?machine_id=777 " in ln for ln in log)
    wd = (tmp_path / "wd.log").read_text()
    assert "armed" in wd and "has exited" in wd and "grace over" in wd and "pause accepted" in wd
    assert not list(tmp_path.glob(".t12-hdr.*"))           # the header file is gone


def test_pause_at_the_global_deadline_even_while_the_job_runs(tmp_path):
    d, ef = make_world(tmp_path)
    script = ('sleep 60 & JOB=$!\n'
              f'bash {WD} job $JOB report\nrc=$?\nkill $JOB 2>/dev/null\nexit $rc\n')
    t0 = time.time()
    r = run_bash(script, env_for(tmp_path, d, ef, T12_BUDGET_REMAINING_S=3, T12_WD_MARGIN_S=0,
                                 T12_FETCH_GRACE_S=1000), stubs=d, timeout=60)
    assert r.returncode == 0, r.stderr
    assert time.time() - t0 < 30                           # not the 1000 s grace, not the 60 s job
    assert any("PAUSED 777" in ln for ln in curl_log(d))
    assert "GLOBAL DEADLINE" in (tmp_path / "wd.log").read_text()


def test_the_deadline_margin_is_added(tmp_path):
    d, ef = make_world(tmp_path)
    script = ('sleep 60 & JOB=$!\n'
              f'bash {WD} job $JOB report\nrc=$?\nkill $JOB 2>/dev/null\nexit $rc\n')
    t0 = time.time()
    r = run_bash(script, env_for(tmp_path, d, ef, T12_BUDGET_REMAINING_S=1, T12_WD_MARGIN_S=3),
                 stubs=d, timeout=60)
    assert r.returncode == 0, r.stderr
    paused = [ln for ln in curl_log(d) if "PAUSED 777" in ln]
    assert paused and int(paused[0].split()[0]) - int(t0) >= 3     # remaining 1 s + margin 3 s


def test_a_fresh_hold_file_extends_the_grace_but_not_past_the_hard_limit(tmp_path):
    d, ef = make_world(tmp_path)
    hold = posix(tmp_path / "hold")
    # the laptop keeps touching the hold file for 6 s; the grace alone would end after 2 s
    script = (f'true & JOB=$!; wait $JOB\n'
              f'(for i in 1 2 3 4 5 6; do touch "{hold}"; sleep 1; done) &\n'
              f'bash {WD} job $JOB calib\n')
    t0 = time.time()
    r = run_bash(script, env_for(tmp_path, d, ef, T12_BUDGET_REMAINING_S=100, T12_FETCH_GRACE_S=2,
                                 T12_WD_HOLD_FRESH_S=2), stubs=d, timeout=60)
    assert r.returncode == 0, r.stderr
    paused = [ln for ln in curl_log(d) if "PAUSED 777" in ln]
    assert paused and int(paused[0].split()[0]) - int(t0) >= 6
    # ... but the hard limit wins over a hold that never goes stale
    (tmp_path / "b").mkdir(exist_ok=True)
    d2, ef2 = make_world(tmp_path / "b")
    hold = posix(tmp_path / "b" / "hold")
    script = (f'true & JOB=$!; wait $JOB\n'
              f'(for i in $(seq 1 30); do touch "{hold}"; sleep 1; done) &\nBG=$!\n'
              f'bash {WD} job $JOB calib\nrc=$?\nkill $BG 2>/dev/null\nexit $rc\n')
    t0 = time.time()
    r = run_bash(script, env_for(tmp_path / "b", d2, ef2, T12_BUDGET_REMAINING_S=3,
                                 T12_FETCH_GRACE_S=2, T12_WD_HOLD_FRESH_S=5), stubs=d2, timeout=60)
    assert r.returncode == 0, r.stderr
    assert any("PAUSED 777" in ln for ln in curl_log(d2))
    assert time.time() - t0 < 20
    assert "hard limit reached" in (tmp_path / "b" / "wd.log").read_text()


def test_an_instance_that_is_already_paused_is_left_alone(tmp_path):
    d, ef = make_world(tmp_path, status="Paused")
    script = f'true & JOB=$!; wait $JOB\nbash {WD} job $JOB calib\n'
    r = run_bash(script, env_for(tmp_path, d, ef, T12_FETCH_GRACE_S=1), stubs=d, timeout=60)
    assert r.returncode == 0, r.stderr
    assert posts(d) == []
    assert "nothing to do" in (tmp_path / "wd.log").read_text()


def test_fallback_to_jl_pause_when_the_post_keeps_failing(tmp_path):
    d, ef = make_world(tmp_path)
    (d / "fail_pause").write_text("x")
    script = f'true & JOB=$!; wait $JOB\nbash {WD} job $JOB calib\n'
    r = run_bash(script, env_for(tmp_path, d, ef, T12_FETCH_GRACE_S=1), stubs=d, timeout=90)
    assert r.returncode == 0, r.stderr
    log = curl_log(d)
    assert len([ln for ln in log if " POST " in ln]) >= 3
    assert any("PAUSED-BY-JL 777" in ln for ln in log)
    assert any(ln.endswith("jl pause 777 --yes") or " jl pause 777 --yes" in ln for ln in log)


def test_fetch_mode_pauses_after_the_grace_without_a_job(tmp_path):
    d, ef = make_world(tmp_path)
    t0 = time.time()
    r = run_bash(f"bash {WD} fetch", env_for(tmp_path, d, ef, T12_FETCH_GRACE_S=3), stubs=d,
                 timeout=60)
    assert r.returncode == 0, r.stderr
    paused = [ln for ln in curl_log(d) if "PAUSED 777" in ln]
    assert paused and int(paused[0].split()[0]) - int(t0) >= 3 - 1


def test_the_key_never_reaches_a_log_or_a_stream(tmp_path):
    d, ef = make_world(tmp_path)
    script = f'true & JOB=$!; wait $JOB\nbash -x {WD} job $JOB calib 2>&1\n'
    r = run_bash(script, env_for(tmp_path, d, ef, T12_FETCH_GRACE_S=1), stubs=d, timeout=60)
    # (even under `bash -x` the script switches tracing off before it reads the key)
    for text in (r.stdout, r.stderr, (tmp_path / "wd.log").read_text(),
                 (d / "curl.log").read_text()):
        assert KEY not in text
    assert "auth-ok" in (d / "curl.log").read_text()       # ... and it was sent, as a header file


def test_a_new_watchdog_replaces_the_previous_one(tmp_path):
    d, ef = make_world(tmp_path)
    env = env_for(tmp_path, d, ef, T12_FETCH_GRACE_S=1000)
    script = (f'bash {WD} fetch & OLD=$!\nsleep 2\n'
              f'bash {WD} fetch & NEW=$!\nsleep 2\n'
              f'if kill -0 $OLD 2>/dev/null; then echo OLD-ALIVE; else echo OLD-GONE; fi\n'
              f'kill $NEW 2>/dev/null; wait 2>/dev/null\n')
    r = run_bash(script, env, stubs=d, timeout=60)
    assert "OLD-GONE" in r.stdout, (r.stdout, r.stderr)
    assert "replaced the previous watchdog" in (tmp_path / "wd.log").read_text()


def test_it_refuses_to_run_without_the_env_file_or_the_key(tmp_path):
    d, ef = make_world(tmp_path)
    env = env_for(tmp_path, d, ef)
    env["T12_WD_ENV"] = posix(tmp_path / "missing.env")
    r = run_bash(f"bash {WD} fetch", env, stubs=d, timeout=30)
    assert r.returncode == 1 and "no " in (tmp_path / "wd.log").read_text()
    ef.write_text("T12_INSTANCE_NAME=x\n")
    r = run_bash(f"bash {WD} fetch", env_for(tmp_path, d, ef), stubs=d, timeout=30)
    assert r.returncode == 1 and "no JL_API_KEY" in (tmp_path / "wd.log").read_text()
    assert run_bash(f"bash {WD}", env, stubs=d, timeout=30).returncode == 2
