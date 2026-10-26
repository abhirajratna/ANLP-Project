"""More of the watchdog: a pause that is under way, the absolute deadline file, the nonce, and the
job's supervisor restarting a watchdog that has died (ensure_watchdog in scripts/jarvis_t12.sh)."""
import time

import pytest

from shell_helpers import ROOT, needs_bash, posix, run_bash, script_ok, stub
from test_t12_watchdog import WD, curl_log, env_for, make_world

pytestmark = needs_bash


@pytest.fixture(autouse=True)
def _lf():
    script_ok("t12_watchdog.sh")
    script_ok("jarvis_t12.sh")


def test_an_instance_on_its_way_down_is_waited_for_and_only_a_stuck_pause_is_retried(tmp_path):
    d, ef = make_world(tmp_path, status="Pausing")
    t0 = time.time()
    r = run_bash(f"true & JOB=$!; wait $JOB\nbash {WD} job $JOB calib",
                 env_for(tmp_path, d, ef, T12_FETCH_GRACE_S=1, T12_WD_STUCK_S=4), stubs=d, timeout=60)
    assert r.returncode == 0, r.stderr
    paused = [ln for ln in curl_log(d) if "PAUSED 777" in ln]
    assert paused and int(paused[0].split()[0]) - int(t0) >= 4          # not before it had been stuck 4 s
    wd = (tmp_path / "wd.log").read_text()
    assert "waiting for it to be paused" in wd and "for over 4s: pausing again" in wd


def test_the_deadline_is_the_absolute_epoch_in_the_deadline_file(tmp_path):
    d, ef = make_world(tmp_path)
    (tmp_path / "dl").write_text(str(int(time.time()) + 3))       # the instance's clock; REMAIN says 100
    script = f"sleep 60 & JOB=$!\nbash {WD} job $JOB report\nrc=$?\nkill $JOB 2>/dev/null\nexit $rc\n"
    t0 = time.time()
    r = run_bash(script, env_for(tmp_path, d, ef, T12_BUDGET_REMAINING_S=100, T12_WD_MARGIN_S=0,
                                 T12_DEADLINE_FILE=posix(tmp_path / "dl"), T12_FETCH_GRACE_S=1000),
                 stubs=d, timeout=60)
    assert r.returncode == 0, r.stderr
    assert time.time() - t0 < 30 and any("PAUSED 777" in ln for ln in curl_log(d))
    assert "GLOBAL DEADLINE" in (tmp_path / "wd.log").read_text()


def test_the_margin_is_20_minutes_and_the_nonce_is_logged(tmp_path):
    assert "T12_WD_MARGIN_S:-1200" in (ROOT / "scripts" / "t12_watchdog.sh").read_text()
    d, ef = make_world(tmp_path)
    r = run_bash(f"true & JOB=$!; wait $JOB\nbash {WD} job $JOB calib",
                 env_for(tmp_path, d, ef, T12_FETCH_GRACE_S=1, T12_WD_NONCE="n0nce-42"), stubs=d, timeout=60)
    assert r.returncode == 0 and "nonce n0nce-42;" in (tmp_path / "wd.log").read_text()


def test_a_dead_watchdog_is_restarted_for_the_same_pid_and_a_live_one_is_left(tmp_path):
    stub(tmp_path / "bin", "setsid", 'exec "$@"')
    stub(tmp_path, "t12-watchdog.sh", 'echo "$@" >> "$ARGS"; echo $$ > "$T12_WD_PIDFILE"; sleep 30')
    (tmp_path / "t12.env").write_text("JL_API_KEY=x\n")
    t = posix(tmp_path)
    load = (f'eval "$(sed -n "/^ensure_watchdog() {{/,/^}}/p" "{posix(ROOT)}/scripts/jarvis_t12.sh")"\n'
            'WD_LAST_TRY=0\nPHASE=report\n'
            f'export T12_WATCHDOG="{t}/t12-watchdog.sh" T12_SECRET="{t}/t12.env" '
            f'T12_WD_PIDFILE="{t}/pid" ARGS="{t}/args" T12_WD_RESTART_MIN_S=0\n')
    body = (load + 'export T12_WATCHED_PID=4242\n'
            'ensure_watchdog; sleep 2\n'
            'ensure_watchdog; sleep 1\n'                        # alive: nothing new
            'kill "$(cat "$T12_WD_PIDFILE")"; sleep 1\n'
            'ensure_watchdog; sleep 2\n'                        # dead: restarted, same pid watched
            'kill "$(cat "$T12_WD_PIDFILE")" 2>/dev/null\n')
    r = run_bash(body, cwd=tmp_path, timeout=60, stubs=tmp_path / "bin")
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in (tmp_path / "args").read_text().split("\n") if ln]
    assert lines == ["job 4242 report", "job 4242 report"], r.stdout
    assert r.stdout.count("restarting it for pid 4242") == 2
    # no secret: it says it cannot, and starts nothing
    (tmp_path / "t12.env").unlink()
    (tmp_path / "args").unlink()
    r = run_bash(load + "export T12_WATCHED_PID=4242\nrm -f \"$T12_WD_PIDFILE\"\nensure_watchdog", cwd=tmp_path,
                 timeout=60, stubs=tmp_path / "bin")
    assert "NOT RUNNING and cannot be restarted" in r.stdout and not (tmp_path / "args").exists()
    # not under the laptop's bootstrap (no T12_WATCHED_PID): nothing at all
    r = run_bash(load + "ensure_watchdog", cwd=tmp_path, timeout=60, stubs=tmp_path / "bin")
    assert r.stdout.strip() == "" and not (tmp_path / "args").exists()
