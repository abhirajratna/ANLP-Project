from test_jl_t12 import _running_world


def test_a_connection_failure_in_run_status_mid_run_does_not_pause(tmp_path):
    # real jl 0.2.x: cli/run.py _get_instance swallows APIError("Request Connection failed:
    # GET users/fetch/<id>") and `jl run status --json` prints {"state": "instance-missing"}, exit 0
    w = _running_world(tmp_path, run_state="instance-missing", job_running=True)
    r = w.sh("watch smoke", T12_RETRY_S=1, T12_WATCH_GIVEUP_S=4)
    verbs = w.verbs()
    assert "pause" not in verbs and "download" not in verbs, (r.stdout, r.stderr)
    assert w.get("instances.json")[0]["status"] == "Running"


def test_a_stale_marker_plus_a_connection_failure_does_not_pause(tmp_path):
    # resumed instance: .job-<phase>.status left by the previous attempt
    w = _running_world(tmp_path, run_state="instance-missing", job_running=False)
    r = w.sh("watch smoke", T12_RETRY_S=1, T12_WATCH_GIVEUP_S=4)
    assert "pause" not in w.verbs(), (r.stdout, r.stderr)


def test_a_job_that_died_without_its_marker_is_still_paused_by_the_laptop(tmp_path):
    # jl's own exit_code file says the job ended (bootstrap FATAL, die before the trap, SIGKILL)
    w = _running_world(tmp_path, run="fail", job_running=True)
    r = w.sh("watch smoke", T12_RETRY_S=1, T12_WATCH_GIVEUP_S=4)
    assert "pause 901 --yes" in w.calls(), (r.stdout, r.stderr)
