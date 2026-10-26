"""scripts/jl_t12.sh (the laptop wrapper) against a fake `jl`: no Jarvis account, no network.

The fake (tests/fake_jl.py) keeps its state in a directory and logs every call.  What is under
test: the order of the calls of `launch` (create -> secret -> watchdog -> run), that the key
reaches only the instance's /home/.t12.env and the local copy is deleted, the wallet check, the
fall-through to `watch` after a dropped link, the fetch with ONE GPU, the same-instance resume,
the origin check, RESUME_FROM, the bootstrap's refusal to run without a watchdog.
"""
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from shell_helpers import BASH, ROOT, needs_bash, posix, run_bash, script_ok, stub

pytestmark = needs_bash

KEY = "SECRET-KEY-must-not-leak-9876543210"
FULL = "a" * 40


@pytest.fixture(autouse=True)
def _lf():
    script_ok("jl_t12.sh")
    script_ok("t12_watchdog.sh")


GPUS = [
    # a VM row must never give the price or the free count of a container
    {"gpu_type": "RTX-PRO6000", "region": "IN1", "num_free_devices": 0, "effective_num_free_devices": 0,
     "price_per_hour": 1.0, "workload_type": "vm"},
    {"gpu_type": "RTX-PRO6000", "region": "IN1", "num_free_devices": 4, "effective_num_free_devices": 4,
     "price_per_hour": 208.0, "workload_type": "container"},
    {"gpu_type": "H100", "region": "IN2", "num_free_devices": 1, "effective_num_free_devices": 1,
     "price_per_hour": 250.0, "workload_type": None},
]


class World:
    def __init__(self, tmp: Path, phase="smoke", hours="0.34", balance=10000.0, grants=0.0):
        tmp.mkdir(parents=True, exist_ok=True)
        self.tmp = tmp
        self.phase = phase
        self.state = tmp / "state"
        self.state.mkdir()
        self.stubs = tmp / "stubs"
        (tmp / "scripts").mkdir()
        shutil.copy(ROOT / "scripts" / "jl_t12.sh", tmp / "scripts" / "jl_t12.sh")
        stub(self.stubs, "jl", f'exec "{posix(sys.executable)}" "{posix(ROOT / "tests" / "fake_jl.py")}" "$@"')
        stub(self.stubs, "setsid", 'exec "$@"')           # Git Bash has no setsid
        self.put("account.json", {"balance": balance, "grants": grants, "currency": "INR"})
        self.put("gpus.json", GPUS)
        self.put("scenario.json", {"phase": phase})
        self.put("instances.json", [])
        short = FULL[:7]
        self.up = tmp / "runs" / "jl-upload" / f"{phase}-{short}"
        (self.up / "payload").mkdir(parents=True)
        (self.up / "COMMIT").write_text(FULL + "\n")
        (self.up / "HOURS").write_text(hours + "\n")
        (self.up / "NAME").write_text(f"t12-{phase}-{short}.sh\n")
        (self.up / f"t12-{phase}-{short}.sh").write_text("#!/bin/bash\n")
        (self.up / "payload" / "job.env").write_text("")
        shutil.copy(ROOT / "scripts" / "t12_watchdog.sh", self.up / "t12_watchdog.sh")
        (tmp / "runs" / "jl-upload" / f".last-{phase}").write_text(f"runs/jl-upload/{phase}-{short}\n")
        # what the instance's tarball holds, for the fetch
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            data = f"phase={phase} rc=0\n".encode()
            ti = tarfile.TarInfo(f".job-{phase}.status")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        (self.state / "outputs.tgz").write_bytes(buf.getvalue())

    def put(self, name, obj):
        (self.state / name).write_text(json.dumps(obj))

    def get(self, name):
        return json.loads((self.state / name).read_text())

    def scenario(self, **kw):
        sc = self.get("scenario.json")
        sc.update(kw)
        self.put("scenario.json", sc)

    def add_instance(self, mid, name, status, num_gpus=2):
        ins = self.get("instances.json")
        ins.append({"machine_id": mid, "name": name, "status": status, "num_gpus": num_gpus})
        self.put("instances.json", ins)

    def env(self, **extra):
        e = {"FAKE_JL_DIR": posix(self.state), "JL_API_KEY": KEY, "YES": "1", "T12_NAP": "0",
             "T12_WATCH_SLEEP": "1", "T12_FETCH_GRACE_MIN": "20"}
        e.update({k: str(v) for k, v in extra.items() if v is not None})
        for k, v in extra.items():
            if v is None:
                e.pop(k, None)
        return e

    def sh(self, args: str, timeout=120, **env):
        e = self.env(**env)
        return run_bash(f"bash scripts/jl_t12.sh {args}", e, cwd=self.tmp, stubs=self.stubs, timeout=timeout)

    def calls(self):
        f = self.state / "calls.log"
        return [ln.split(" ", 1)[1] for ln in f.read_text().splitlines()] if f.exists() else []

    def verbs(self):
        return [c.split()[0] if c.split()[0] != "run" else "run " + c.split()[1] for c in self.calls()]


def no_key_anywhere(w: World, *outputs: str):
    texts = list(outputs)
    texts.append((w.state / "calls.log").read_text())
    for f in w.up.glob("*.log"):
        texts.append(f.read_text())
    for t in texts:
        assert KEY not in t


# --------------------------------------------------------------------------- #
# launch
# --------------------------------------------------------------------------- #

def test_launch_creates_arms_the_watchdog_then_runs_then_fetches_and_pauses(tmp_path):
    w = World(tmp_path)
    r = w.sh("launch smoke")
    assert r.returncode == 0, (r.stdout, r.stderr)
    calls = w.calls()
    idx = lambda pat: next(i for i, c in enumerate(calls) if re.search(pat, c))  # noqa: E731
    create, sec, wd, guard = idx(r"^create "), idx(r"^upload 901 .*/home/\.t12\.env"), \
        idx(r"^upload 901 .*\.t12-watchdog\.sh\.new"), idx(r"^exec 901 .*setsid")
    run = idx(r"^run t12-smoke-aaaaaaa\.sh --on 901 --yes")
    assert create < min(sec, wd) and max(sec, wd) < guard < run      # armed before anything else
    # the guard's command: chmod 600 FIRST, the absolute deadline, jl for the fallback, a nonce
    g = calls[guard]
    assert g.index("chmod 600 /home/.t12.env") < g.index("setsid") and "/home/.t12-deadline" in g
    assert "pip install -q --target /home/.t12-jl jarvislabs" in g and "T12_WD_NONCE=" in g
    assert "--gpu RTX-PRO6000 --num-gpus 1 --region IN1" in calls[create] and "--json --yes" in calls[create]
    pause = idx(r"^pause 901 --yes")
    assert run < idx(r"^download 901") < pause
    assert not any(c.startswith("destroy") for c in calls)
    assert sum(1 for c in calls if c.startswith("pause")) == 1
    assert "PAUSED instance 901" in r.stdout and "FETCHED into runs/t12/jl-smoke-aaaaaaa" in r.stdout
    assert (tmp_path / "runs" / "t12" / "jl-smoke-aaaaaaa" / ".job-smoke.status").exists()
    # the secret that went up: the key, the name and id, what is left of the budget (20 min - a bit)
    sec_txt = (w.state / "up-home_.t12.env").read_text()
    assert f"JL_API_KEY={KEY}" in sec_txt and "T12_MACHINE_ID=901" in sec_txt
    assert re.search(r"T12_INSTANCE_NAME=t12-smoke-aaaaaaa-\d+", sec_txt)
    rem = int(re.search(r"T12_BUDGET_REMAINING_S=(\d+)", sec_txt).group(1))
    assert 1000 < rem <= 0.34 * 3600
    assert "T12_FETCH_GRACE_MIN=20" in sec_txt
    # the local copy is gone (it was in a private temp dir, not in the repo) and nowhere else
    src = Path((w.state / "up-secret-src.txt").read_text())
    assert not src.exists() and not src.parent.exists()
    assert not list(tmp_path.rglob("secret.env"))
    no_key_anywhere(w, r.stdout, r.stderr)


def test_the_key_stays_out_of_a_traced_run(tmp_path):
    w = World(tmp_path)
    r = run_bash("bash -x scripts/jl_t12.sh launch smoke", w.env(), cwd=tmp_path, stubs=w.stubs, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    no_key_anywhere(w, r.stdout, r.stderr)


def test_launch_refuses_without_a_dedicated_key(tmp_path):
    w = World(tmp_path)
    r = w.sh("launch smoke", JL_API_KEY=None)
    assert r.returncode == 1 and "JL_API_KEY is not set" in r.stderr and "revoke" in r.stderr
    assert "DEDICATED" in r.stderr
    assert not any(c.startswith(("create", "upload", "run")) for c in w.calls())


def test_the_wallet_must_cover_the_budget_and_the_override_is_loud(tmp_path):
    # smoke: 208 x 1 GPU x (0.34 + 0.5) h = 174.72
    w = World(tmp_path, balance=100.0)
    r = w.sh("launch smoke")
    assert r.returncode == 1 and "wallet is short" in r.stderr and "174.72" in r.stderr
    assert not any(c.startswith(("create", "upload", "run")) for c in w.calls())
    r = w.sh("launch smoke", T12_SKIP_WALLET_CHECK=1)
    assert r.returncode == 0 and "T12_SKIP_WALLET_CHECK=1" in r.stderr and "WARNING" in r.stderr
    assert any(c.startswith("create") for c in w.calls())


def test_grants_count_towards_the_wallet(tmp_path):
    w = World(tmp_path, phase="calib", hours="2.5", balance=1000.0, grants=300.0)
    # calib: 208 x 2 x (2.5 + 0.5) = 1248 <= 1300
    r = w.sh("launch calib", JL_NUM_GPUS=2)
    assert r.returncode == 0, r.stderr
    w2 = World(tmp_path / "b", phase="calib", hours="2.5", balance=1000.0, grants=200.0)
    r = w2.sh("launch calib")
    assert r.returncode == 1 and "1248.00" in r.stderr


def test_a_dropped_link_does_not_end_the_launch(tmp_path):
    w = World(tmp_path)
    w.scenario(polls_running=2)             # `jl run` returns at once; the job runs on for 2 polls
    r = w.sh("launch smoke")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "STILL RUNNING" in r.stderr and "FETCHED" in r.stdout and "PAUSED instance 901" in r.stdout
    statuses = [c for c in w.calls() if c.startswith("run status")]
    assert len(statuses) >= 3                # polled until the job had ended
    assert w.verbs().index("download") < w.verbs().index("pause")


def test_no_job_without_an_armed_watchdog(tmp_path):
    w = World(tmp_path)
    w.scenario(no_armed=True)
    r = w.sh("launch smoke")
    assert r.returncode != 0 and "watchdog is not running" in r.stderr
    assert not any(c.startswith("run t12") for c in w.calls())
    assert any(c.startswith("pause 901") for c in w.calls())     # and the instance is paused again


def test_create_failure_leaves_nothing_billing(tmp_path):
    w = World(tmp_path)
    w.scenario(create_fails=True)
    r = w.sh("launch smoke")
    assert r.returncode != 0 and "no instance named" in r.stderr
    assert not any(c.startswith(("upload", "run t12", "pause")) for c in w.calls())


def test_fetch_resumes_a_paused_instance_with_one_gpu_then_pauses(tmp_path):
    w = World(tmp_path, phase="calib", hours="2.5")
    (w.up / "launch.name").write_text("t12-calib-aaaaaaa-10261200\n")
    w.add_instance(905, "t12-calib-aaaaaaa-10261200", "Paused", num_gpus=2)
    w.scenario(new_id_on_resume=906)
    r = w.sh("fetch calib")
    assert r.returncode == 0, (r.stdout, r.stderr)
    calls = w.calls()
    assert "resume 905 --num-gpus 1 --yes" in calls
    assert any(c.startswith("exec 906") and "setsid" in c and "T12_MACHINE_ID=906" in c for c in calls)     # a guard first, with the new id
    assert w.verbs().index("resume") < w.verbs().index("download") < w.verbs().index("pause")
    assert "pause 906 --yes" in calls                                       # the id after the resume
    inst = w.get("instances.json")[0]
    assert inst["status"] == "Paused" and inst["num_gpus"] == 1


def test_resume_the_same_instance_with_ON(tmp_path):
    w = World(tmp_path, phase="calib", hours="2.5")
    w.add_instance(905, "t12-calib-aaaaaaa-10250000", "Paused", num_gpus=1)
    w.scenario(new_id_on_resume=906, budget_left=3000)
    r = w.sh("launch calib", ON=905, JL_NUM_GPUS=2)
    assert r.returncode == 0, (r.stdout, r.stderr)
    calls = w.calls()
    assert "resume 905 --num-gpus 2 --yes" in calls
    assert not any(c.startswith("create") for c in calls)
    assert any(re.match(r"^upload 906 .*/home/\.t12\.env", c) for c in calls)
    assert any(re.match(r"^run t12-calib-aaaaaaa\.sh --on 906 --yes", c) for c in calls)
    assert "T12_MACHINE_ID=906" in (w.state / "up-home_.t12.env").read_text()
    assert (w.up / "launch.name").read_text().strip() == "t12-calib-aaaaaaa-10250000"
    assert "pause 906 --yes" in calls


def test_a_spent_budget_on_the_instance_needs_NEW_BUDGET(tmp_path):
    w = World(tmp_path, phase="calib", hours="2.5")
    w.add_instance(905, "t12-calib-aaaaaaa-10250000", "Paused", num_gpus=2)
    w.scenario(new_id_on_resume=906, budget_left=-30)
    r = w.sh("launch calib", ON=905)
    assert r.returncode != 0 and "budget on instance 906 is spent" in r.stderr and "NEW_BUDGET=1" in r.stderr
    assert not any(c.startswith("run t12") for c in w.calls())
    assert "pause 906 --yes" in w.calls()
    w2 = World(tmp_path / "b", phase="calib", hours="2.5")
    w2.add_instance(905, "t12-calib-aaaaaaa-10250000", "Paused", num_gpus=2)
    w2.scenario(new_id_on_resume=906, budget_left=-30)
    (w2.up / "payload" / "job.env").write_text("export NEW_BUDGET=1\n")
    r = w2.sh("launch calib", ON=905)
    assert r.returncode == 0, r.stderr
    assert any(c.startswith("run t12") for c in w2.calls())


def test_status_shows_the_watchdog_log(tmp_path):
    w = World(tmp_path)
    (w.up / "launch.name").write_text("t12-smoke-aaaaaaa-1\n")
    w.add_instance(901, "t12-smoke-aaaaaaa-1", "Running", num_gpus=1)
    r = w.sh("status smoke")
    assert r.returncode == 0
    assert any("t12-watchdog.log" in c for c in w.calls())


# --------------------------------------------------------------------------- #
# the pieces, sourced
# --------------------------------------------------------------------------- #

def sourced(w: World, body: str, **env):
    script = "source scripts/jl_t12.sh\n" + body
    return run_bash(script, w.env(**env), cwd=w.tmp, stubs=w.stubs, timeout=60)


def test_gpu_row_takes_the_container_row_and_reads_a_price_per_gpu(tmp_path):
    w = World(tmp_path)
    assert sourced(w, "gpu_row RTX-PRO6000 IN1").stdout.strip() == "4 208.0"
    assert sourced(w, "gpu_row rtx-pro6000 in1").stdout.strip() == "4 208.0"
    assert sourced(w, "gpu_row RTX-PRO6000 IN2").stdout.strip() == "noregion IN1"
    assert sourced(w, "gpu_row H100 IN2").stdout.strip() == "1 250.0"       # workload_type null = both
    assert sourced(w, "gpu_row B300 IN1").stdout.split()[0] == "absent"


def test_wallet_check_cases(tmp_path):
    w = World(tmp_path, balance=4159.0)
    r = sourced(w, "wallet_check 9.5 2 208")                    # need 208 x 2 x 10 = 4160
    assert r.returncode == 1 and "short" in r.stderr and "4160.00" in r.stderr
    w.put("account.json", {"balance": 4000.0, "grants": 160.0, "currency": "INR"})
    r = sourced(w, "wallet_check 9.5 2 208")
    assert r.returncode == 0 and "covers 4160.00" in r.stdout
    r = sourced(w, "wallet_check 9.5 2 None")
    assert r.returncode == 1 and "could not read" in r.stderr
    r = sourced(w, "wallet_check 9.5 2 ''", T12_SKIP_WALLET_CHECK=1)
    assert r.returncode == 0 and "NOT checked" in r.stderr


def test_default_hours_agree_between_the_laptop_and_the_job(tmp_path):
    w = World(tmp_path)
    for phase, want in (("calib", "2.5"), ("smoke", "0.34"), ("report", "9.0"),
                        ("report-gsm8k", "9.0"), ("report-mbpp", "9.0")):
        laptop = sourced(w, f"default_hours {phase}").stdout.strip()
        job = run_bash(f'eval "$(sed -n "/^phase_hours() {{/,/^}}/p" scripts/jarvis_t12.sh)"; phase_hours {phase}',
                       cwd=ROOT).stdout.strip()
        assert laptop == job == want, phase


def _git(cwd, *args, check=True):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@x")
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=check).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="no git")
def test_origin_check_asks_origin_now_not_a_stale_tracking_ref(tmp_path):
    w = World(tmp_path / "w")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    work, other = tmp_path / "work", tmp_path / "other"
    _git(tmp_path, "clone", "-q", str(origin), str(work))
    _git(work, "checkout", "-q", "-b", "feat")
    (work / "f").write_text("1")
    _git(work, "add", "f")
    _git(work, "commit", "-q", "-m", "c1")
    c1 = _git(work, "rev-parse", "HEAD")
    _git(work, "push", "-q", "origin", "feat")
    (work / "f").write_text("2")
    _git(work, "commit", "-qam", "c2")
    c2 = _git(work, "rev-parse", "HEAD")

    def check(c):
        r = run_bash(f'source "{posix(w.tmp / "scripts" / "jl_t12.sh")}"; cd "{posix(work)}"\n'
                     f'if origin_has_commit {c}; then echo "YES $ORIGIN_BRANCH ${{ORIGIN_SHA:0:7}}"; else echo NO; fi',
                     w.env(), cwd=w.tmp, stubs=w.stubs, timeout=60)
        return r.stdout.strip().splitlines()[-1]

    assert check(c1).startswith("YES feat")
    assert check(c2) == "NO"                                      # committed, not pushed
    _git(work, "push", "-q", "origin", "feat")
    assert check(c2).startswith("YES feat")
    # somebody force-moves origin's branch back to c1: our remote-tracking ref still says c2
    _git(tmp_path, "clone", "-q", "-b", "feat", str(origin), str(other))
    _git(other, "push", "-qf", "origin", f"{c1}:refs/heads/feat")
    assert "origin/feat" in _git(work, "branch", "-r", "--contains", c2)      # the stale ref lies
    assert check(c2) == "NO"
    assert check(c1).startswith("YES feat")
    # a branch that has moved on: the commit is an ancestor, and the branch is fetched by name
    _git(other, "pull", "-q", "origin", "feat")
    (other / "g").write_text("x")
    _git(other, "add", "g")
    _git(other, "commit", "-q", "-m", "c3")
    _git(other, "push", "-q", "origin", "feat")
    assert check(c1).startswith("YES feat")


def test_resume_pack_needs_the_same_commit_and_the_same_shard_count(tmp_path):
    w = World(tmp_path)
    src = tmp_path / "jl-report-aaaaaaa"
    for task in ("gsm8k", "mbpp"):
        d = src / f"sweep-{task}"
        d.mkdir(parents=True)
        for i in (0, 1):
            (d / f"t12.shard{i}of2.jsonl").write_text('{"doc_id": 1}\n{"doc_id": 3}\n')
            (d / f"t12.shard{i}of2.meta.json").write_text(
                json.dumps({"num_shards": 2, "starts": [{"git_commit": FULL}, {"git_commit": FULL}]}))
        (d / "worker0.log").write_text("not packed")
    dest = tmp_path / "dest"
    r = sourced(w, f'resume_pack "{posix(src)}" "{posix(dest)}" {FULL} 2')
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (dest / "sweep-gsm8k").iterdir()) == [
        "t12.shard0of2.jsonl", "t12.shard0of2.meta.json", "t12.shard1of2.jsonl", "t12.shard1of2.meta.json"]
    assert "2 records" in r.stdout
    r = sourced(w, f'resume_pack "{posix(src)}" "{posix(tmp_path / "d2")}" {"b" * 40} 2')
    assert r.returncode != 0 and "same commit is required" in r.stderr
    r = sourced(w, f'resume_pack "{posix(src)}" "{posix(tmp_path / "d3")}" {FULL} 1')
    assert r.returncode != 0 and "same number of GPUs is required" in r.stderr
    r = sourced(w, f'resume_pack "{posix(tmp_path)}" "{posix(tmp_path / "d4")}" {FULL} 2')
    assert r.returncode != 0 and "no sweep-" in r.stderr


# --------------------------------------------------------------------------- #
# the bootstrap that runs on the instance
# --------------------------------------------------------------------------- #

def make_bootstrap(w: World, phase="smoke") -> Path:
    f = w.tmp / "boot.sh"
    r = sourced(w, f'write_bootstrap "{posix(f)}" {phase} {FULL}')
    assert r.returncode == 0, r.stderr
    subprocess.run([BASH, "-n", str(f)], check=True)
    return f


def test_the_bootstrap_refuses_a_job_nothing_would_pause(tmp_path):
    w = World(tmp_path)
    boot = make_bootstrap(w)
    env = {"T12_SECRET": posix(tmp_path / "t12.env"), "T12_WATCHDOG": posix(tmp_path / "wd.sh"),
           "T12_WD_LOG": posix(tmp_path / "wd.log"), "T12_BASE": posix(tmp_path / "base"), "T12_ARM_WAIT": "2"}
    r = run_bash(f'bash "{posix(boot)}"', {**w.env(), **env}, cwd=tmp_path, stubs=w.stubs)
    assert r.returncode == 1 and "no " in r.stderr and "t12.env" in r.stderr and "refusing" in r.stderr
    (tmp_path / "t12.env").write_text("JL_API_KEY=x\n")
    r = run_bash(f'bash "{posix(boot)}"', {**w.env(), **env}, cwd=tmp_path, stubs=w.stubs)
    assert r.returncode == 1 and "wd.sh" in r.stderr and "refusing" in r.stderr
    # a watchdog that never reports 'armed'
    stub(tmp_path, "wd.sh", "sleep 30")
    r = run_bash(f'bash "{posix(boot)}"', {**w.env(), **env}, cwd=tmp_path, stubs=w.stubs, timeout=60)
    assert r.returncode == 1 and "did not arm" in r.stderr
    assert not (tmp_path / "base" / "ANLP-Project").exists()               # nothing was checked out
    # one that does: started with `job <bootstrap pid> <phase>`, then the bootstrap goes on (and
    # fails here, there is no payload)
    stub(tmp_path, "wd.sh", 'echo "$@" > "$T12_WD_LOG.args"; echo "armed: nonce $T12_WD_NONCE; x" >> "$T12_WD_LOG"; sleep 20')
    r = run_bash(f'bash "{posix(boot)}"', {**w.env(), **env}, cwd=tmp_path, stubs=w.stubs, timeout=60)
    assert "watchdog armed for pid" in r.stdout
    args = (tmp_path / "wd.log.args").read_text().split()
    assert args[0] == "job" and args[1].isdigit() and args[2] == "smoke"


@pytest.mark.skipif(shutil.which("git") is None, reason="no git")
def test_the_bootstrap_restores_partial_shards_only_where_the_instance_has_none(tmp_path):
    w = World(tmp_path)
    boot = make_bootstrap(w, "report")
    repo = tmp_path / "src"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "jarvis_t12.sh").write_text('#!/bin/bash\n[ "$1" = verify-inputs ] && exit 0\nexit 1\n')
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")
    commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/jl-t12/commit", commit)
    pay = tmp_path / "payload"
    (pay / "resume" / "sweep-gsm8k").mkdir(parents=True)
    _git(repo, "bundle", "create", str(pay / "repo.bundle"), "refs/jl-t12/commit")
    (pay / "job.env").write_text("")
    for i in (0, 1):
        (pay / "resume" / "sweep-gsm8k" / f"t12.shard{i}of2.jsonl").write_text('{"doc_id": 1}\n{"doc_id": 3}\n')
        (pay / "resume" / "sweep-gsm8k" / f"t12.shard{i}of2.meta.json").write_text("{}")
    tgz = tmp_path / "payload.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(pay, arcname=".")
    text = boot.read_bytes().replace(FULL.encode(), commit.encode())      # bytes: no CRLF on Windows
    boot.write_bytes(text + base64.b64encode(tgz.read_bytes()) + b"\n")
    base = tmp_path / "base"
    # the instance already has shard 1 with more progress: it is kept
    have = base / "ANLP-Project" / "runs" / "t12" / "sweep-gsm8k"
    env = {**w.env(), "T12_DRYRUN": "1", "T12_BASE": posix(base)}
    (base / "ANLP-Project").mkdir(parents=True)
    _git(base / "ANLP-Project", "init", "-q")
    have.mkdir(parents=True)
    (have / "t12.shard1of2.jsonl").write_text('{"doc_id": 1}\n{"doc_id": 3}\n{"doc_id": 5}\n')
    r = run_bash(f'bash "{posix(boot)}"', env, cwd=tmp_path, timeout=90)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "restored" in r.stdout and "t12.shard0of2.jsonl" in r.stdout
    assert "keeping" in r.stdout and "t12.shard1of2.jsonl" in r.stdout
    assert (have / "t12.shard0of2.meta.json").exists()
    assert len((have / "t12.shard1of2.jsonl").read_text().splitlines()) == 3


# --------------------------------------------------------------------------- #
# an outage of the link or of the API must never be read as "nothing to pause"
# --------------------------------------------------------------------------- #

def _running_world(tmp_path, **sc):
    w = World(tmp_path)
    (w.up / "launch.name").write_text("t12-smoke-aaaaaaa-1\n")
    (w.up / "launch.id").write_text("901\n")
    (w.state / "run-started").write_text("1")
    w.add_instance(901, "t12-smoke-aaaaaaa-1", "Running", num_gpus=1)
    w.scenario(**sc)
    return w


def test_watch_keeps_retrying_through_a_jl_outage_and_finally_pauses(tmp_path):
    # `jl status` fails 5 times (what the old `watch` died on) and `jl list` 3 times
    w = _running_world(tmp_path, fail={"status": 5, "list": 3})
    r = w.sh("watch smoke", T12_RETRY_S=1)
    assert r.returncode == 0, (r.stdout, r.stderr)
    out = r.stdout + r.stderr
    assert "nothing to pause" not in out and "cannot read the instance list" in out
    verbs = w.verbs()
    assert verbs.count("FAILED") == 3 and "status" not in verbs        # watch does not need `jl status`
    assert "pause 901 --yes" in w.calls()
    assert max(i for i, v in enumerate(verbs) if v == "FAILED") < verbs.index("pause")
    assert "MAY STILL BE BILLING" not in out                          # it ended well: no banner
    assert w.get("instances.json")[0]["status"] == "Paused"


def test_a_failing_run_lookup_is_unknown_not_a_reason_to_pause_the_job(tmp_path):
    w = _running_world(tmp_path, fail={"run list": 4})
    r = w.sh("watch smoke", T12_RETRY_S=1)
    assert r.returncode == 0, (r.stdout, r.stderr)
    verbs = w.verbs()
    assert verbs.count("FAILED") == 4
    assert max(i for i, v in enumerate(verbs) if v == "FAILED") < verbs.index("pause")
    assert "download" in verbs and verbs.index("download") < verbs.index("pause")


def test_watch_that_cannot_settle_says_loudly_that_the_instance_may_be_billing(tmp_path):
    w = _running_world(tmp_path, fail={"list": 10000})
    r = w.sh("watch smoke", T12_RETRY_S=1, T12_WATCH_GIVEUP_S=3)
    assert r.returncode != 0
    assert "nothing to pause" not in r.stdout + r.stderr
    assert "MAY STILL BE BILLING" in r.stderr and "jl pause 901 --yes" in r.stderr and "jl list" in r.stderr
    assert "t12-smoke-aaaaaaa-1" in r.stderr
    assert not any(c.startswith("pause") for c in w.calls())


def test_the_wifi_dropping_right_after_create_still_ends_with_a_pause(tmp_path):
    # the secret cannot be uploaded (6 = 2 parallel uploads x 3 tries) and `jl list` is down for 2 calls
    w = World(tmp_path)
    w.scenario(fail={"upload": 6, "list": 2})
    r = w.sh("launch smoke", T12_RETRY_S=1)
    assert r.returncode != 0 and "could not upload the watchdog" in r.stderr
    out = r.stdout + r.stderr
    assert "nothing to pause" not in out
    assert "pause 901 --yes" in w.calls()
    assert not any(c.startswith("run t12") for c in w.calls())
    assert w.get("instances.json")[0]["status"] == "Paused"
    assert "MAY STILL BE BILLING" not in out


def test_a_launch_that_dies_with_the_api_down_prints_the_banner(tmp_path):
    w = World(tmp_path)
    w.scenario(fail={"upload": 6, "list": 10000})
    r = w.sh("launch smoke", T12_RETRY_S=1, T12_WATCH_GIVEUP_S=3)
    assert r.returncode != 0
    assert "MAY STILL BE BILLING" in r.stderr and "jl pause 901 --yes" in r.stderr


def test_a_fetch_keeps_the_watchdogs_hold_fresh_while_it_downloads(tmp_path):
    w = _running_world(tmp_path, download_sleep=3, phase="smoke")
    r = w.sh("fetch smoke", T12_HOLD_EVERY_S=1)
    assert r.returncode == 0, (r.stdout, r.stderr)
    touches = [c for c in w.calls() if c.startswith("exec 901") and "t12-hold" in c]
    assert len(touches) >= 3
    verbs = w.verbs()
    assert verbs.index("download") < verbs.index("pause")


def test_the_gpu_count_is_checked_before_anything_is_created(tmp_path):
    w = World(tmp_path, phase="calib", hours="2.5")
    (w.up / "payload" / "job.env").write_text("export NGPU=1\n")
    r = w.sh("launch calib")                                   # JL_NUM_GPUS defaults to 2
    assert r.returncode == 1 and "NGPU=1" in r.stderr
    (w.up / "payload" / "job.env").write_text("")
    (w.up / "RESUME_NGPU").write_text("1\n")
    r = w.sh("launch calib")
    assert r.returncode == 1 and "RESUME_FROM was packed for 1 shards" in r.stderr
    assert not any(c.startswith(("create", "resume", "upload")) for c in w.calls())
