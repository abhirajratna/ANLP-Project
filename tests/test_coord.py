"""The team message bus, end to end: two clones of a throwaway bare remote."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

COORD = Path(__file__).resolve().parents[1] / "tools" / "coord.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x",
           GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@x", C3_COORD_INTERVAL="600")
ENV.pop("C3_MEMBER", None)


def sh(*args, cwd):
    return subprocess.run(args, cwd=cwd, env=ENV, check=True, capture_output=True, text=True)


def coord(clone, *args, stdin=None, env=None):
    return subprocess.run([sys.executable, str(COORD), *args], cwd=clone, input=stdin,
                          env={**ENV, **(env or {})}, capture_output=True, text=True)


@pytest.fixture
def team(tmp_path):
    origin = tmp_path / "origin.git"
    sh("git", "init", "-q", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    seed = tmp_path / "seed"
    sh("git", "clone", "-q", str(origin), str(seed), cwd=tmp_path)
    (seed / "README.md").write_text("x\n")
    sh("git", "add", ".", cwd=seed)
    sh("git", "commit", "-qm", "init", cwd=seed)
    sh("git", "push", "-q", "origin", "HEAD:main", cwd=seed)
    clones = {}
    for name in ("alice", "bob"):
        c = tmp_path / name
        sh("git", "clone", "-q", str(origin), str(c), cwd=tmp_path)
        r = coord(c, "whoami", "--set", name)
        assert r.returncode == 0, r.stderr
        clones[name] = c
    return origin, clones


def test_first_member_creates_the_bus_and_both_are_registered(team):
    origin, c = team
    assert "coord-bus" in sh("git", "branch", "--list", "coord-bus", cwd=origin).stdout
    assert coord(c["alice"], "members").stdout.split() == ["alice", "bob"]


def test_request_reply_close_round_trip(team):
    _, c = team
    r = coord(c["alice"], "send", "bob", "request", "rerun floor", "--body", "on LLaDA fp16")
    assert r.returncode == 0, r.stderr
    mid = r.stdout.split()[-1]

    inbox = coord(c["bob"], "inbox").stdout
    assert mid in inbox and "REQUEST from alice" in inbox and "on LLaDA fp16" in inbox

    r = coord(c["bob"], "reply", mid, "done in abc123", "--close")
    assert r.returncode == 0, r.stderr
    assert coord(c["bob"], "inbox").stdout.strip() == "inbox empty"

    answer = coord(c["alice"], "inbox").stdout
    assert "ANSWER from bob" in answer and f"re: {mid}" in answer
    assert "status: closed" in coord(c["alice"], "show", mid).stdout


def test_unknown_recipient_is_refused(team):
    _, c = team
    r = coord(c["alice"], "send", "bobb", "fyi", "typo")
    assert r.returncode == 1 and "unknown member" in r.stderr


def test_broadcast_reaches_others_but_not_the_sender(team):
    _, c = team
    coord(c["alice"], "send", "all", "fyi", "shards rebuilt")
    assert "shards rebuilt" in coord(c["bob"], "inbox").stdout
    assert coord(c["alice"], "inbox").stdout.strip() == "inbox empty"


def test_claims_exclude_each_other(team):
    _, c = team
    assert coord(c["alice"], "claim", "path:c3/regression.py", "--note", "Eq. 3").returncode == 0
    r = coord(c["bob"], "claim", "path:c3/regression.py")
    assert r.returncode == 3 and "claimed by alice" in r.stderr
    assert coord(c["bob"], "release", "path:c3/regression.py").returncode == 1
    assert coord(c["alice"], "release", "path:c3/regression.py").returncode == 0
    assert coord(c["bob"], "claim", "path:c3/regression.py").returncode == 0


def test_stale_claim_needs_takeover_and_tells_the_old_owner(team):
    _, c = team
    coord(c["alice"], "claim", "run:c1-llada", "--hours", "0")
    r = coord(c["bob"], "claim", "run:c1-llada")
    assert r.returncode == 3 and "--takeover" in r.stderr
    assert coord(c["bob"], "claim", "run:c1-llada", "--takeover").returncode == 0
    assert "took over stale claim run:c1-llada" in coord(c["alice"], "inbox").stdout


def test_session_start_hook_shows_inbox_board_and_claims(team):
    _, c = team
    coord(c["alice"], "send", "bob", "blocker", "shards missing")
    coord(c["bob"], "claim", "artifact:traces-full")
    coord(c["bob"], "status", "collecting traces")
    out = coord(c["bob"], "hook", "session-start").stdout
    assert "team member 'bob'" in out
    assert "BLOCKER from alice" in out
    assert "artifact:traces-full -> bob" in out
    assert "collecting traces" in out


def test_prompt_hook_reports_new_messages_once_per_interval(team):
    _, c = team
    fast = {"C3_COORD_INTERVAL": "0"}
    coord(c["bob"], "hook", "session-start")
    coord(c["alice"], "send", "bob", "fyi", "new figure")
    first = coord(c["bob"], "hook", "prompt", env=fast).stdout
    assert "new figure" in first
    assert coord(c["bob"], "hook", "prompt", env=fast).stdout == ""      # already seen
    coord(c["alice"], "send", "bob", "fyi", "second")
    assert coord(c["bob"], "hook", "prompt").stdout == ""                # throttled


def test_hooks_without_a_member_name_never_fail(tmp_path):
    sh("git", "init", "-q", str(tmp_path), cwd=tmp_path)
    r = coord(tmp_path, "hook", "session-start")
    assert r.returncode == 0 and "whoami --set" in r.stdout
    assert coord(tmp_path, "hook", "prompt").returncode == 0


@pytest.mark.parametrize("command,blocked", [
    ("git push origin main", True),
    ("rtk git push origin HEAD:main", True),
    ("git push -u origin alice/c2-probe", False),
    ("git push origin alice/x:refs/heads/main", True),
    ("git push --all origin", True),
    ("git status && git push", False),        # current branch is a topic branch
    ("echo push main", False),
])
def test_push_guard(team, command, blocked):
    _, c = team
    sh("git", "switch", "-qc", "alice/topic", cwd=c["alice"])
    payload = json.dumps({"tool_input": {"command": command}, "cwd": str(c["alice"])})
    out = coord(c["alice"], "hook", "guard-push", stdin=payload).stdout
    assert ('"deny"' in out) is blocked


def test_push_guard_blocks_bare_push_from_main(team):
    _, c = team
    payload = json.dumps({"tool_input": {"command": "git push"}, "cwd": str(c["alice"])})
    assert '"deny"' in coord(c["alice"], "hook", "guard-push", stdin=payload).stdout
