#!/usr/bin/env python3
"""A fake `jl` (jarvislabs CLI) for tests/test_jl_t12.py: state in $FAKE_JL_DIR, every call logged
to calls.log (arguments only; the API key is never written anywhere by this script).

State files (JSON): instances.json, account.json ({"balance", "grants", "currency"}),
gpus.json, scenario.json ({"run": "ok" | "drop" | "fail", "polls_running": N, "no_armed": bool,
"create_fails": bool, "new_id_on_resume": int | null, "download_wait_touches": N}).

Several fake `jl` processes can run at once (the script's hold loop touches /home/.t12-hold in
the background while the main flow goes on, and a touch that was killed mid-flight may finish
late), so every read-modify-write of a state file runs under an exclusive lock: without it a
slow touch could write back a stale fails.json and "un-count" simulated failures.
"""
import hashlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:          # Windows (Git Bash runs these tests): msvcrt byte-range lock instead
    fcntl = None
    import msvcrt

D = Path(os.environ["FAKE_JL_DIR"])


@contextmanager
def locked():
    """An exclusive lock on $FAKE_JL_DIR/.lock, held for the whole read-modify-write."""
    with (D / ".lock").open("a+") as fh:
        if fcntl:
            fcntl.flock(fh, fcntl.LOCK_EX)
        else:
            while True:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
        try:
            yield
        finally:
            if fcntl:
                fcntl.flock(fh, fcntl.LOCK_UN)
            else:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


def load(name, default=None):
    f = D / name
    return json.loads(f.read_text()) if f.exists() else default


def save(name, obj):
    (D / name).write_text(json.dumps(obj, indent=1))


def log(line):
    with (D / "calls.log").open("a") as fh:
        fh.write(f"{time.time():.3f} {line}\n")


def out(obj):
    print(json.dumps(obj))


def instances():
    return load("instances.json", [])


def find(mid):
    for i in instances():
        if str(i["machine_id"]) == str(mid):
            return i
    return None


def setstatus(mid, status, new_id=None, num_gpus=None):
    with locked():
        ins = instances()
        for i in ins:
            if str(i["machine_id"]) == str(mid):
                i["status"] = status
                if new_id:
                    i["machine_id"] = new_id
                if num_gpus:
                    i["num_gpus"] = num_gpus
        save("instances.json", ins)


def take_failure(key, scenario):
    """True if this call of `key` is one of the simulated outage's failures (and count it)."""
    with locked():
        fails = load("fails.json", dict(scenario.get("fail", {})))
        if fails.get(key, 0) <= 0:
            return False
        fails[key] -= 1
        save("fails.json", fails)
        return True


def hold_touches():
    f = D / "calls.log"
    lines = f.read_text().splitlines() if f.exists() else []
    return sum(1 for ln in lines if " exec " in ln and "t12-hold" in ln)


def main(argv):
    log(" ".join(argv))
    sc = load("scenario.json", {})
    cmd = argv[0]
    rest = argv[1:]
    # scenario["fail"] = {"list": 3, "status": 2, "run list": 1, "upload": 3}: that many failing calls
    # of the verb (an outage), counted in fails.json, then it works again
    key = "run " + rest[0] if cmd == "run" and rest else cmd
    if take_failure(key, sc):
        log("FAILED " + key)
        print("fake jl: simulated outage", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("JL_API_KEY") in (None, ""):
        print("fake jl: no JL_API_KEY in the environment", file=sys.stderr)
    if cmd == "status":
        acc = load("account.json")
        out({"user": {}, "balance": {"balance": acc["balance"], "grants": acc["grants"]},
             "resources": {}, "currency": acc.get("currency", "INR"), "cli": {}})
    elif cmd == "gpus":
        out(load("gpus.json"))
    elif cmd == "ssh-key":
        out({"ssh_keys": [{"id": 1}]})
    elif cmd == "list":
        out(instances())
    elif cmd == "create":
        if sc.get("create_fails"):
            print('{"error": "no capacity"}')
            sys.exit(1)
        a = {rest[i]: rest[i + 1] for i in range(len(rest) - 1) if rest[i].startswith("--")}
        mid = 900 + len(instances()) + 1
        ins = instances() + [{"machine_id": mid, "name": a["--name"], "status": "Running",
                              "num_gpus": int(a["--num-gpus"]), "gpu_type": a["--gpu"],
                              "region": a["--region"]}]
        save("instances.json", ins)
        out({"machine_id": mid, "name": a["--name"], "status": "Running"})
    elif cmd == "upload":
        mid, src, dest = rest[0], rest[1], rest[2]
        if not Path(src).exists():
            sys.exit(f"fake jl: {src} does not exist")
        data = Path(src).read_bytes()
        (D / ("up-" + dest.strip("/").replace("/", "_"))).write_bytes(data)
        (D / "up-secret-src.txt").write_text(src) if dest.endswith(".t12.env") else None
    elif cmd == "exec":
        j = rest.index("--")
        mid, c = rest[0], " ".join(rest[j + 1:])
        if "setsid" in c:
            m = re.search(r"T12_WD_NONCE=(\S+?);", c)
            print("" if sc.get("no_armed") else f"2026-10-26T00:00:00Z [watchdog 1 fetch] armed: x; nonce {m.group(1)}; y")
        elif "t12-hold" in c:
            pass
        elif "ls -1" in c:
            print("/home/ANLP-Project/runs/t12-out/" + (load("scenario.json", {}).get("phase", "smoke")) + ".tgz")
        elif ".sha256" in c:
            f = D / "outputs.tgz"
            print(hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "")
        elif ".job-" in c:   # the job's end marker (scenario "job_running": not written yet)
            print("" if sc.get("job_running") else "T12_JOB_ENDED")
        elif ".deadline-" in c:
            if "budget_left" in sc:
                print(sc["budget_left"])
        else:
            print("ok")
    elif cmd == "download":
        mid, remote, local = rest[0], rest[1], rest[2]
        time.sleep(sc.get("download_sleep", 0))
        # "download_wait_touches": N: a slow download that ends only once N hold touches were
        # logged (event, not a fixed sleep); gives up after 120 s so a broken hold loop fails the
        # test instead of hanging it
        deadline = time.time() + 120
        while hold_touches() < sc.get("download_wait_touches", 0) and time.time() < deadline:
            time.sleep(0.1)
        Path(local).write_bytes((D / "outputs.tgz").read_bytes())
    elif cmd == "run":
        sub = rest[0]
        if sub == "list":
            out([{"run_id": "r1"}] if (D / "run-started").exists() else [])
        elif sub == "status":
            with locked():
                n = load("polls.json", {"n": 0})
                n["n"] += 1
                save("polls.json", n)
            if sc.get("run_state"):            # what real jl prints when users/fetch fails
                out({"state": sc["run_state"]})
            elif sc.get("run") == "fail":
                out({"state": "failed"})
            else:
                out({"state": "running" if n["n"] <= sc.get("polls_running", 0) else "succeeded"})
        elif sub == "logs":
            print("job log")
        else:
            (D / "run-started").write_text("1")
            if "--on" not in rest:
                sys.exit("fake jl: this launcher always runs --on")
            if sc.get("run") == "fail":
                sys.exit(3)
    elif cmd == "pause":
        setstatus(rest[0], "Paused")
        print("paused")
    elif cmd == "resume":
        a = {rest[i]: rest[i + 1] for i in range(len(rest) - 1) if rest[i].startswith("--")}
        setstatus(rest[0], "Running", new_id=sc.get("new_id_on_resume"),
                  num_gpus=int(a.get("--num-gpus", 0)) or None)
        print("resumed")
        if sc.get("resume_applies_then_fails") and not (D / "resume-failed-once").exists():
            (D / "resume-failed-once").write_text("1")      # worked on the server, timed out here
            sys.exit(1)
    elif cmd == "destroy":
        print("DESTROY CALLED", file=sys.stderr)
        sys.exit(9)
    else:
        sys.exit(f"fake jl: unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
