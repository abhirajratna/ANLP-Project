#!/usr/bin/env python3
"""A fake `jl` (jarvislabs CLI) for tests/test_jl_t12.py: state in $FAKE_JL_DIR, every call logged
to calls.log (arguments only; the API key is never written anywhere by this script).

State files (JSON): instances.json, account.json ({"balance", "grants", "currency"}),
gpus.json, scenario.json ({"run": "ok" | "drop" | "fail", "polls_running": N, "no_armed": bool,
"create_fails": bool, "new_id_on_resume": int | null}).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

D = Path(os.environ["FAKE_JL_DIR"])


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
    ins = instances()
    for i in ins:
        if str(i["machine_id"]) == str(mid):
            i["status"] = status
            if new_id:
                i["machine_id"] = new_id
            if num_gpus:
                i["num_gpus"] = num_gpus
    save("instances.json", ins)


def main(argv):
    log(" ".join(argv))
    sc = load("scenario.json", {})
    cmd = argv[0]
    rest = argv[1:]
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
            print("" if sc.get("no_armed") else "2026-10-26T00:00:00Z [watchdog 1 fetch] armed: x")
        elif "t12-hold" in c:
            pass
        elif "ls -1" in c:
            print("/home/ANLP-Project/runs/t12-out/" + (load("scenario.json", {}).get("phase", "smoke")) + ".tgz")
        elif ".sha256" in c:
            f = D / "outputs.tgz"
            print(hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "")
        elif ".deadline-" in c:
            if "budget_left" in sc:
                print(sc["budget_left"])
        else:
            print("ok")
    elif cmd == "download":
        mid, remote, local = rest[0], rest[1], rest[2]
        Path(local).write_bytes((D / "outputs.tgz").read_bytes())
    elif cmd == "run":
        sub = rest[0]
        if sub == "list":
            out([{"run_id": "r1"}] if (D / "run-started").exists() else [])
        elif sub == "status":
            n = load("polls.json", {"n": 0})
            n["n"] += 1
            save("polls.json", n)
            if sc.get("run") == "fail":
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
    elif cmd == "destroy":
        print("DESTROY CALLED", file=sys.stderr)
        sys.exit(9)
    else:
        sys.exit(f"fake jl: unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
