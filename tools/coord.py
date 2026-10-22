#!/usr/bin/env python3
"""Team coordination for four people and their agents, over a git branch.

Messages, claims and status live on a dedicated branch (`coord-bus`) of the
project remote, checked out in a private worktree under .git/. One file per
message, claim or status, so concurrent writers never touch the same path --
except two claims on the same resource, which is exactly the conflict a claim
is supposed to detect.

Stdlib only, so it runs from the Claude Code hooks before any venv exists.

    python3 tools/coord.py whoami --set <name>          # once per clone
    python3 tools/coord.py send <to> <type> "<subject>" --body "..."
    python3 tools/coord.py inbox
    python3 tools/coord.py reply <id> "..." [--close]
    python3 tools/coord.py claim <resource> [--hours 24] [--note "..."]
    python3 tools/coord.py board
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BRANCH = "coord-bus"
REMOTE = "origin"
TYPES = ("request", "handoff", "blocker", "fyi", "answer")
NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,23}$")
PROMPT_INTERVAL = int(os.environ.get("C3_COORD_INTERVAL", "600"))  # seconds
PROTECTED = ("main", "master")


class CoordError(Exception):
    pass


class Conflict(CoordError):
    pass


# --------------------------------------------------------------------- git

def _env() -> dict:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    if "GIT_SSH_COMMAND" not in env:
        env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes -o ConnectTimeout=8"
    return env


def git(*args, cwd=None, check=True, timeout=60) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, env=_env(), timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and r.returncode != 0:
        raise CoordError(f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def git_ok(*args, cwd=None, timeout=60) -> bool:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, env=_env(), timeout=timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def common_dir() -> Path:
    d = Path(git("rev-parse", "--git-common-dir"))
    return d if d.is_absolute() else (Path.cwd() / d).resolve()


def worktree() -> Path:
    return common_dir() / BRANCH


def state_path() -> Path:
    return common_dir() / "coord-state.json"


def load_state() -> dict:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"seen": [], "last_check": 0.0}


def save_state(state: dict) -> None:
    state["seen"] = state.get("seen", [])[-2000:]
    state_path().write_text(json.dumps(state), encoding="utf-8")


class Lock:
    """Serialises coord operations from several sessions in one clone."""

    def __init__(self, wait=30.0, stale=120.0):
        self.path = common_dir() / "coord.lock"
        self.wait, self.stale = wait, stale

    def __enter__(self):
        deadline = time.time() + self.wait
        while True:
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink()
                        continue
                except OSError:
                    continue
                if time.time() > deadline:
                    raise CoordError(f"coord is locked by another session ({self.path})")
                time.sleep(0.2)

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------- the bus

def ensure_worktree(timeout=60) -> Path:
    wt = worktree()
    if (wt / ".git").exists():
        return wt
    if wt.exists():  # left over from a failed bootstrap
        git("worktree", "prune")
        if (wt / ".git").exists():
            return wt
        raise CoordError(f"{wt} exists but is not a worktree; remove it and retry")
    if git_ok("fetch", "-q", REMOTE, BRANCH, timeout=timeout):
        git("worktree", "add", "-q", "-B", BRANCH, str(wt), f"{REMOTE}/{BRANCH}")
        return wt
    # The branch does not exist yet: the first member to run this creates it.
    git("worktree", "add", "-q", "--detach", str(wt))
    git("checkout", "-q", "--orphan", BRANCH, cwd=wt)
    git("rm", "-rfq", "--ignore-unmatch", ".", cwd=wt)
    for p in wt.iterdir():
        if p.name != ".git":
            _rmtree(p)
    (wt / "README.md").write_text(
        "# coord-bus\n\nTeam message bus, claims and status. Written by "
        "`tools/coord.py` on main; do not edit by hand.\n", encoding="utf-8")
    git("add", "README.md", cwd=wt)
    git("commit", "-q", "--no-verify", "-m", "coord: create bus", cwd=wt)
    if not git_ok("push", "-q", "-u", REMOTE, BRANCH, cwd=wt, timeout=timeout):
        # Someone else created it at the same moment; take theirs.
        git("fetch", "-q", REMOTE, BRANCH, cwd=wt, timeout=timeout)
        git("reset", "-q", "--hard", f"{REMOTE}/{BRANCH}", cwd=wt)
    git("branch", "-q", f"--set-upstream-to={REMOTE}/{BRANCH}", BRANCH, cwd=wt, check=False)
    return wt


def _rmtree(p: Path) -> None:
    if p.is_dir() and not p.is_symlink():
        for c in p.iterdir():
            _rmtree(c)
        p.rmdir()
    else:
        p.unlink()


def sync(timeout=60) -> bool:
    """Bring the bus up to date. Returns False when the remote is unreachable."""
    wt = ensure_worktree(timeout)
    if not git_ok("fetch", "-q", REMOTE, BRANCH, cwd=wt, timeout=timeout):
        return False
    if git_ok("rebase", "-q", f"{REMOTE}/{BRANCH}", cwd=wt):
        if git("rev-list", "--count", f"{REMOTE}/{BRANCH}..HEAD", cwd=wt) != "0":
            git_ok("push", "-q", REMOTE, f"HEAD:{BRANCH}", cwd=wt, timeout=timeout)
        return True
    git_ok("rebase", "--abort", cwd=wt)
    git("reset", "-q", "--hard", f"{REMOTE}/{BRANCH}", cwd=wt)
    return True


def publish(paths, message: str) -> None:
    wt = worktree()
    git("add", "-A", "--", *paths, cwd=wt)
    if git_ok("diff", "--cached", "--quiet", cwd=wt):
        return
    git("commit", "-q", "--no-verify", "-m", message, cwd=wt)
    for _ in range(6):
        if git_ok("push", "-q", REMOTE, f"HEAD:{BRANCH}", cwd=wt):
            return
        git("fetch", "-q", REMOTE, BRANCH, cwd=wt)
        if not git_ok("rebase", "-q", f"{REMOTE}/{BRANCH}", cwd=wt):
            git_ok("rebase", "--abort", cwd=wt)
            git("reset", "-q", "--hard", f"{REMOTE}/{BRANCH}", cwd=wt)
            raise Conflict("someone else changed the same file first; nothing was written")
    raise CoordError("could not push to the bus after 6 attempts (offline?)")


# ------------------------------------------------------------- documents

def now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def write_doc(path: Path, meta: dict, body: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = "\n".join(f"{k}: {v}" for k, v in meta.items() if v not in (None, ""))
    path.write_text(f"---\n{head}\n---\n{body.rstrip()}\n", encoding="utf-8")


def read_doc(path: Path):
    text = path.read_text(encoding="utf-8")
    meta, body = {}, text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
            body = text[end + 5:]
    return meta, body.strip()


def me(required=True):
    name = os.environ.get("C3_MEMBER") or git("config", "--get", "c3.member", check=False)
    if required and not name:
        raise CoordError("no member name for this clone: run "
                         "`python3 tools/coord.py whoami --set <name>`")
    return name or None


def members() -> list:
    d = worktree() / "members"
    return sorted(p.stem for p in d.glob("*.md")) if d.is_dir() else []


def messages(box: str):
    d = worktree() / "inbox" / box
    out = []
    for p in sorted(d.glob("*.md")) if d.is_dir() else []:
        meta, body = read_doc(p)
        out.append((p, meta, body))
    return out


def find_message(mid: str):
    for p in (worktree() / "inbox").glob(f"*/{mid}.md"):
        meta, body = read_doc(p)
        return p, meta, body
    raise CoordError(f"no message {mid}")


def my_open_messages(name: str):
    mine = [m for m in messages(name) if m[1].get("status") == "open"]
    mine += [m for m in messages("all")
             if m[1].get("status") == "open" and m[1].get("from") != name]
    return sorted(mine, key=lambda m: m[1].get("id", ""))


def slug(resource: str) -> str:
    if not re.match(r"^[A-Za-z0-9._:/-]{1,120}$", resource):
        raise CoordError("resource names may use letters, digits and . _ : / -")
    return re.sub(r"[^A-Za-z0-9._-]", "_", resource)


def claims():
    d = worktree() / "claims"
    out = []
    for p in sorted(d.glob("*.md")) if d.is_dir() else []:
        meta, body = read_doc(p)
        meta["stale"] = parse_iso(meta["expires"]) <= now()
        out.append((p, meta, body))
    return out


def fmt_message(meta: dict, body: str, full=True) -> str:
    line = (f"[{meta.get('id')}] {meta.get('type', '?').upper()} from {meta.get('from')} "
            f"to {meta.get('to')}: {meta.get('subject', '')}")
    extra = [f"{k}: {meta[k]}" for k in ("re", "needs_by", "status", "closed_by")
             if meta.get(k) and not (k == "status" and meta[k] == "open")]
    if extra:
        line += f"  ({'; '.join(extra)})"
    if full and body:
        line += "\n" + "\n".join("    " + l for l in body.splitlines())
    return line


def fmt_claim(meta: dict) -> str:
    s = (f"{meta['resource']} -> {meta['owner']} until {meta['expires']}"
         + (" [STALE]" if meta["stale"] else ""))
    return s + (f" ({meta['note']})" if meta.get("note") else "")


# -------------------------------------------------------------- commands

def cmd_whoami(a):
    if not a.set:
        print(me(required=False) or "(not set)")
        return
    if not NAME_RE.match(a.set):
        raise CoordError("names are lowercase letters, digits and '-', 2-24 chars, e.g. 'ojas'")
    if a.set == "all":
        raise CoordError("'all' is reserved for broadcasts")
    git("config", "--local", "c3.member", a.set)
    sync()
    p = worktree() / "members" / f"{a.set}.md"
    if not p.exists():
        write_doc(p, {"name": a.set, "git_user": git("config", "user.name", check=False),
                      "joined": iso(now())})
        publish([p], f"coord: {a.set} joins")
    print(f"this clone now acts as '{a.set}'. members: {', '.join(members())}")


def cmd_members(a):
    sync()
    print("\n".join(members()) or "(nobody has joined yet)")


def cmd_send(a):
    sender = me()
    if a.type not in TYPES:
        raise CoordError(f"type must be one of {', '.join(TYPES)}")
    body = a.body if a.body is not None else (
        Path(a.body_file).read_text(encoding="utf-8") if a.body_file else "")
    sync()
    targets = [t.strip() for t in a.to.split(",") if t.strip()]
    known = set(members())
    unknown = [t for t in targets if t != "all" and t not in known]
    if unknown:
        raise CoordError(f"unknown member(s) {', '.join(unknown)}; members: {', '.join(sorted(known))}")
    paths, ids = [], []
    for t in targets:
        mid = f"{now().strftime('%Y%m%dT%H%M%SZ')}-{sender}-{secrets.token_hex(2)}"
        p = worktree() / "inbox" / t / f"{mid}.md"
        write_doc(p, {"id": mid, "from": sender, "to": t, "type": a.type,
                      "subject": a.subject.replace("\n", " "), "re": a.re,
                      "needs_by": a.needs_by, "created": iso(now()), "status": "open"}, body)
        paths.append(p)
        ids.append(mid)
    publish(paths, f"coord: {sender} -> {a.to} {a.type}: {a.subject[:60]}")
    print("sent " + ", ".join(ids))


def cmd_inbox(a):
    name = me()
    online = sync()
    if a.all:
        msgs = messages(name) + [m for m in messages("all") if m[1].get("from") != name]
    else:
        msgs = my_open_messages(name)
    if not online:
        print("(remote unreachable: showing the last synced state)")
    if not msgs:
        print("inbox empty" if not a.all else "no messages")
    for _, meta, body in msgs:
        print(fmt_message(meta, body, full=not a.brief))
    state = load_state()
    state["seen"] = sorted(set(state.get("seen", [])) | {m[1]["id"] for m in msgs})
    save_state(state)


def cmd_sent(a):
    name = me()
    sync()
    out = [m for box in (worktree() / "inbox").glob("*") for m in messages(box.name)
           if m[1].get("from") == name]
    for _, meta, body in sorted(out, key=lambda m: m[1]["id"])[-a.limit:]:
        print(fmt_message(meta, body, full=False))


def cmd_show(a):
    sync()
    _, meta, body = find_message(a.id)
    print(fmt_message(meta, body))


def _close(p: Path, meta: dict, body: str, by: str, note: str = "") -> None:
    meta.update(status="closed", closed_by=by, closed_at=iso(now()))
    if note:
        body = f"{body}\n\n[closed by {by}] {note}"
    write_doc(p, meta, body)


def cmd_reply(a):
    name = me()
    sync()
    p, meta, body = find_message(a.id)
    mid = f"{now().strftime('%Y%m%dT%H%M%SZ')}-{name}-{secrets.token_hex(2)}"
    rp = worktree() / "inbox" / meta["from"] / f"{mid}.md"
    subject = meta.get("subject", "")
    write_doc(rp, {"id": mid, "from": name, "to": meta["from"], "type": a.type,
                   "subject": subject if subject.startswith("Re: ") else f"Re: {subject}",
                   "re": a.id, "created": iso(now()), "status": "open"}, a.body)
    paths = [rp]
    if a.close:
        _close(p, meta, body, name)
        paths.append(p)
    publish(paths, f"coord: {name} replies to {a.id}")
    print(f"sent {mid}" + (f"; closed {a.id}" if a.close else ""))


def cmd_close(a):
    name = me()
    sync()
    for mid in a.ids:
        p, meta, body = find_message(mid)
        _close(p, meta, body, name, a.note)
        publish([p], f"coord: {name} closes {mid}")
        print(f"closed {mid}")


def cmd_claim(a):
    name = me()
    sync()
    p = worktree() / "claims" / f"{slug(a.resource)}.md"
    old = None
    if p.exists():
        old, _ = read_doc(p)
        stale = parse_iso(old["expires"]) <= now()
        if old["owner"] != name and not stale:
            raise Conflict(f"{a.resource} is claimed by {old['owner']} until {old['expires']}"
                           + (f" ({old['note']})" if old.get("note") else "")
                           + f". Message them: coord.py send {old['owner']} request ...")
        if old["owner"] != name and not a.takeover:
            raise Conflict(f"{a.resource} has a stale claim by {old['owner']} "
                           f"(expired {old['expires']}); re-run with --takeover to take it")
    t = now()
    meta = {"resource": a.resource, "owner": name,
            "since": old["since"] if old and old["owner"] == name else iso(t),
            "updated": iso(t), "expires": iso(t + timedelta(hours=a.hours)),
            "note": a.note}
    if not a.note and old and old["owner"] == name:
        meta["note"] = old.get("note", "")  # a refresh keeps the note
    write_doc(p, meta)
    paths = [p]
    if old and old["owner"] != name:
        mid = f"{t.strftime('%Y%m%dT%H%M%SZ')}-{name}-{secrets.token_hex(2)}"
        fp = worktree() / "inbox" / old["owner"] / f"{mid}.md"
        write_doc(fp, {"id": mid, "from": name, "to": old["owner"], "type": "fyi",
                       "subject": f"took over stale claim {a.resource}",
                       "created": iso(t), "status": "open"},
                  f"Your claim expired at {old['expires']}. {name} holds it now.")
        paths.append(fp)
    publish(paths, f"coord: {name} claims {a.resource}")
    print(f"claimed {a.resource} until {meta['expires']}")


def cmd_release(a):
    name = me()
    sync()
    for r in a.resources:
        p = worktree() / "claims" / f"{slug(r)}.md"
        if not p.exists():
            print(f"{r} was not claimed")
            continue
        meta, _ = read_doc(p)
        if meta["owner"] != name and not a.force:
            raise CoordError(f"{r} belongs to {meta['owner']}; use --force only if they agreed")
        p.unlink()
        publish([p], f"coord: {name} releases {r}")
        print(f"released {r}")


def cmd_claims(a):
    name = me(required=False)
    sync()
    rows = [m for _, m, _ in claims() if not a.mine or m["owner"] == name]
    print("\n".join(fmt_claim(m) for m in rows) or "no claims")


def cmd_status(a):
    name = me()
    sync()
    p = worktree() / "status" / f"{name}.md"
    branch = git("rev-parse", "--abbrev-ref", "HEAD", check=False)
    write_doc(p, {"member": name, "updated": iso(now()), "branch": branch}, a.text)
    publish([p], f"coord: {name} status")
    print("status updated")


def board_text(name) -> str:
    lines = ["## Team board"]
    for m in members():
        p = worktree() / "status" / f"{m}.md"
        if p.exists():
            meta, body = read_doc(p)
            lines.append(f"- {m} ({meta.get('updated')}, branch {meta.get('branch')}): {body}")
        else:
            lines.append(f"- {m}: (no status yet)")
        n = len([x for x in messages(m) if x[1].get("status") == "open"])
        if n:
            lines[-1] += f"  [{n} open message(s)]"
    try:
        wss = load_workstreams()
    except CoordError:
        wss = {}
    owned = [f"{n}: {ws_owner(n) or 'unowned'}" for n, ws in wss.items() if not ws.get("shared")]
    if owned:
        lines.append("## Workstreams")
        lines.append("- " + "; ".join(owned))
    cl = claims()
    lines.append("## Claims")
    lines += [f"- {fmt_claim(m)}" for _, m, _ in cl] or ["- none"]
    return "\n".join(lines)


def load_workstreams() -> dict:
    p = Path(__file__).resolve().parent / "workstreams.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CoordError(f"cannot read {p}: {e}")
    return {k: v for k, v in data.items() if not k.startswith("_")}


def ws_owner(ws: str):
    p = worktree() / "owners" / f"{ws}.md"
    return read_doc(p)[0].get("owner") if p.exists() else None


def norm(path: str) -> str:
    return path[2:] if path.startswith("./") else path


def matches(path: str, pattern: str) -> bool:
    path = norm(path)
    return path.startswith(pattern) if pattern.endswith("/") else path == pattern


def cmd_workstreams(a):
    sync()
    for name, ws in load_workstreams().items():
        who = "shared (PR + one approval)" if ws.get("shared") else (ws_owner(name) or "unowned")
        print(f"{name}: {ws['title']}  [{who}]")
        if a.paths:
            print("    " + "  ".join(ws["paths"]))


def cmd_own(a):
    name = me()
    wss = load_workstreams()
    if a.workstream not in wss:
        raise CoordError(f"no workstream {a.workstream}; have: {', '.join(wss)}")
    if wss[a.workstream].get("shared"):
        raise CoordError(f"{a.workstream} is shared and cannot be owned")
    sync()
    p = worktree() / "owners" / f"{a.workstream}.md"
    if a.release:
        if ws_owner(a.workstream) != name:
            raise CoordError(f"you do not own {a.workstream}")
        p.unlink()
        publish([p], f"coord: {name} gives up {a.workstream}")
        print(f"{a.workstream} is unowned now")
        return
    current = ws_owner(a.workstream)
    if current and current != name:
        raise Conflict(f"{a.workstream} is owned by {current}; ask them to run "
                       f"`coord.py own {a.workstream} --release` first")
    write_doc(p, {"workstream": a.workstream, "owner": name, "since": iso(now())})
    publish([p], f"coord: {name} owns {a.workstream}")
    print(f"you own {a.workstream}: {', '.join(wss[a.workstream]['paths'])}")


def cmd_owner(a):
    name = me(required=False)
    sync()
    held = {m["resource"]: m for _, m, _ in claims()}
    for path in a.paths:
        hits = [(n, ws) for n, ws in load_workstreams().items()
                if any(matches(path, pat) for pat in ws["paths"])]
        if not hits:
            line = f"{path}: no workstream (free to edit)"
        else:
            parts = []
            for n, ws in hits:
                if ws.get("shared"):
                    parts.append(f"{n} (shared: PR + one other member's approval)")
                else:
                    o = ws_owner(n)
                    parts.append(f"{n} owned by {o}" + (" (you)" if o and o == name else "")
                                 if o else f"{n} (unowned)")
            line = f"{path}: " + "; ".join(parts)
        c = held.get(f"path:{norm(path)}")
        if c:
            line += f"; CLAIMED by {c['owner']} until {c['expires']}"
        print(line)


def cmd_board(a):
    sync()
    print(board_text(me(required=False)))


# ----------------------------------------------------------------- hooks

def hook_session_start() -> str:
    name = me(required=False)
    if not name:
        return ("[coord] This clone has no team member name, so the team inbox is off. "
                "Ask the user which member they are, then run "
                "`python3 tools/coord.py whoami --set <name>` (see CLAUDE.md).")
    online = sync(timeout=15)
    out = [f"[coord] You are working as team member '{name}'."
           + ("" if online else " (Remote unreachable: this is the last synced state.)")]
    msgs = my_open_messages(name)
    if msgs:
        out.append(f"## Open messages for you ({len(msgs)}); handle or acknowledge them")
        out += [fmt_message(m, b) for _, m, b in msgs]
    else:
        out.append("No open messages for you.")
    mine = [m for _, m, _ in claims() if m["owner"] == name]
    if mine:
        out.append("## Your claims (release what you are done with)")
        out += [f"- {fmt_claim(m)}" for m in mine]
    out.append(board_text(name))
    state = load_state()
    state["seen"] = sorted(set(state.get("seen", [])) | {m["id"] for _, m, _ in msgs})
    state["last_check"] = time.time()
    save_state(state)
    return "\n".join(out)


def hook_prompt() -> str:
    name = me(required=False)
    if not name:
        return ""
    state = load_state()
    if time.time() - state.get("last_check", 0.0) < PROMPT_INTERVAL:
        return ""
    state["last_check"] = time.time()
    save_state(state)
    sync(timeout=10)
    seen = set(state.get("seen", []))
    new = [(m, b) for _, m, b in my_open_messages(name) if m["id"] not in seen]
    out = []
    if new:
        out.append(f"[coord] {len(new)} new team message(s) for '{name}':")
        out += [fmt_message(m, b) for m, b in new]
    expired = [m for _, m, _ in claims() if m["owner"] == name and m["stale"]]
    if expired:
        out.append("[coord] Your claim(s) expired; refresh with `claim` or release them: "
                   + ", ".join(m["resource"] for m in expired))
    state = load_state()
    state["seen"] = sorted(seen | {m["id"] for m, _ in new})
    save_state(state)
    return "\n".join(out)


def push_targets_protected(command: str, cwd) -> str:
    """Name the protected branch a `git push` command would update, or ''."""
    for seg in re.split(r"&&|\|\||;|\|", command):
        try:
            words = shlex.split(seg)
        except ValueError:
            words = seg.split()
        if "git" not in words or "push" not in words[words.index("git"):]:
            continue
        args = words[words.index("push", words.index("git")) + 1:]
        if any(x in ("--all", "--mirror") for x in args):
            return "every branch"
        pos = [x for x in args if not x.startswith("-")]
        for spec in pos[1:]:
            dst = spec.split(":")[-1].lstrip("+").replace("refs/heads/", "")
            if dst in PROTECTED or (dst == "HEAD" and _current_branch(cwd) in PROTECTED):
                return dst if dst != "HEAD" else _current_branch(cwd)
        if len(pos) <= 1 and _current_branch(cwd) in PROTECTED:
            return _current_branch(cwd)
    return ""


def _current_branch(cwd) -> str:
    try:
        return git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd, check=False, timeout=5)
    except Exception:
        return ""


def hook_guard_push(payload: dict) -> str:
    cmd = (payload.get("tool_input") or {}).get("command", "")
    if not re.search(r"\bgit\b.*\bpush\b", cmd):
        return ""
    target = push_targets_protected(cmd, payload.get("cwd") or None)
    if not target:
        return ""
    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny",
        "permissionDecisionReason": (
            f"Team rule (CLAUDE.md): agents never push to {target}. Push a "
            "<member>/<topic> branch and open a pull request instead.")}})


def cmd_hook(a):
    # Hooks must never break a session: any failure prints at most one line.
    try:
        if a.event == "session-start":
            with Lock(wait=10):
                out = hook_session_start()
        elif a.event == "prompt":
            with Lock(wait=2):
                out = hook_prompt()
        else:
            out = hook_guard_push(json.loads(sys.stdin.read() or "{}"))
        if out:
            print(out)
    except Exception as e:  # noqa: BLE001
        if a.event == "session-start":
            print(f"[coord] team inbox unavailable: {e}")


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="coord.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("whoami", help="show or set this clone's member name")
    s.add_argument("--set")
    s.set_defaults(fn=cmd_whoami)
    sub.add_parser("members", help="list members").set_defaults(fn=cmd_members)

    s = sub.add_parser("send", help="send a message")
    s.add_argument("to", help="member, comma list, or 'all'")
    s.add_argument("type", choices=TYPES)
    s.add_argument("subject")
    s.add_argument("--body")
    s.add_argument("--body-file")
    s.add_argument("--re", help="id of the message this follows up")
    s.add_argument("--needs-by", help="e.g. 2026-10-25")
    s.set_defaults(fn=cmd_send)

    s = sub.add_parser("inbox", help="open messages for you")
    s.add_argument("--all", action="store_true", help="include closed messages")
    s.add_argument("--brief", action="store_true", help="subjects only")
    s.set_defaults(fn=cmd_inbox)

    s = sub.add_parser("sent", help="messages you sent and their status")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(fn=cmd_sent)

    s = sub.add_parser("show", help="print one message")
    s.add_argument("id")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("reply", help="answer a message (goes to its sender)")
    s.add_argument("id")
    s.add_argument("body")
    s.add_argument("--type", default="answer", choices=TYPES)
    s.add_argument("--close", action="store_true", help="also close the original")
    s.set_defaults(fn=cmd_reply)

    s = sub.add_parser("close", help="mark messages done")
    s.add_argument("ids", nargs="+")
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_close)

    s = sub.add_parser("claim", help="claim a resource (path:, run:, artifact:, ...)")
    s.add_argument("resource")
    s.add_argument("--hours", type=float, default=24.0)
    s.add_argument("--note", default="")
    s.add_argument("--takeover", action="store_true", help="take over a stale claim")
    s.set_defaults(fn=cmd_claim)

    s = sub.add_parser("release", help="release claims")
    s.add_argument("resources", nargs="+")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_release)

    s = sub.add_parser("claims", help="list claims")
    s.add_argument("--mine", action="store_true")
    s.set_defaults(fn=cmd_claims)

    s = sub.add_parser("status", help="set your line on the board")
    s.add_argument("text")
    s.set_defaults(fn=cmd_status)
    sub.add_parser("board", help="everyone's status and all claims").set_defaults(fn=cmd_board)

    s = sub.add_parser("workstreams", help="areas of the repo and their owners")
    s.add_argument("--paths", action="store_true")
    s.set_defaults(fn=cmd_workstreams)

    s = sub.add_parser("own", help="take (or --release) ownership of a workstream")
    s.add_argument("workstream")
    s.add_argument("--release", action="store_true")
    s.set_defaults(fn=cmd_own)

    s = sub.add_parser("owner", help="who owns these paths, and are they claimed?")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=cmd_owner)

    s = sub.add_parser("hook", help="Claude Code hook entry points")
    s.add_argument("event", choices=("session-start", "prompt", "guard-push"))
    s.set_defaults(fn=cmd_hook)

    a = ap.parse_args(argv)
    try:
        if a.cmd == "hook":
            a.fn(a)
        else:
            with Lock():
                a.fn(a)
    except Conflict as e:
        print(f"conflict: {e}", file=sys.stderr)
        return 3
    except (CoordError, subprocess.TimeoutExpired) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
