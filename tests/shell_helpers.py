"""Helpers for the tests that run the T12 shell scripts against stubbed `jl` / `curl`.

A bash is looked for in this order: $T12_TEST_BASH, Git Bash (Windows: the WSL launcher in
System32 is not a bash these scripts can use), `bash` on PATH.  Tests skip when there is none.
Paths handed to bash are written with forward slashes (``C:/Users/...`` works in Git Bash).
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def find_bash() -> str | None:
    cand = [os.environ.get("T12_TEST_BASH")]
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                     os.environ.get("LOCALAPPDATA")):
            if base:
                cand += [os.path.join(base, "Git", "bin", "bash.exe"),
                         os.path.join(base, "Programs", "Git", "bin", "bash.exe")]
    else:
        cand.append(shutil.which("bash"))
    for c in cand:
        if c and os.path.exists(c):
            return c
    return None


BASH = find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash")


def posix(p) -> str:
    return Path(p).as_posix()


def stub(dirpath: Path, name: str, body: str) -> Path:
    """An executable script ``name`` in ``dirpath`` (LF line endings, bash shebang)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    f = dirpath / name
    f.write_bytes(("#!/bin/bash\n" + body.strip("\n") + "\n").replace("\r\n", "\n").encode())
    f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return f


def run_bash(script: str, env: dict | None = None, cwd=None, timeout: float = 120,
             stubs: Path | None = None) -> subprocess.CompletedProcess:
    """Run ``script`` with ``bash -c``; ``stubs`` goes first on PATH."""
    e = dict(os.environ)
    e.update(env or {})
    if stubs is not None:       # inside bash: a Windows-form PATH entry would not win on Git Bash
        script = (f'S="{posix(stubs)}"; command -v cygpath >/dev/null 2>&1 && S="$(cygpath -u "$S")"\n'
                  'export PATH="$S:$PATH"\n' + script)
    return subprocess.run([BASH, "-c", script], cwd=cwd, env=e, capture_output=True, text=True,
                          timeout=timeout)


def script_ok(name: str) -> None:
    """Skip when a checkout turned the script into CRLF (bash chokes on it)."""
    data = (ROOT / "scripts" / name).read_bytes()
    if b"\r\n" in data:
        pytest.skip(f"scripts/{name} has CRLF endings in this checkout (see jl_t12.sh header)")
