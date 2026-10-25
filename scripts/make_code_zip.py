#!/usr/bin/env python3
"""Build the course's code-submission ZIP from a committed ref.

The archive comes from ``git archive``, so uncommitted and ignored files never get
in, and everything sits under one top-level folder, ``c3/``.  Files that are not
code or small documents a grader needs are left out, each with a reason:

* anything over ``--max-file-mb`` (default 5);
* model weights: ``*.pt``, ``*.safetensors``, ``*.gguf`` (small parquet and JSON
  artifacts stay, since the analysis scripts read them);
* ``.claude/``, ``.github/`` and any ``coord-bus`` path (the bus lives on a branch,
  so it should not be in the archive at all; this is a second check);
* symlinks and other non-regular files.

``c3/MANIFEST.txt`` records the ref, its commit, the ``c3.provenance`` stamp, and the
kept and excluded files.  The exit status is 1 if the ZIP is over ``--max-mb``
(default 50); the ZIP is then deleted rather than left to be submitted by accident.

    python scripts/make_code_zip.py --ref HEAD --out dist/c3-code.zip
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c3.provenance import collect as provenance

TOP = "c3"
BAD_SUFFIXES = (".pt", ".safetensors", ".gguf")
BAD_DIRS = (".claude", ".github")


def exclusion_reason(path: str, size: int, max_bytes: int) -> str | None:
    """Why a repo-relative path stays out of the ZIP, or None if it goes in."""
    parts = path.split("/")
    for d in BAD_DIRS:
        if parts[0] == d:
            return f"{d}/ is not part of the submission"
    if any("coord-bus" in p for p in parts):
        return "coord-bus material"
    if path.lower().endswith(BAD_SUFFIXES):
        return f"{Path(path).suffix.lower()} model weights"
    if size > max_bytes:
        return f"over size limit ({size / 2**20:.1f} MB > {max_bytes / 2**20:g} MB)"
    return None


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def read_archive(repo: Path, commit: str, max_bytes: int):
    """Stream ``git archive`` and split it into kept (path, info, bytes) and excluded."""
    kept, excluded = [], []
    proc = subprocess.Popen(["git", "-C", str(repo), "archive", "--format=tar", commit],
                            stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for m in tar:
            if m.isdir():
                continue
            if not m.isreg():
                excluded.append((m.name, m.size, "symlink or special file"))
                continue
            why = exclusion_reason(m.name, m.size, max_bytes)
            if why:
                excluded.append((m.name, m.size, why))
            else:
                kept.append((m.name, m, tar.extractfile(m).read()))
    if proc.wait():
        raise RuntimeError(f"git archive {commit} failed")
    return sorted(kept, key=lambda k: k[0]), sorted(excluded)


def manifest_text(ref: str, commit: str, kept, excluded) -> str:
    stamp = provenance(extra={"git_commit": commit, "git_dirty": False, "ref": ref})
    lines = [f"ref: {ref}", f"commit: {commit}", "provenance: " + json.dumps(stamp, sort_keys=True),
             "", f"== kept ({len(kept)} files, plus this manifest) =="]
    lines += [f"{len(data):>10}  {name}" for name, _, data in kept]
    lines += ["", f"== excluded ({len(excluded)} files) =="]
    lines += [f"{size:>10}  {name}  [{why}]" for name, size, why in excluded]
    return "\n".join(lines) + "\n"


def write_zip(out: Path, kept, manifest: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, m, data in kept:
            # The commit time as the timestamp keeps the ZIP reproducible.
            stamp = max((1980, 1, 1, 0, 0, 0), tuple(time.gmtime(m.mtime))[:6])
            info = zipfile.ZipInfo(f"{TOP}/{name}", date_time=stamp)
            info.external_attr = (m.mode & 0o777) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
        z.writestr(f"{TOP}/MANIFEST.txt", manifest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="HEAD", help="committed ref to archive")
    ap.add_argument("--out", default="dist/c3-code.zip")
    ap.add_argument("--repo", default=str(ROOT), help="git repository (default: this one)")
    ap.add_argument("--max-file-mb", type=float, default=5.0, help="exclude files above this")
    ap.add_argument("--max-mb", type=float, default=50.0, help="fail if the ZIP is above this")
    a = ap.parse_args(argv)

    repo, out = Path(a.repo), Path(a.out)
    commit = git(repo, "rev-parse", "--verify", f"{a.ref}^{{commit}}")
    kept, excluded = read_archive(repo, commit, int(a.max_file_mb * 2**20))
    write_zip(out, kept, manifest_text(a.ref, commit, kept, excluded))

    raw = sum(len(d) for _, _, d in kept)
    packed = out.stat().st_size
    print(f"{a.ref} = {commit[:12]}  ->  {out}")
    print(f"{len(kept) + 1} files (incl. MANIFEST.txt), {raw / 2**20:.2f} MB uncompressed, "
          f"{packed / 2**20:.2f} MB compressed")
    print("largest kept:")
    for name, _, data in sorted(kept, key=lambda k: -len(k[2]))[:10]:
        print(f"  {len(data) / 2**10:>9.1f} KB  {name}")
    print(f"excluded ({len(excluded)}):")
    for name, size, why in excluded:
        print(f"  {size / 2**10:>9.1f} KB  {name}  [{why}]")

    if packed > a.max_mb * 2**20:
        out.unlink()
        print(f"ERROR: ZIP is {packed / 2**20:.1f} MB, over --max-mb {a.max_mb:g}; deleted {out}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
