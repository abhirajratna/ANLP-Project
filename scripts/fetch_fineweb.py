#!/usr/bin/env python3
"""Download FineWeb-Edu sample-10BT once, verified, so no job has to stream it.

    https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT

A stream that stalls kills a multi-hour job, and downloads from the compute
nodes have stalled before.  So the 14 parquet files (28.5 GB) are fetched once:

  download  fetch every file (--first N: the first N) at a pinned revision into
            --dir, resuming partial downloads and retrying, check each against
            the Hub's size and sha256, and write --dir/MANIFEST.json last.  The
            manifest always lists all files; `check --first N` says how many are
            there.  Files already present and correct are kept, so a rerun only
            fetches what is missing.
  names     print the manifest's first N file names (for rsync --files-from).
  check     verify a copy against MANIFEST.json (sizes; --sha256 for hashes).

``c3.data.fineweb.fineweb_stream`` reads the copy when C3_FINEWEB_DIR points at it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

REPO = "HuggingFaceFW/fineweb-edu"
PREFIX = "sample/10BT"
# Pinned, so every download -- and so every shard build -- sees the same files.
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def is_ok(path: Path, f: dict, full: bool = True) -> bool:
    if not path.is_file() or path.stat().st_size != f["size"]:
        return False
    return not full or f["sha256"] is None or sha256(path) == f["sha256"]


def remote_files(repo: str, prefix: str, revision: str, attempts: int) -> list[dict]:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    for attempt in range(1, attempts + 1):
        try:
            entries = list(HfApi().list_repo_tree(repo, path_in_repo=prefix,
                                                  repo_type="dataset", revision=revision))
            break
        except Exception as exc:              # network errors come in many types
            if attempt == attempts:
                raise
            wait = min(300, 30 * attempt)
            print(f"[fetch] listing failed ({type(exc).__name__}: {exc}); retrying in {wait} s",
                  flush=True)
            time.sleep(wait)
    files = [{"name": Path(e.path).name, "path": e.path, "size": e.size,
              "sha256": e.lfs.sha256 if e.lfs else None}
             for e in entries if isinstance(e, RepoFile) and e.path.endswith(".parquet")]
    if not files:
        raise SystemExit(f"[fetch] FATAL: no parquet files under {repo}/{prefix}@{revision}")
    return sorted(files, key=lambda f: f["name"])


def download(a) -> int:
    from huggingface_hub import hf_hub_download

    dest = Path(a.dir)
    dest.mkdir(parents=True, exist_ok=True)
    man_path = dest / "MANIFEST.json"
    old = json.loads(man_path.read_text()) if man_path.exists() else None
    if old and (old["repo"], old["prefix"], old["revision"]) == (a.repo, a.prefix, a.revision):
        files = old["files"]                  # re-verifying a copy needs no listing
    else:
        man_path.unlink(missing_ok=True)
        files = remote_files(a.repo, a.prefix, a.revision, a.attempts)
    total = sum(f["size"] for f in files)
    want = files[:a.first] if getattr(a, "first", 0) else files
    print(f"[fetch] {a.repo}/{a.prefix}@{a.revision[:12]}: {len(files)} files, "
          f"{total / 1e9:.2f} GB; fetching {len(want)} -> {dest}", flush=True)

    # Downloads land here and move into --dir only once verified, so --dir never
    # holds a partial or corrupt file.  hf_hub_download resumes from here too.
    work = dest.parent / f"{dest.name}.download"
    for k, f in enumerate(want, 1):
        target = dest / f["name"]
        if is_ok(target, f):
            print(f"[fetch] {k}/{len(want)} {f['name']}: present, verified", flush=True)
            continue
        target.unlink(missing_ok=True)
        force = False
        for attempt in range(1, a.attempts + 1):
            t0 = time.time()
            try:
                got = Path(hf_hub_download(a.repo, f["path"], repo_type="dataset",
                                           revision=a.revision, local_dir=work,
                                           force_download=force))
            except Exception as exc:
                if attempt == a.attempts:
                    break
                wait = min(300, 30 * attempt)
                print(f"[fetch] {f['name']}: {type(exc).__name__}: {exc}; retry "
                      f"{attempt}/{a.attempts - 1} in {wait} s (partial download kept)", flush=True)
                time.sleep(wait)
                continue
            if is_ok(got, f):
                os.replace(got, target)
                mins = (time.time() - t0) / 60
                print(f"[fetch] {k}/{len(want)} {f['name']}: {f['size'] / 1e9:.2f} GB, "
                      f"verified ({mins:.1f} min)", flush=True)
                break
            print(f"[fetch] {f['name']}: size or sha256 mismatch; downloading it again", flush=True)
            got.unlink(missing_ok=True)
            force = True
        if not is_ok(target, f, full=False):
            raise SystemExit(f"[fetch] FATAL: {f['name']} failed {a.attempts} attempts; rerun "
                             f"the job to resume")

    man_path.write_text(json.dumps({"repo": a.repo, "prefix": a.prefix, "revision": a.revision,
                                    "total_bytes": total, "files": files}, indent=1))
    got = sum(f["size"] for f in want)
    print(f"[fetch] complete: {len(want)} files, {got / 1e9:.2f} GB verified in {dest}", flush=True)
    return 0


def _manifest_files(d: Path, first: int) -> tuple[list[dict], list[dict]]:
    files = json.loads((d / "MANIFEST.json").read_text())["files"]
    return files, (files[:first] if first else files)


def names(a) -> int:
    _, want = _manifest_files(Path(a.dir), a.first)
    print("\n".join(f["name"] for f in want))
    return 0


def check(a) -> int:
    d = Path(a.dir)
    files, want = _manifest_files(d, a.first)
    bad = [f["name"] for f in want if not is_ok(d / f["name"], f, full=a.sha256)]
    # Files beyond the first N may be present from an earlier job on this node;
    # fineweb_stream reads every parquet file here, so they must be right too.
    bad += [f["name"] for f in files[len(want):]
            if (d / f["name"]).exists() and not is_ok(d / f["name"], f, full=a.sha256)]
    known = {f["name"] for f in files}
    extra = sorted(p.name for p in d.glob("*.parquet") if p.name not in known)
    print(f"[fetch] {len(want) - len(bad)}/{len(want)} required files verified in {d}"
          + (f"; bad: {bad}" if bad else "") + (f"; not in manifest: {extra}" if extra else ""),
          flush=True)
    return 1 if bad or extra else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download")
    p.add_argument("--dir", required=True)
    p.add_argument("--repo", default=REPO)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--attempts", type=int, default=20, help="per file")
    p.add_argument("--first", type=int, default=0, help="only the first N files (0: all)")
    for name in ("names", "check"):
        p = sub.add_parser(name)
        p.add_argument("--dir", required=True)
        p.add_argument("--first", type=int, default=0, help="only the first N files (0: all)")
        if name == "check":
            p.add_argument("--sha256", action="store_true", help="also verify hashes")
    a = ap.parse_args()
    return {"download": download, "names": names, "check": check}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
