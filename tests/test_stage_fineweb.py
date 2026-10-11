"""c3_stage_fineweb (slurm/stage_fineweb.sh): node copy, then /share1, then the Hub.

The real bash function runs against stub commands: `rsync` is the system rsync
with `user@ada:` paths mapped to a local directory standing in for /share1, and
the Hub is a local directory served through scripts/fetch_fineweb.py with
`hf_hub_download` and the repo listing patched.  Everything else -- the
`check` that decides whether the node copy is usable, the sha256 verification
of downloads, the lock -- is the production code.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RSYNC = shutil.which("rsync")
pytestmark = pytest.mark.skipif(not (RSYNC and shutil.which("flock")),
                                reason="needs rsync and flock")
pytest.importorskip("huggingface_hub")

BLOBS = {"000_00000.parquet": b"a" * 1000, "001_00000.parquet": b"b" * 2000}

FAKE_FETCH = r'''
import importlib.util, json, os, shutil, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("ff", os.environ["REAL_FETCH"])
ff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ff)
import huggingface_hub
hub = Path(os.environ["FAKE_HUB"])

def fake_download(repo, filename, repo_type, revision, local_dir, force_download):
    with open(os.environ["SHIM_LOG"], "a") as fh:
        fh.write(f"download {filename} {revision}\n")
    out = Path(local_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hub / Path(filename).name, out)
    return str(out)

huggingface_hub.hf_hub_download = fake_download
ff.remote_files = lambda repo, prefix, revision, attempts: json.loads((hub / "listing.json").read_text())
sys.exit(ff.main())
'''

RSYNC_SHIM = r'''#!/bin/bash
echo "rsync $*" >> "$SHIM_LOG"
args=(); skip=0
for a in "$@"; do
    if [ "$skip" = 1 ]; then skip=0; continue; fi
    case "$a" in
        -e) skip=1; continue ;;
        *@ada:*) a="${a#*@ada:}" ;;
    esac
    args+=("$a")
done
exec REAL_RSYNC "${args[@]}"
'''


def _listing(blobs):
    return [{"name": n, "path": f"sample/10BT/{n}", "size": len(b),
             "sha256": hashlib.sha256(b).hexdigest()} for n, b in sorted(blobs.items())]


def _manifest(d: Path, blobs=BLOBS):
    d.mkdir(parents=True, exist_ok=True)
    (d / "MANIFEST.json").write_text(json.dumps(
        {"repo": "HuggingFaceFW/fineweb-edu", "prefix": "sample/10BT",
         "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "files": _listing(blobs)}))


@pytest.fixture
def stage(tmp_path):
    """Returns run(first) -> (returncode, stdout, calls) for c3_stage_fineweb."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "fetch_fineweb.py").write_text(FAKE_FETCH)
    hub = tmp_path / "hub"
    hub.mkdir()
    for n, b in BLOBS.items():
        (hub / n).write_bytes(b)
    (hub / "listing.json").write_text(json.dumps(_listing(BLOBS)))
    shims = tmp_path / "shims"
    shims.mkdir()
    (shims / "rsync").write_text(RSYNC_SHIM.replace("REAL_RSYNC", RSYNC))
    (shims / "rsync").chmod(0o755)
    log = tmp_path / "calls.log"
    env = {**os.environ, "C3_PY": sys.executable, "C3_SCRATCH": str(tmp_path / "scratch"),
           "C3_BLOBS": str(tmp_path / "share1"), "C3_SSH_OPTS": "-o BatchMode=yes",
           "C3_REPO": str(repo), "USER": "tester", "HF_HOME": str(tmp_path / "hf"),
           "PATH": f"{shims}:{os.environ['PATH']}", "SHIM_LOG": str(log),
           "REAL_FETCH": str(ROOT / "scripts" / "fetch_fineweb.py"), "FAKE_HUB": str(hub)}
    (tmp_path / "scratch").mkdir()

    def run(first=0):
        log.write_text("")
        script = ('set -euo pipefail; source "$1"; c3_stage_fineweb "$2"; '
                  'echo "C3_FINEWEB_DIR=$C3_FINEWEB_DIR"')
        p = subprocess.run(["bash", "-c", script, "_", str(ROOT / "slurm" / "stage_fineweb.sh"),
                            str(first)], env=env, capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout + p.stderr, log.read_text().splitlines()

    run.node = tmp_path / "scratch" / "fineweb-10BT"
    run.share1 = tmp_path / "share1" / "fineweb-10BT"
    run.hub = hub
    return run


def _files(d: Path):
    return {p.name: p.read_bytes() for p in d.glob("*.parquet")}


def test_a_verified_node_copy_is_used_without_any_transfer(stage):
    _manifest(stage.node)
    for n, b in BLOBS.items():
        (stage.node / n).write_bytes(b)
    rc, out, calls = stage()
    assert rc == 0, out
    assert "using the copy already on" in out
    assert calls == []                                   # no rsync, no download
    assert f"C3_FINEWEB_DIR={stage.node}" in out


def test_without_a_node_copy_the_share1_copy_is_staged(stage):
    _manifest(stage.share1)
    for n, b in BLOBS.items():
        (stage.share1 / n).write_bytes(b)
    rc, out, calls = stage()
    assert rc == 0, out
    assert "staging 2 file(s)" in out
    assert not any(c.startswith("download") for c in calls)
    assert _files(stage.node) == BLOBS


def test_without_node_or_share1_copy_the_pinned_revision_is_downloaded(stage):
    rc, out, calls = stage()
    assert rc == 0, out
    assert "downloading the pinned revision from the Hub" in out
    downloads = [c for c in calls if c.startswith("download")]
    assert downloads == [f"download sample/10BT/{n} 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
                         for n in sorted(BLOBS)]
    assert _files(stage.node) == BLOBS
    assert f"C3_FINEWEB_DIR={stage.node}" in out


def test_first_n_downloads_only_those_files(stage):
    rc, out, calls = stage(first=1)
    assert rc == 0, out
    assert [c for c in calls if c.startswith("download")] == \
        ["download sample/10BT/000_00000.parquet 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"]
    assert _files(stage.node) == {"000_00000.parquet": BLOBS["000_00000.parquet"]}
    # A later job that needs the whole stream finds the node copy short and fetches the rest.
    rc, out, calls = stage(first=0)
    assert rc == 0, out
    assert [c for c in calls if c.startswith("download")] == \
        ["download sample/10BT/001_00000.parquet 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"]
    assert _files(stage.node) == BLOBS


def test_a_truncated_node_copy_is_repaired_from_the_hub(stage):
    _manifest(stage.node)
    (stage.node / "000_00000.parquet").write_bytes(BLOBS["000_00000.parquet"])
    (stage.node / "001_00000.parquet").write_bytes(b"b" * 10)        # truncated
    rc, out, calls = stage()
    assert rc == 0, out
    assert [c for c in calls if c.startswith("download")] == \
        ["download sample/10BT/001_00000.parquet 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"]
    assert _files(stage.node) == BLOBS


def test_a_hub_file_that_never_verifies_fails_the_job(stage):
    (stage.hub / "001_00000.parquet").write_bytes(b"z" * 2000)       # right size, wrong hash
    rc, out, calls = stage()
    assert rc != 0
    assert "FATAL: no verified FineWeb copy" in out
    assert "C3_FINEWEB_DIR=" not in out
    assert "001_00000.parquet" not in _files(stage.node)
