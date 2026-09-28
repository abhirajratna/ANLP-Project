"""The local FineWeb copy: download verification, and reading it in the Hub's order."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("datasets")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from c3.data.fineweb import CorpusSpec, fineweb_stream, sample_documents

ROOT = Path(__file__).resolve().parents[1]


def _fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_fineweb", ROOT / "scripts" / "fetch_fineweb.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_parquet(path: Path, ids, n_chars=1300):
    table = pa.table({"id": [str(i) for i in ids], "text": [f"{i} " + "x" * n_chars for i in ids]})
    pq.write_table(table, path)


def test_local_copy_is_read_in_sorted_file_order(tmp_path, monkeypatch):
    # Written out of order: the Hub resolves sample/10BT/* to sorted file names.
    _write_parquet(tmp_path / "001_00000.parquet", range(10, 20))
    _write_parquet(tmp_path / "000_00000.parquet", range(0, 10))
    monkeypatch.setenv("C3_FINEWEB_DIR", str(tmp_path))
    assert [r["id"] for r in fineweb_stream()] == [str(i) for i in range(20)]


def test_empty_local_copy_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("C3_FINEWEB_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        fineweb_stream()


def test_corpus_sampling_refuses_a_stream_shorter_than_its_pool(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    _write_parquet(data / "000_00000.parquet", range(30))
    monkeypatch.setenv("C3_FINEWEB_DIR", str(data))
    sample_documents(CorpusSpec(n_documents=5, reservoir_pool=30), tmp_path / "ok")
    assert sum(1 for _ in open(tmp_path / "ok" / "calibration_docs.jsonl")) == 5
    with pytest.raises(RuntimeError, match="stream ended"):
        sample_documents(CorpusSpec(n_documents=5, reservoir_pool=31), tmp_path / "short")


def _manifest(d: Path, blobs: dict) -> list[dict]:
    files = [{"name": n, "path": f"sample/10BT/{n}", "size": len(b),
              "sha256": hashlib.sha256(b).hexdigest()} for n, b in sorted(blobs.items())]
    (d / "MANIFEST.json").write_text(json.dumps(
        {"repo": "r", "prefix": "sample/10BT", "revision": "v", "files": files}))
    return files


def test_download_retries_a_corrupt_file_and_verifies_every_file(tmp_path, monkeypatch):
    import huggingface_hub

    fetch = _fetch_module()
    blobs = {"000.parquet": b"a" * 100, "001.parquet": b"b" * 200}
    dest = tmp_path / "copy"
    dest.mkdir()
    _manifest(dest, blobs)
    (dest / "000.parquet").write_bytes(blobs["000.parquet"])       # already verified
    calls = []

    def fake_download(repo, filename, repo_type, revision, local_dir, force_download):
        calls.append((filename, force_download))
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        name = Path(filename).name
        # First attempt returns a corrupt file of the right size.
        out.write_bytes(blobs[name] if len(calls) > 1 else b"z" * len(blobs[name]))
        return str(out)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    args = type("A", (), dict(dir=str(dest), repo="r", prefix="sample/10BT", revision="v", attempts=3))
    assert fetch.download(args) == 0
    assert calls == [("sample/10BT/001.parquet", False), ("sample/10BT/001.parquet", True)]
    assert (dest / "001.parquet").read_bytes() == blobs["001.parquet"]


def test_check_rejects_wrong_sizes_and_unlisted_files(tmp_path):
    fetch = _fetch_module()
    blobs = {"000.parquet": b"a" * 100, "001.parquet": b"b" * 200}
    _manifest(tmp_path, blobs)
    args = lambda first=0, sha=False: type("A", (), dict(dir=str(tmp_path), first=first, sha256=sha))
    (tmp_path / "000.parquet").write_bytes(blobs["000.parquet"])
    assert fetch.check(args(first=1)) == 0            # only the first file is required
    assert fetch.check(args()) == 1                   # the second is missing
    (tmp_path / "001.parquet").write_bytes(b"b" * 199)
    assert fetch.check(args(first=1)) == 1            # present beyond N, but truncated
    (tmp_path / "001.parquet").write_bytes(blobs["001.parquet"])
    assert fetch.check(args(sha=True)) == 0
    (tmp_path / "stray.parquet").write_bytes(b"x")
    assert fetch.check(args()) == 1                   # would be read by fineweb_stream
