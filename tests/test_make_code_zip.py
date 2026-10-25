"""scripts/make_code_zip.py on a tiny temporary git repository."""
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_code_zip as Z  # noqa: E402

FILES = {
    "c3/core.py": b"x = 1\n",
    "scripts/run.py": b"print('hi')\n",
    "paper/main.tex": b"\\documentclass{article}\n",
    "artifacts/small.json": b"{}\n",
    "artifacts/big.json": b"0" * (2 * 2**20),
    "artifacts/weights.pt": b"blob",
    "artifacts/table.parquet": b"PAR1",
    ".claude/settings.json": b"{}\n",
    ".github/workflows/tests.yml": b"on: push\n",
    "tools/coord-bus/note.txt": b"bus\n",
}


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    d = tmp_path_factory.mktemp("repo")
    for name, data in FILES.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(data)
    (d / "link.py").symlink_to("c3/core.py")
    git = ["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A", "-f"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)
    (d / "untracked.py").write_text("uncommitted\n")
    return d


def build(repo, tmp_path, *args):
    out = tmp_path / "out" / "code.zip"
    rc = Z.main(["--repo", str(repo), "--out", str(out), "--max-file-mb", "1", *args])
    return rc, out


def test_included_and_excluded(repo, tmp_path, capsys):
    rc, out = build(repo, tmp_path)
    assert rc == 0
    names = set(zipfile.ZipFile(out).namelist())
    assert names == {"c3/c3/core.py", "c3/scripts/run.py", "c3/paper/main.tex",
                     "c3/artifacts/small.json", "c3/artifacts/table.parquet",
                     "c3/MANIFEST.txt"}
    text = capsys.readouterr().out
    for gone in ("big.json", "weights.pt", ".claude/settings.json",
                 "tests.yml", "coord-bus/note.txt", "link.py"):
        assert gone in text


def test_manifest(repo, tmp_path):
    _, out = build(repo, tmp_path)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    man = zipfile.ZipFile(out).read("c3/MANIFEST.txt").decode()
    assert f"commit: {commit}" in man and "ref: HEAD" in man
    assert f'"git_commit": "{commit}"' in man
    assert "c3/core.py" in man
    assert "artifacts/weights.pt  [.pt model weights]" in man
    assert "artifacts/big.json  [over size limit" in man


def test_size_limit_fails_and_removes_zip(repo, tmp_path):
    rc, out = build(repo, tmp_path, "--max-mb", "0.00001")
    assert rc == 1
    assert not out.exists()


def test_default_file_limit_keeps_two_mb(repo, tmp_path):
    out = tmp_path / "code.zip"
    assert Z.main(["--repo", str(repo), "--out", str(out)]) == 0
    assert "c3/artifacts/big.json" in zipfile.ZipFile(out).namelist()
