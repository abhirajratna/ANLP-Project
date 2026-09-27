"""Provenance stamped into every trace and artifact (plan v2 Sec. 3.4, T14).

Every number in the paper has to come from a script run at a known commit, so
each output records the git commit (and whether the tree was dirty), the library
versions, the machine and GPU, and the SLURM job.  On the cluster the code runs
from an environment tarball without a ``.git`` directory; there the commit comes
from ``C3_GIT_COMMIT``, which the job scripts export.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_PACKAGES = ("numpy", "scipy", "pandas", "pyarrow", "torch", "transformers",
             "bitsandbytes", "accelerate", "matplotlib")


def git_commit() -> tuple[str, bool]:
    """(commit sha, dirty flag).  Falls back to $C3_GIT_COMMIT, then 'unknown'."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=_REPO, capture_output=True, text=True, timeout=10,
                                check=True).stdout.strip()
        return sha, bool(status)
    except (OSError, subprocess.SubprocessError):
        env = os.environ.get("C3_GIT_COMMIT")
        return (env, False) if env else ("unknown", False)


def package_versions(names=_PACKAGES) -> dict[str, str]:
    from importlib import metadata

    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def gpu_info() -> dict:
    """GPU name and compute capability, without importing torch if it is absent."""
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            return {}
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    major, minor = torch.cuda.get_device_capability()
    return {"cuda": True, "gpu": torch.cuda.get_device_name(), "capability": f"{major}.{minor}",
            "n_gpus": torch.cuda.device_count()}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect(extra: dict | None = None, with_gpu: bool = False) -> dict:
    """A provenance record: commit, versions, host, SLURM job, optional GPU."""
    sha, dirty = git_commit()
    rec = {
        "git_commit": sha,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "packages": package_versions(),
    }
    if with_gpu:
        rec["gpu"] = gpu_info()
    if extra:
        rec.update(extra)
    return rec
