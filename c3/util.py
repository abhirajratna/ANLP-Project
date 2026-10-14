"""Small process-level helpers."""

from __future__ import annotations

import os
import sys


def hard_exit(code: int = 0) -> None:
    """Exit without running interpreter finalization.

    HuggingFace ``datasets`` streaming leaves aiohttp/fsspec worker threads
    running after the iterator is exhausted.  When CPython finalizes, one of them
    releases the GIL from a thread state that is no longer current and the process
    aborts:

        Fatal Python error: PyGILState_Release: thread state ... must be current

    By then the work is done and the output files are written, but the non-zero
    exit makes SLURM mark the job FAILED, which cancels every job queued behind it
    with ``DependencyNeverSatisfied``.  A completed job that reports failure is
    worse than a slow one, so finalization is skipped -- buffers are flushed
    explicitly first, which is the only part of shutdown that matters here.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


FROZEN_SPLITS = ("split_manifest.json", "split_manifest_gsm8k.json", "split_manifest_mbpp.json",
                 "split_manifest_timesplit.json")
"""The pre-registered fit/report splits in ``artifacts/`` (value 1 = report half)."""


def require_report_ok(manifest, reads_report: bool = True) -> None:
    """Refuse to read a frozen split's report half unless C3_REPORT_HALF_OK=1.

    The same guard as ``prereg/analysis_commands.sh``: the report half is read once,
    after the pre-registration is frozen.  Development manifests (any other file,
    e.g. a re-split of the fit half) pass freely.
    """
    from pathlib import Path

    if not reads_report or manifest is None:
        return
    if Path(manifest).name in FROZEN_SPLITS and os.environ.get("C3_REPORT_HALF_OK") != "1":
        raise SystemExit(f"refusing: {manifest} is a pre-registered split and this run reads its "
                         f"report half. Set C3_REPORT_HALF_OK=1 only after the freeze.")


def flat_scalars(d, prefix: str = "") -> dict:
    """Flatten nested dicts to ``a/b`` keys, keeping only scalar leaves (for W&B)."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat_scalars(v, key + "/"))
        elif isinstance(v, (bool, int, float, str)) or v is None:
            out[key] = v
    return out
