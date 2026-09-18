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
