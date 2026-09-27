#!/usr/bin/env python3
"""Download the task test splits (GSM8K, MBPP, HumanEval) at their pinned revisions.

    python scripts/prepare_tasks.py --out-dir runs/corpus/tasks \\
        --manifest artifacts/task_manifest.json

Writes one JSONL per task (``doc_id`` = index in the split) and a manifest of
revision, record count and file hash, so a trace can be traced to exact records.
Compute nodes can also read the pinned hub revision directly; the JSONL is for
machines without network access and for the record.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.data.tasks import TASKS, save_task
from c3.provenance import collect as provenance
from c3.util import hard_exit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=sorted(TASKS), choices=sorted(TASKS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", default=None, help="where to write the manifest JSON")
    a = ap.parse_args()

    entries = []
    for task in a.tasks:
        entry = save_task(task, Path(a.out_dir) / f"{task}_test.jsonl")
        print(f"[tasks] {task}: {entry['n']} records @ {entry['revision'][:12]} -> "
              f"{Path(a.out_dir) / entry['file']}", flush=True)
        entries.append(entry)
    if a.manifest:
        Path(a.manifest).write_text(json.dumps({"tasks": entries, "provenance": provenance()},
                                               indent=2))
        print(f"[tasks] manifest -> {a.manifest}")
    return 0


if __name__ == "__main__":
    # hard_exit: datasets leaves worker threads that can abort at finalization
    hard_exit(main())
