#!/usr/bin/env python3
"""Freeze the fit/report split of the GSM8K and MBPP test documents (plan v2 Sec. 3.4, 5).

    python scripts/make_task_split.py --task gsm8k --jsonl runs/corpus/tasks/gsm8k_test.jsonl \\
        --out artifacts/split_manifest_gsm8k.json

The same rule as the FineWeb manifest (scripts/make_split_manifest.py): 50/50 by
``c3.data.fineweb.document_split(n, seed=20260917)`` over the documents in
``doc_id`` order, written as ``{doc_id: 0|1}`` (0 = fit half) for
``analyze_traces.py --split half --manifest``.  A task document's ``doc_id`` is
its index in the pinned test split (``c3.data.tasks``), which is the id every
task trace carries.  The record file is checked against
artifacts/task_manifest.json, and an existing manifest is never overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.data.fineweb import CorpusSpec, document_split
from c3.data.tasks import TASKS, load_task
from c3.provenance import collect as provenance, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["gsm8k", "mbpp"])
    ap.add_argument("--jsonl", required=True, help="scripts/prepare_tasks.py output")
    ap.add_argument("--task-manifest", default="artifacts/task_manifest.json")
    ap.add_argument("--seed", type=int, default=CorpusSpec().seed)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; the split is frozen once written")
    sha = sha256_file(a.jsonl)
    ref = {t["task"]: t for t in json.loads(Path(a.task_manifest).read_text())["tasks"]}[a.task]
    if sha != ref["sha256"]:
        raise SystemExit(f"{a.jsonl} sha256 {sha} != {ref['sha256']} in {a.task_manifest}")
    recs = load_task(a.task, a.jsonl)
    ids = [int(r["doc_id"]) for r in recs]
    if ids != list(range(len(ids))):
        raise SystemExit("doc_id is not the index in the split; the halves would not line up")
    fit, report = document_split(len(ids), seed=a.seed)
    manifest = {str(int(i)): 0 for i in fit} | {str(int(i)): 1 for i in report}
    manifest = dict(sorted(manifest.items(), key=lambda kv: int(kv[0])))
    meta = {"task": a.task, "seed": a.seed, "n": len(ids), "n_fit": len(fit),
            "n_report": len(report), "revision": TASKS[a.task].revision, "records": a.jsonl,
            "records_sha256": sha, "split_function": "c3.data.fineweb.document_split",
            "halves": "0 = fit, 1 = report", "provenance": provenance()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=0))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[split] {a.task}: {len(fit)} fit / {len(report)} report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
