#!/usr/bin/env python3
"""Score T12 completions: per-configuration accuracy for GSM8K / MBPP / HumanEval.

    python scripts/score_t12.py runs/t12/gsm8k-report --task-jsonl data/tasks/gsm8k.jsonl \\
        --out artifacts/t12/llada-8b-instruct_bf16_gsm8k-report.json

Reads every ``t12.shard*.jsonl`` that ``scripts/run_t12.py`` wrote under each
directory given, scores each (config, doc_id) record with
``c3.data.scoring.score`` against its ``c3.data.tasks.load_task`` record, and
writes, per (task, config): n, n_correct, accuracy and a 95% Wilson interval.
A (task, config, doc_id) seen twice (a resumed shard) is scored once, and a
config whose doc_ids differ from the others' is flagged in ``warnings``.
``--per-record`` also writes every verdict as JSONL, for error analysis.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.data.scoring import TIMEOUT_S, score
from c3.data.tasks import load_task
from c3.provenance import collect as provenance, sha256_file


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def read_records(dirs: list[Path]) -> tuple[list[dict], list[str]]:
    files = sorted(f for d in dirs for f in d.glob("t12.shard*.jsonl"))
    if not files:
        raise SystemExit(f"no t12.shard*.jsonl under {[str(d) for d in dirs]}")
    seen, out = set(), []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["task"], r["config"], int(r["doc_id"]))
                if key not in seen:
                    seen.add(key)
                    out.append(r)
    return out, [str(f) for f in files]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="run_t12.py output directories")
    ap.add_argument("--task-jsonl", action="append", default=[], metavar="[TASK=]PATH",
                    help="task file from prepare_tasks.py (default: pinned hub revision); "
                         "TASK= is needed when the records span several tasks")
    ap.add_argument("--out", type=Path, required=True, help="summary JSON")
    ap.add_argument("--per-record", type=Path, default=None, help="verdicts as JSONL")
    ap.add_argument("--workers", type=int, default=4, help="parallel code executions")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S, help="seconds per program")
    a = ap.parse_args(argv)

    records, files = read_records(a.dirs)
    tasks = sorted({r["task"] for r in records})
    jsonl = {}
    for spec in a.task_jsonl:
        task, _, path = spec.rpartition("=")
        if not task:
            if len(tasks) != 1:
                raise SystemExit(f"records span {tasks}; pass --task-jsonl TASK=PATH")
            task = tasks[0]
        jsonl[task] = path
    refs = {t: {int(r["doc_id"]): r for r in load_task(t, jsonl.get(t))} for t in tasks}

    def one(r):
        v = score(r["task"], refs[r["task"]][int(r["doc_id"])], r["completion"], a.timeout)
        return {"task": r["task"], "config": r["config"], "doc_id": int(r["doc_id"]),
                "stopped": r.get("stopped"), **v}

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        verdicts = list(pool.map(one, records))

    groups = defaultdict(list)
    for v in verdicts:
        groups[(v["task"], v["config"])].append(v)
    rows, warnings = [], []
    for task in tasks:
        id_sets = {cfg: frozenset(v["doc_id"] for v in vs)
                   for (t, cfg), vs in groups.items() if t == task}
        common = frozenset.intersection(*id_sets.values())
        for cfg, ids in sorted(id_sets.items()):
            if ids != common:
                warnings.append(f"{task} {cfg}: {len(ids)} doc_ids, {len(common)} shared by all")
            vs = groups[(task, cfg)]
            k, n = sum(v["correct"] for v in vs), len(vs)
            lo, hi = wilson(k, n)
            rows.append({"task": task, "config": cfg, "n": n, "n_correct": k,
                         "accuracy": k / n, "ci95": [lo, hi],
                         "n_unstopped": sum(v["stopped"] is False for v in vs),
                         "n_error": sum(v["error"] is not None for v in vs)})

    out = {"created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "inputs": files, "task_jsonl": {t: {"path": p, "sha256": sha256_file(p)}
                                           for t, p in jsonl.items()},
           "timeout_s": a.timeout, "rows": rows, "warnings": warnings,
           "provenance": provenance()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    if a.per_record:
        a.per_record.parent.mkdir(parents=True, exist_ok=True)
        with open(a.per_record, "w", encoding="utf-8") as fh:
            for v in verdicts:
                fh.write(json.dumps(v) + "\n")
    for r in rows:
        print(f"{r['task']:9s} {r['config']:32s} {r['n_correct']:5d}/{r['n']:<5d} "
              f"{r['accuracy']:.3f}  [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
