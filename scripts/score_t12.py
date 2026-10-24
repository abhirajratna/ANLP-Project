#!/usr/bin/env python3
"""Score T12 completions: per-configuration accuracy for GSM8K / MBPP / HumanEval.

    python scripts/score_t12.py runs/t12/gsm8k-report --task-jsonl data/tasks/gsm8k.jsonl \\
        --out artifacts/t12/llada-8b-instruct_fp16_gsm8k-report.json

Reads every ``t12.shard*.jsonl`` that ``scripts/run_t12.py`` wrote under each
directory given, with its ``t12.shard*.meta.json``, scores each record with
``c3.data.scoring.score`` against its ``c3.data.tasks.load_task`` record, and
writes one row per (run, task, config): n, n_correct, accuracy, a 95% Wilson
interval, and for GSM8K lm-eval's strict-match and flexible-extract accuracy.

A *run* is the driver's identity fields (model, precision, prompt, half, task
file, grid, ...) without the shard number, so the shards of one sweep pool
into one row, while a chat sweep and an lm-eval sanity run never merge.  A
shard with no meta file falls back to its records' ``run_id``.  Within a run a
(task, config, doc_id) seen twice (a resumed shard) is scored once; if the two
completions differ it is counted in ``warnings``.

Records the driver could not generate (``completion`` None, ``error`` set, e.g.
CUDA OOM) score as wrong and are counted in ``n_driver_error``; timeouts,
syntax errors and GSM8K answers with no number have their own counts, and
every row with a driver error, or with doc_ids that differ from the other
configs of its run, is named in ``warnings``.  ``--per-record`` writes every
verdict as JSONL, for error analysis.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
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

# run_t12.IDENTITY without the shard fields: what makes shards one run
RUN_FIELDS = ("model", "model_revision", "precision", "task", "task_jsonl_sha256",
              "manifest_sha256", "half", "gen_length", "block_length", "decoding", "prompt",
              "configs_sha256", "calibrator_sha256", "c3a_clamp", "pc_sampler_variant")
SYNTAX = ("SyntaxError", "IndentationError", "TabError")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _prompt_format(meta: dict | None) -> str | None:
    p = (meta or {}).get("prompt")
    return p.get("format") if isinstance(p, dict) else p


def run_info(meta: dict | None, rec: dict, shard: Path) -> tuple[str, dict]:
    """(run key, description) of one record; see the module docstring."""
    if meta is not None:
        ident = {f: meta.get(f) for f in RUN_FIELDS}
        key = hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()
        return key[:12], {"model": meta.get("model"), "precision": meta.get("precision"),
                          "half": meta.get("half"), "prompt_format": _prompt_format(meta),
                          "label": meta.get("label")}
    rid = rec.get("run_id")
    key = f"run_id={rid}" if rid else f"dir={shard.parent}"
    return key, {"model": None, "precision": rec.get("precision"), "half": None,
                 "prompt_format": None, "label": None}


def read_records(dirs: list[Path]) -> tuple[list[dict], list[dict], list[str]]:
    """(unique records with ``_run`` / ``_run_info`` added, input descriptions, warnings)."""
    files = sorted(f for d in dirs for f in d.glob("t12.shard*.jsonl"))
    if not files:
        raise SystemExit(f"no t12.shard*.jsonl under {[str(d) for d in dirs]}")
    seen, out, inputs, warnings = {}, [], [], []
    conflicts = defaultdict(int)
    for f in files:
        meta_path = f.with_name(f.name[:-len(".jsonl")] + ".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if meta is None:
            warnings.append(f"{f}: no meta file; grouped by record run_id")
        starts = (meta or {}).get("starts") or [{}]
        inputs.append({"path": str(f), "sha256": sha256_file(f),
                       "meta": str(meta_path) if meta else None,
                       "run_id": (meta or {}).get("run_id"), "prompt": (meta or {}).get("prompt"),
                       "half": (meta or {}).get("half"), "model": (meta or {}).get("model"),
                       "git_commit": starts[-1].get("git_commit")
                       or (meta or {}).get("provenance", {}).get("git_commit")})
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                run, info = run_info(meta, r, f)
                key = (run, r["task"], r["config"], int(r["doc_id"]))
                if key in seen:
                    if seen[key] != r.get("completion"):
                        conflicts[run] += 1
                    continue
                seen[key] = r.get("completion")
                out.append({**r, "_run": run, "_run_info": info})
    warnings += [f"run {run}: {n} repeated (config, doc_id) with a different completion; "
                 "the first was scored" for run, n in sorted(conflicts.items())]
    return out, inputs, warnings


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

    records, inputs, warnings = read_records(a.dirs)
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
        base = {"run": r["_run"], "run_id": r.get("run_id"),
                "prompt_format": r["_run_info"]["prompt_format"], "task": r["task"],
                "config": r["config"], "doc_id": int(r["doc_id"]), "stopped": r.get("stopped")}
        if r.get("error") or r.get("completion") is None:
            return {**base, "correct": False, "extracted": None, "driver_error": True,
                    "error": f"driver: {r.get('error') or 'no completion'}"}
        v = score(r["task"], refs[r["task"]][int(r["doc_id"])], r["completion"], a.timeout)
        return {**base, **v, "driver_error": False}

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        verdicts = list(pool.map(one, records))

    info = {r["_run"]: r["_run_info"] for r in records}
    groups = defaultdict(list)
    for v in verdicts:
        groups[(v["run"], v["task"], v["config"])].append(v)
    rows = []
    for run, task in sorted({(k[0], k[1]) for k in groups}):
        id_sets = {cfg: frozenset(v["doc_id"] for v in vs)
                   for (rn, t, cfg), vs in groups.items() if (rn, t) == (run, task)}
        common = frozenset.intersection(*id_sets.values())
        for cfg, ids in sorted(id_sets.items()):
            if ids != common:
                warnings.append(f"run {run} {task} {cfg}: {len(ids)} doc_ids, "
                                f"{len(common)} shared by all configs")
            vs = groups[(run, task, cfg)]
            k, n = sum(v["correct"] for v in vs), len(vs)
            lo, hi = wilson(k, n)
            errs = [v["error"] or "" for v in vs if not v["driver_error"]]
            row = {"run": run, **info[run], "task": task, "config": cfg, "n": n, "n_correct": k,
                   "accuracy": k / n, "ci95": [lo, hi],
                   "n_unstopped": sum(v["stopped"] is False for v in vs),
                   "n_driver_error": sum(v["driver_error"] for v in vs),
                   "n_timeout": sum(e == "timeout" for e in errs),
                   "n_syntax_error": sum(e.startswith(SYNTAX) for e in errs),
                   "n_no_answer": sum(e == "no number" for e in errs)}
            if task == "gsm8k":
                for m in ("lmeval_strict", "lmeval_flexible"):
                    km = sum(bool(v.get(m)) for v in vs)
                    row[f"n_correct_{m}"], row[f"accuracy_{m}"] = km, km / n
            if row["n_driver_error"]:
                warnings.append(f"run {run} {task} {cfg}: {row['n_driver_error']} driver errors "
                                "scored as wrong")
            rows.append(row)

    out = {"created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "inputs": inputs, "task_jsonl": {t: {"path": p, "sha256": sha256_file(p)}
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
        tag = f"{r['run']}/{r['prompt_format'] or '?'}"
        print(f"{tag:22s} {r['task']:9s} {r['config']:32s} {r['n_correct']:5d}/{r['n']:<5d} "
              f"{r['accuracy']:.3f}  [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
