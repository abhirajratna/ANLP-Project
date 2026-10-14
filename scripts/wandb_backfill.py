#!/usr/bin/env python3
"""Upload a finished run's history.jsonl to W&B, as if it had been logged live.

    python scripts/wandb_backfill.py --history artifacts/history_train110m-v3.jsonl \\
        --run-id train110m-v3 --summary artifacts/train_summary_train110m-v3.json \\
        artifacts/gate_g2_train110m-v3.json artifacts/eval_110m_train110m-v3.json

From 2026-10-12 every run's W&B start failed with HTTP 403 (the key on the account
had lost write access to the team) and the trainer then logged nothing to W&B; the
same rows are in each run's history.jsonl, which this script replays.

* Rows with the same step (a training row and a validation row; or a step re-logged
  after a requeue) are merged, the later value winning -- the rule check_g2.py uses.
* The run is created under ``--run-id`` and **never resumed**: if the id already
  exists the upload is refused, so nothing is logged twice.  The trainer's run
  name can already be taken by a teammate's run in the shared project
  (``train110m-v3`` was, from 28 Sep), so pass a distinct id in that case.
* ``--summary`` JSON files (train summary, G2 record, eval) go into the run summary
  (scalars, flattened); the train summary's ``config`` and ``model_config`` become
  the run config.  The run is tagged ``backfilled``.

``--dry-run`` reads and merges without contacting W&B.  Needs the ``wandb`` library
and a key with write access in ~/.netrc (no ``wandb`` command is needed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT, ENTITY = "c3-dllm-calibration", "ANLP-project"


def rows_by_step(path: str | Path) -> list[tuple[int, dict]]:
    """history.jsonl -> [(step, merged row without 'step')], ascending, one per step."""
    merged: dict[int, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row.pop("step"))
        merged.setdefault(step, {}).update(row)
    return sorted(merged.items())


KINDS = ("train_summary", "gate_g2", "eval_110m")


def summary_prefix(stem: str) -> str:
    """'gate_g2_train110m-v3' -> 'gate_g2': the file's kind, not the run it came from
    (the W&B id can differ from the run name, e.g. train110m-v3-oct12)."""
    return next((k for k in KINDS if stem == k or stem.startswith(k + "_")), stem)


def flat_scalars(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat_scalars(v, key + "/"))
        elif isinstance(v, (bool, int, float, str)) or v is None:
            out[key] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True)
    ap.add_argument("--run-id", required=True, help="the trainer's W&B id (its --run-name)")
    ap.add_argument("--name", default=None, help="display name (default: the id)")
    ap.add_argument("--summary", nargs="*", default=[], help="JSON files for the run summary")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = rows_by_step(a.history)
    if not rows:
        raise SystemExit(f"{a.history}: no rows")
    config, summary = {}, {}
    for path in a.summary:
        d = json.loads(Path(path).read_text())
        for key in ("config", "model_config"):
            if isinstance(d.get(key), dict):
                config.update(d[key] if key == "config" else {"model": d[key]})
        stem = summary_prefix(Path(path).stem)
        summary.update(flat_scalars({k: v for k, v in d.items()
                                     if k not in ("config", "model_config", "provenance",
                                                  "samples")}, stem + "/"))
    n_val = sum(1 for _, r in rows if "val_nelbo" in r)
    print(f"[backfill] {a.run_id}: {len(rows)} steps ({rows[0][0]}..{rows[-1][0]}), "
          f"{n_val} with validation; {len(summary)} summary values -> {a.entity}/{a.project}")
    if a.dry_run:
        return 0

    import wandb
    try:
        viewer = wandb.Api().viewer
        print(f"[backfill] W&B user: {getattr(viewer, 'username', viewer)}")
    except Exception as exc:
        raise SystemExit(f"[backfill] FATAL: cannot reach W&B with the key in ~/.netrc: {exc}")
    # resume="never": wandb.init refuses an id that already exists, so a second
    # backfill of the same run fails instead of logging every step twice.
    run = wandb.init(project=a.project, entity=a.entity, id=a.run_id, name=a.name or a.run_id,
                     config=config, tags=["backfilled", *a.tags], resume="never",
                     notes=f"Backfilled from {Path(a.history).name}: W&B rejected the live "
                           f"run (HTTP 403, key without write access).")
    for step, row in rows:
        run.log(row, step=step)
    for k, v in summary.items():
        run.summary[k] = v
    url = run.url
    run.finish()
    print(f"[backfill] {a.run_id}: uploaded {len(rows)} steps -> {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
