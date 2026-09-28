#!/usr/bin/env python3
"""Gate G2 (plan v2 T3): has the 110M model finished learning?

Reads ``history.jsonl`` written by the trainer and fits a line to the validation
NELBO over the last ``--tail`` fraction of steps.  The gate passes when the
fitted change across that window is smaller than ``--max-rel-change`` of the
mean -- i.e. the curve is flat.  The threshold must be fixed before the run is
read; the cosine schedule decays the learning rate to ``min_lr_frac`` at the end,
so a flat tail is expected of a converged run and a still-falling tail is not.

Also reports the perplexity bound, which the plan compares against the MDLM
paper and an autoregressive reference for a sanity range.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3.provenance import collect as provenance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--tail", type=float, default=0.10)
    ap.add_argument("--max-rel-change", type=float, default=0.005)
    ap.add_argument("--metric", default="ema_val_nelbo",
                    help="falls back to val_nelbo if the EMA metric is absent")
    ap.add_argument("--out", default="artifacts/gate_g2.json")
    a = ap.parse_args()

    rows = [json.loads(line) for line in Path(a.history).read_text().splitlines() if line.strip()]
    metric = a.metric if any(a.metric in r for r in rows) else "val_nelbo"
    # A requeued job re-logs the steps after its last checkpoint; keep the last
    # value written for each step.
    by_step = {r["step"]: r[metric] for r in rows if metric in r}
    pts = sorted(by_step.items())
    if len(pts) < 3:
        raise SystemExit(f"need at least 3 validation points, have {len(pts)}")
    steps = np.array([p[0] for p in pts], dtype=float)
    vals = np.array([p[1] for p in pts], dtype=float)
    cut = steps.max() - a.tail * (steps.max() - steps.min())
    tail = steps >= cut
    if tail.sum() < 2:
        tail[-2:] = True
    slope, intercept = np.polyfit(steps[tail], vals[tail], 1)
    span = steps[tail].max() - steps[tail].min()
    rel = abs(slope * span) / abs(vals[tail].mean())
    ok = bool(rel < a.max_rel_change)
    res = {"gate": "G2", "pass": ok, "metric": metric, "tail_fraction": a.tail,
           "points_in_tail": int(tail.sum()), "fitted_change_over_tail": float(slope * span),
           "relative_change": float(rel), "threshold": a.max_rel_change,
           "final_nelbo": float(vals[-1]), "final_ppl_bound": float(math.exp(vals[-1])),
           "history": str(a.history), "provenance": provenance()}
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=2))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"G2 {'PASS' if ok else 'FAIL'} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
