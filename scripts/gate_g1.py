#!/usr/bin/env python3
"""Gate G1 on the T2 pilot trace (plan v2 T2); see ``c3.gates`` for the rule.

    python scripts/gate_g1.py --traces runs/traces/t2pilot/t2pilot.rep0.parquet \\
        --floor-json artifacts/floor_t2_llada_base_fp16.json --out artifacts/gate_g1.json

The decision is recorded, not acted on: which fallback to take is the team's call.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3 import DEFAULT_KNOTS, DELTA_MIN, N_BOOT
from c3.assemble import SCALAR_COLUMNS, arm_labels, load_traces
from c3.gates import g1_arm, g1_decision
from c3.provenance import collect as provenance, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True,
                    help="pilot parquet, or sim:<scenario>[,n_docs=N] for a test")
    fl = ap.add_mutually_exclusive_group(required=True)
    fl.add_argument("--floor-json", help="scripts/measure_floor.py output (plan_v2.floor_delta)")
    fl.add_argument("--floor", type=float, help="the floor in Delta units, given directly")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--knots", type=int, default=DEFAULT_KNOTS)
    ap.add_argument("--delta-min", type=float, default=DELTA_MIN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.traces.startswith("sim:"):
        from c3.sim.positions import simulate_positions
        name, *kv = a.traces[4:].split(",")
        df = simulate_positions(name, **{k: int(v) for k, v in (p.split("=") for p in kv)})
        source = {"source": a.traces}
    else:
        df = load_traces(a.traces, columns=SCALAR_COLUMNS)
        source = {"source": a.traces, "sha256": sha256_file(a.traces)}
    floor = (a.floor if a.floor is not None
             else float(json.loads(Path(a.floor_json).read_text())["plan_v2"]["floor_delta"]))

    df["arm"] = arm_labels(df)
    arms = {}
    for arm in sorted(df["arm"].unique()):
        print(f"[g1] {arm}", flush=True)
        arms[arm] = g1_arm(df[df["arm"] == arm], a.delta_min, a.n_boot, a.seed, a.knots)
        for key in ("delta_contrast", "field_contrast"):
            r = arms[arm][key]
            print(f"[g1]   {key}: {r['value']:+.4f}, sigma_d {r['sigma_d']:.4f} over "
                  f"{r['n_docs']} docs -> N = {r['n_required']}", flush=True)
    decision = g1_decision(arms, floor, a.delta_min)
    print(f"[g1] decision: {decision['decision']} ({decision.get('meaning', decision.get('reason'))})")

    out = {"gate": "G1", "input": source, "floor_source": a.floor_json or "--floor",
           "config": {"n_boot": a.n_boot, "knots": a.knots, "delta_min": a.delta_min,
                      "seed": a.seed, "field_calibrators": "leave-one-document-out"},
           "arms": arms, "decision": decision,
           "note": "recorded for the team; acting on it is a team decision",
           "provenance": provenance(),
           "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"[g1] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
