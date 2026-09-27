#!/usr/bin/env python3
"""PROPOSAL-ERA record: Stage 0 pilot and gate (proposal Sec. 5.3).

Superseded by plan v2 (see scripts/analyze_traces.py and the T2 pilot).  Kept so
artifacts/stage0_gate_*.json stay reproducible: it pins the proposal-era pool
grid and random stream, and reports the gap in the old sign (agree - conf;
negative = overconfident).  Plan v2 uses conf - agree.

Estimates the two parameters that decide whether C1 is feasible -- the overall
scale of the bias and the dispersion of position-specific noise -- computes the
sample size needed for 80% power, and writes the gate decision to disk so the
decision is recorded before the full study rather than after it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3 import LEGACY_POOL_SIZES
from c3.assemble import frame_from_traces
from c3.calibration import commitment_gap, gap_slope
from c3.dispersion import estimate_dispersion
from c3.power import required_documents, stage0_gate
from c3.regression import fit_and_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="parquet written by collect_traces.py")
    ap.add_argument("--mask-kind", default="uniform", choices=["uniform", "decoder"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-pools", type=int, default=16)
    ap.add_argument("--knots", type=int, default=9)
    ap.add_argument("--floor", type=float, default=0.0,
                    help="trace-level non-determinism floor, from measure_floor.py")
    ap.add_argument("--max-documents", type=int, default=1000)
    ap.add_argument("--dispersion-threshold", type=float, default=0.25)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="artifacts/stage0_gate.json")
    a = ap.parse_args()

    frame = frame_from_traces(a.traces, mask_kind=a.mask_kind,
                              n_pools_per_size=a.n_pools, seed=7,
                              pool_sizes=LEGACY_POOL_SIZES, stream="shared")
    n_docs = int(np.unique(frame.doc_id).size)
    print(f"[stage0] {len(frame)} pool rows, {n_docs} documents, "
          f"{frame.state_index[1]} states, mask_kind={a.mask_kind}")

    # The pilot is far too small to split, so it is fitted pooled. This is the
    # one place cross-fitting is skipped, and it is why Stage 0 is a feasibility
    # estimate rather than a reported result.
    est = fit_and_report(frame, "eq2", n_knots=a.knots, n_boot=a.n_boot,
                         n_jobs=a.n_jobs, cross_fit=False)["beta_z"]
    print(f"[stage0] {est}   (SE {est.se:.4f})")

    disp = estimate_dispersion(frame)
    print(f"[stage0] {disp}")

    power = required_documents(est.value, est.se, n_docs, floor_effect=a.floor)
    print(f"[stage0] {power}")

    gate = stage0_gate(power.n_required, disp.dispersion,
                       max_documents=a.max_documents,
                       dispersion_threshold=a.dispersion_threshold)
    print(f"\n[stage0] GATE: {gate}")

    gaps = commitment_gap(frame, n_boot=a.n_boot)
    print("\n[stage0] commitment gap by field size:")
    for g in gaps:
        print(f"   {g}")
    print(f"   slope in log2(m): {gap_slope(gaps):+.5f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "traces": a.traces, "mask_kind": a.mask_kind, "n_documents": n_docs,
        "beta_z": {"value": est.value, "se": est.se,
                   "ci": [est.ci_low, est.ci_high]},
        "dispersion": disp.dispersion, "power": power.__dict__,
        "gate": gate.to_dict(),
        "commitment_gap": [g.__dict__ for g in gaps],
        "gap_slope_log2m": gap_slope(gaps),
    }, indent=2, default=float))
    print(f"\n[stage0] decision recorded -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
