#!/usr/bin/env python3
"""The trace-level non-determinism floor.

Sec. 7 declares an effect only if the estimate exceeds twice this floor, so it
has to be measured before any endpoint can be read.  Identical inputs are run
repeatedly; whatever ``beta_z`` moves by across those runs is noise the hardware
and kernels produce on their own, and no effect smaller than that is real.

Plan v2 (T1 step 4) defines the floor as the largest change in ell and in
Delta(m) across 5 runs of the 20 pilot documents; that is reported per arm under
``plan_v2``, and ``plan_v2.floor_delta`` (Delta units) is what
``analyze_traces.py --floor`` takes.  The beta_z spread is kept for the
proposal-era record.  ``--out`` has no default so a re-check can never overwrite
the committed ``artifacts/nondeterminism_floor.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3 import LEGACY_POOL_SIZES, N_POOLS, POOL_SIZES
from c3.assemble import (SCALAR_COLUMNS, arm_labels, frame_from_traces, load_traces,
                         position_and_pool_frames)
from c3.endpoints.e1 import delta_m
from c3.provenance import collect as provenance
from c3.regression import fit_once
from c3.splines import Spline
from c3.stats import doc_boot_counts


def plan_v2_floor(paths) -> dict:
    """Per arm: the largest change across repeats in ell and in Delta(m)."""
    dfs = [load_traces(p, columns=SCALAR_COLUMNS) for p in paths]
    for df in dfs:
        df["arm"] = arm_labels(df)
    arms, overall = {}, 0.0
    for arm in sorted(dfs[0]["arm"].unique()):
        ells, deltas, levels = [], [], None
        for df in dfs:
            d = df[df["arm"] == arm].reset_index(drop=True)
            ells.append(d["ell"].to_numpy())
            try:
                pos, pools = position_and_pool_frames(d, pool_sizes=POOL_SIZES,
                                                      n_pools_per_size=N_POOLS, seed=0,
                                                      min_state_size=max(POOL_SIZES))
            except ValueError:
                continue                  # no state large enough for a pool
            codes = np.unique(pos.doc_id)
            g = delta_m(pools, pos, codes, doc_boot_counts(codes.size, 1, label="floor"))
            deltas.append(g.point)
            levels = g.levels
        same = len({e.shape for e in ells}) == 1
        rec = {"max_abs_delta_ell": float(max(np.abs(e - ells[0]).max() for e in ells))
               if same else None, "rows_identical_shape": same}
        if deltas and len({x.shape for x in deltas}) == 1:
            D = np.vstack(deltas)
            spread = np.abs(D - D[0]).max(axis=0)
            rec["max_abs_change_delta_m"] = {int(m): float(v) for m, v in zip(levels, spread)}
            rec["max_abs_change_delta"] = float(spread.max())
            overall = max(overall, rec["max_abs_change_delta"])
        arms[arm] = rec
    return {"per_arm": arms, "floor_delta": overall, "effect_bar_2x_floor": 2 * overall,
            "definition": "max over arms and m of |Delta_rep(m) - Delta_rep0(m)|"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True,
                    help="repeated trace files over identical inputs")
    ap.add_argument("--mask-kind", default="uniform")
    ap.add_argument("--knots", type=int, default=9)
    ap.add_argument("--n-pools", type=int, default=16)
    ap.add_argument("--out", required=True,
                    help="new path; never the committed artifacts/nondeterminism_floor.json")
    a = ap.parse_args()

    if len(a.traces) < 2:
        raise SystemExit("need at least two repeats to measure a floor")

    betas, agree_rates, ell_means, ells, agrees = [], [], [], [], []
    for path in a.traces:
        fr = frame_from_traces(path, mask_kind=a.mask_kind,
                               n_pools_per_size=a.n_pools, seed=7,
                               pool_sizes=LEGACY_POOL_SIZES, stream="shared")
        sp = Spline.fit(fr.ell, n_knots=a.knots)
        betas.append(fit_once(fr, sp, "eq2").coef("beta_z"))
        agree_rates.append(float(fr.agree.mean()))
        ell_means.append(float(fr.ell.mean()))
        ells.append(fr.ell)
        agrees.append(fr.agree)

    betas = np.asarray(betas)
    floor = float(betas.std(ddof=1))

    # Compare the raw traces too, not just the endpoint: if the forward pass is
    # deterministic the endpoint spread is zero for an uninteresting reason, and
    # that needs saying rather than reporting a floor of 0 as though it were a
    # measurement of noise.
    same_shape = len({e.shape for e in ells}) == 1
    max_dell = float(max(np.abs(e - ells[0]).max() for e in ells)) if same_shape else float("nan")
    n_agree_flips = int(sum(int((ag != agrees[0]).sum()) for ag in agrees)) if same_shape else -1

    result = {
        "n_repeats": len(betas), "beta_z_per_run": betas.tolist(),
        "beta_z_sd": floor, "beta_z_range": float(betas.max() - betas.min()),
        "effect_bar_2x_floor": 2 * floor,
        "max_abs_delta_ell": max_dell, "agree_flips_across_repeats": n_agree_flips,
        "agree_rate_per_run": agree_rates, "ell_mean_per_run": ell_means,
        "bitwise_identical": bool(same_shape and max_dell == 0.0 and n_agree_flips == 0),
        "plan_v2": plan_v2_floor(a.traces),
        "traces": a.traces,
        "provenance": provenance(),
    }
    for k, v in result.items():
        if k not in ("plan_v2", "traces", "provenance"):
            print(f"  {k}: {v}")
    v2 = result["plan_v2"]
    print(f"  plan v2 floor (max |change in Delta(m)|): {v2['floor_delta']:.6f}; "
          f"per arm max |change in ell|: "
          f"{ {a: r['max_abs_delta_ell'] for a, r in v2['per_arm'].items()} }")

    if result["bitwise_identical"]:
        print("""
  FLOOR IS EXACTLY ZERO. Repeated runs are bitwise identical, so the forward
  pass is deterministic at fixed batch composition on one GPU. This is a real
  property of the instrument, but it means the proposal's rule -- declare an
  effect only if it exceeds TWICE the trace-level non-determinism floor -- is
  vacuous here: 2 x 0 = 0, and the bar does no work.

  The effect bar has to come from somewhere else. Two candidates, both larger:
    * the bootstrap interval excluding zero (already the other half of the rule);
    * the worst spurious |beta_z| from scripts/validate_estimator.py, which is
      ~0.12 under heteroskedastic noise with a 5-knot control. That is a far more
      binding bar than hardware noise, and it is the one that should be used.

  A floor that measures something would need to vary what actually varies in
  practice: batch composition (reduction order), node, or precision path.""")
    else:
        print(f"\n  => an effect counts only if |beta_z| > {2*floor:.5f}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"  floor -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
