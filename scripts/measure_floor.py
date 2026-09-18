#!/usr/bin/env python3
"""The trace-level non-determinism floor.

Sec. 7 declares an effect only if the estimate exceeds twice this floor, so it
has to be measured before any endpoint can be read.  Identical inputs are run
repeatedly; whatever ``beta_z`` moves by across those runs is noise the hardware
and kernels produce on their own, and no effect smaller than that is real.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3.assemble import frame_from_traces
from c3.regression import fit_once
from c3.splines import Spline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True,
                    help="repeated trace files over identical inputs")
    ap.add_argument("--mask-kind", default="uniform")
    ap.add_argument("--knots", type=int, default=9)
    ap.add_argument("--n-pools", type=int, default=16)
    ap.add_argument("--out", default="artifacts/nondeterminism_floor.json")
    a = ap.parse_args()

    if len(a.traces) < 2:
        raise SystemExit("need at least two repeats to measure a floor")

    betas, agree_rates, ell_means, ells, agrees = [], [], [], [], []
    for path in a.traces:
        fr = frame_from_traces(path, mask_kind=a.mask_kind,
                               n_pools_per_size=a.n_pools, seed=7)
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
    }
    for k, v in result.items():
        print(f"  {k}: {v}")

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
