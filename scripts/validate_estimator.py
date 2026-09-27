#!/usr/bin/env python3
"""Validate the Eq. 2-3 estimator on synthetic data with known truth.

This is the proposal's week 3-4 requirement -- check the regression code on a toy
with a known answer before pointing it at a real model -- and it is a gate in its
own right: if the estimator manufactures an effect under a null DGP, nothing
measured on LLaDA or Dream can be believed.

Three checks:
  1. NULL       homoskedastic noise, no rank effect.  beta_z must be ~0.
  2. POSITIVE   a planted rank effect.  beta_z must recover its sign and rough size.
  3. ROBUSTNESS heteroskedastic noise, no rank effect, swept over spline
                flexibility.  This is where the specification is fragile.

Writes a JSON report to --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3 import LEGACY_POOL_SIZES
from c3.regression import fit_once
from c3.simulate import simulate
from c3.splines import Spline


def beta_z(dgp, knots, seed, **kw):
    fr, truth = simulate(dgp=dgp, seed=seed, **kw)
    sp = Spline.fit(fr.ell, n_knots=knots)
    return fit_once(fr, sp, "eq2").coef("beta_z"), truth, len(fr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=60)
    ap.add_argument("--n-positions", type=int, default=256)
    ap.add_argument("--n-pools", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--knots", type=int, nargs="+", default=[5, 7, 9, 13, 17])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="largest |beta_z| accepted under a null DGP")
    ap.add_argument("--out", default="artifacts/estimator_validation.json")
    a = ap.parse_args()

    # Proposal-era pool design, pinned so artifacts/estimator_validation.json
    # reproduces now that the library defaults follow plan v2.
    base = dict(n_docs=a.n_docs, n_states_per_doc=3, n_positions=a.n_positions,
                n_pools_per_size=a.n_pools, pool_sizes=LEGACY_POOL_SIZES,
                pool_stream="shared")
    report: dict = {"config": vars(a), "checks": {}}
    ok = True

    print("1. NULL: homoskedastic, no rank effect (beta_z must be ~0)")
    vals = [beta_z("homoskedastic", 9, s, **base)[0] for s in a.seeds]
    mean = float(np.mean(vals))
    passed = abs(mean) <= a.tolerance
    ok &= passed
    print(f"   beta_z = {mean:+.4f}  [{'PASS' if passed else 'FAIL'}]  (tol {a.tolerance})")
    report["checks"]["null_homoskedastic"] = {"beta_z": mean, "values": vals, "pass": passed}

    print("\n2. POSITIVE CONTROL: planted rank effect (must recover sign and size)")
    planted = {}
    for delta in (-0.30, -0.10):
        vals = [beta_z("direct_rank", 9, s, tau=0.0, delta=delta, **base)[0] for s in a.seeds]
        mean = float(np.mean(vals))
        passed = (mean < 0) and (0.5 * abs(delta) <= abs(mean) <= 1.5 * abs(delta))
        ok &= passed
        print(f"   planted {delta:+.2f} -> recovered {mean:+.4f}  [{'PASS' if passed else 'FAIL'}]")
        planted[str(delta)] = {"recovered": mean, "pass": passed}
    report["checks"]["positive_control"] = planted

    print("\n3. ROBUSTNESS: heteroskedastic, NO rank effect, by spline flexibility")
    print("   (every cell should be ~0; departures are specification error, not signal)")
    hdr = "   tau  | " + " | ".join(f"{k:>2d} knots" for k in a.knots)
    print(hdr + "\n   " + "-" * (len(hdr) - 3))
    grid: dict = {}
    worst = 0.0
    for tau in a.taus:
        row = []
        for k in a.knots:
            vals = [beta_z("heteroskedastic", k, s, tau=tau, **base)[0] for s in a.seeds]
            mean = float(np.mean(vals))
            row.append(mean)
            worst = max(worst, abs(mean))
        grid[str(tau)] = dict(zip(map(str, a.knots), row))
        print(f"   {tau:4.1f} | " + " | ".join(f"{v:+8.4f}" for v in row))
    report["checks"]["heteroskedastic_grid"] = grid
    report["checks"]["worst_spurious_beta_z"] = worst

    print(f"\n   worst spurious |beta_z| over the grid: {worst:.4f}")
    print("   -> the audit's effect bar must exceed this, at the dispersion Stage 0 finds")

    report["overall_pass"] = bool(ok)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
