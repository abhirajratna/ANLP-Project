"""T0: what each simulated truth looks like through the plan's estimators (plan v2 T0).

Every number here comes from the same code path the real traces go through --
pools, document-first means, cross-fitted calibrators, Eq. 2 with and without
state fixed effects, the document bootstrap -- applied to ``c3.sim.positions``
scenarios whose truth is known.  The table records what each possible real
result would mean before any real result exists.

Fit half / report half as in the main study: calibrators and f(l) are fitted on
one half of the simulated documents (``frame.hash_halves``) and every reported
number comes from the other half.

Nothing in a scenario is tuned to produce the expected pattern: the scenarios
run at ``simulate_positions``' defaults, the expected pattern is the plan's, and
each cell records whether the two agree.  A mismatch is a finding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .. import DEFAULT_KNOTS, N_BOOT, N_POOLS, POOL_SIZES, RETAIN
from ..assemble import position_and_pool_frames
from ..calibrators import fit_on_half
from ..endpoints.e1 import delta_m, winner_gap, winner_gap_exact
from ..endpoints.e2 import decompose, icc_residuals
from ..frame import PositionFrame, hash_halves
from ..metrics import ece_equal_mass
from ..regression import fit_and_report
from ..stats import declare, doc_boot_counts, percentile_ci
from .positions import SCENARIOS, simulate_positions
from .truth import TRUTH_SCENARIOS, true_calibrators

EXPECTED = {
    # Plan v2 T0 table: Delta(m) grows? | field term | beta_z no FE | beta_z with FE
    "calibrated": {"grows": "no", "field": "0", "beta_nofe": "0", "beta_fe": "0"},
    "uniform": {"grows": "yes", "field": "0", "beta_nofe": "0", "beta_fe": "0"},
    "context": {"grows": "yes", "field": ">0", "beta_nofe": "<0", "beta_fe": "0"},
    "misreading": {"grows": "yes", "field": "?", "beta_nofe": "<0", "beta_fe": "!=0"},
}

FIGURE1_REFERENCE = {
    # Plan v2 Figure 1 caption (from the plan's sim_field.py, not in this repo)
    "gap_m1": 0.001, "gap_m32": 0.18, "ece": 0.087, "own_m32": 0.11, "field_m32": 0.07,
}


@dataclass
class T0Config:
    n_docs: int = 200
    states_per_doc: int = 4
    n_positions: int = 128
    seed: int = 0
    n_boot: int = N_BOOT
    eq_boot: int = N_BOOT
    knots: int = DEFAULT_KNOTS
    n_jobs: int = 1
    ridge: float = 1e-6
    noise_levels: tuple = (0.25, 0.5, 1.0, 1.5, 2.0)
    check_docs: int = 1000
    check_boot: int = 2000


def sign_class(ci) -> str:
    lo, hi = ci
    if lo > 0:
        return ">0"
    if hi < 0:
        return "<0"
    return "0"


def matches(expected: str, observed: str) -> bool:
    if expected == "?":
        return True
    if expected == "!=0":
        return observed != "0"
    return expected == observed


def _cell(value: float, ci) -> dict:
    return {"value": float(value), "ci": [float(ci[0]), float(ci[1])], "class": sign_class(ci)}


def run_scenario(scenario: str, cfg: T0Config) -> dict:
    """The table row, Figure 1 series and diagnostics for one scenario."""
    df = simulate_positions(scenario, n_docs=cfg.n_docs, states_per_doc=cfg.states_per_doc,
                            n_positions=cfg.n_positions, seed=cfg.seed)
    pos, pools = position_and_pool_frames(df, pool_sizes=POOL_SIZES, n_pools_per_size=N_POOLS,
                                          seed=cfg.seed, retain=RETAIN,
                                          min_state_size=max(POOL_SIZES))
    codes = np.unique(pos.doc_id)
    counts = doc_boot_counts(codes.size, cfg.n_boot, seed=cfg.seed, label="t0")
    half = hash_halves(codes, cfg.seed)
    fit_docs, report_docs = codes[half == 0], codes[half == 1]
    report = np.isin(pos.doc_id, report_docs)
    hi, lo = max(POOL_SIZES), min(POOL_SIZES)

    pi_own, pi_ctx, _ = fit_on_half(pos, fit_docs, n_knots=cfg.knots)
    delta = delta_m(pools, pos, codes, counts, pos_mask=report)
    c_val, c_draws = delta.contrast(hi, lo)
    c_ci = percentile_ci(c_draws)
    dec = decompose(pools, pos, pi_own, pi_ctx, codes, counts, pos_mask=report)
    g = dec.field.index(hi)
    field_ci = dec.field.ci()[g]

    pr = pos.select(report)
    ece = ece_equal_mass(pr.conf, pr.agree, 15, np.searchsorted(codes, pr.doc_id), counts)

    reg = {}
    winners = pools.winners()
    for fe in (False, True):
        est = fit_and_report(winners, "eq2", n_knots=cfg.knots, n_boot=cfg.eq_boot,
                             seed=cfg.seed, n_jobs=cfg.n_jobs, cross_fit=True, fe=fe,
                             fit_docs=fit_docs, max_rank=RETAIN, ridge_beta=cfg.ridge)
        bz = est["beta_z"]
        reg["fe" if fe else "nofe"] = {**_cell(bz.value, (bz.ci_low, bz.ci_high)),
                                       "se": bz.se, "converged": bz.converged}

    row = {
        "contrast": {**_cell(c_val, c_ci), "name": f"Delta({hi}) - Delta({lo})"},
        "grows": "yes" if declare(c_val, c_ci, 0.0) and c_val > 0 else "no",
        "field_m32": _cell(dec.field.point[g], field_ci),
        "beta_nofe": reg["nofe"],
        "beta_fe": reg["fe"],
    }
    if scenario in TRUTH_SCENARIOS:
        own_t, ctx_t = true_calibrators(df, scenario)
        ft = winner_gap(pools, pos, own_t, ctx_t, codes, counts, pos_mask=report)
        row["field_m32_true"] = _cell(ft.point[ft.index(hi)], ft.ci()[ft.index(hi)])
    exp = EXPECTED[scenario]
    observed = {"grows": row["grows"], "field": row["field_m32"]["class"],
                "beta_nofe": row["beta_nofe"]["class"], "beta_fe": row["beta_fe"]["class"]}
    row["expected"] = exp
    row["observed"] = observed
    row["matches"] = {k: matches(exp[k], observed[k]) for k in exp}

    return {
        "row": row,
        "series": {name: {"m": [int(m) for m in est.levels], "point": est.point.tolist(),
                          "ci": est.ci().tolist()}
                   for name, est in (("delta", dec.delta), ("delta_own", dec.delta_own),
                                     ("delta_ctx", dec.delta_ctx), ("field", dec.field),
                                     ("residual", dec.residual))},
        "ece": {"value": ece.ece, "ci": list(ece.ci)},
        "icc_residuals": icc_residuals(pos, pi_ctx, codes, counts, pos_mask=report),
        "n_docs_fit": int(fit_docs.size), "n_docs_report": int(report_docs.size),
    }


def field_growth(cfg: T0Config) -> dict:
    """Does the *true* field term grow with m?  Context scenario, several noise levels.

    Uses the exact pi(l) and pi(l, s) and the exact expectation over all pools
    (``c3.pools``), so neither calibrator error nor pool sampling enters: what is
    left is the model's property plus the finite sample of states.
    """
    out = {}
    for sigma in cfg.noise_levels:
        df = simulate_positions("context", n_docs=cfg.check_docs,
                                states_per_doc=cfg.states_per_doc,
                                n_positions=cfg.n_positions, seed=cfg.seed, sigma=sigma)
        pos = PositionFrame.from_frame(df)
        own_t, ctx_t = true_calibrators(df, "context", sigma=sigma)
        codes = np.unique(pos.doc_id)
        counts = doc_boot_counts(codes.size, cfg.check_boot, seed=cfg.seed, label="t0-growth")
        g = winner_gap_exact(pos, own_t, ctx_t, codes, counts,
                             pool_sizes=POOL_SIZES, min_state_size=max(POOL_SIZES))
        steps = np.diff(g.point)
        step_draws = np.diff(g.draws, axis=1)
        out[str(sigma)] = {
            "sigma": sigma, "m": [int(m) for m in g.levels], "field": g.point.tolist(),
            "ci": g.ci().tolist(),
            "increments": steps.tolist(),
            "increment_ci": [list(percentile_ci(step_draws[:, j])) for j in range(steps.size)],
            "monotone_point": bool(np.all(steps > 0)),
            "every_increment_ci_above_zero": bool(all(percentile_ci(step_draws[:, j])[0] > 0
                                                      for j in range(steps.size))),
        }
    return out


def run_t0(cfg: T0Config, scenarios=SCENARIOS) -> dict:
    rows = {s: run_scenario(s, cfg) for s in scenarios}
    mismatches = [f"{s}: {k} expected {r['row']['expected'][k]}, observed {r['row']['observed'][k]}"
                  for s, r in rows.items() for k, ok in r["row"]["matches"].items() if not ok]
    return {"config": asdict(cfg), "scenarios": rows, "mismatches": mismatches,
            "field_growth": field_growth(cfg) if cfg.noise_levels else {},
            "figure1_reference": FIGURE1_REFERENCE}
