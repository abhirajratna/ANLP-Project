"""Gate G1 (plan v2 T2): is the effect detectable with the documents we have?

From the 20-document pilot, for every mask arm:

* the per-document contrast delta_d = Delta_d(32) - Delta_d(1), its standard
  deviation sigma_d across documents, and N = 7.85 sigma_d^2 / delta_min^2
  (``power.required_documents_sd``), with delta_min fixed before looking
  (``c3.DELTA_MIN``);
* the same for the field-term contrast field_d(32) - field_d(1);
* the ECE over all masked positions, and the non-determinism floor.

N is powered against delta_min, not against the pilot's own estimate: with 20
documents the point estimate is too noisy to power against, and doing so would
let a lucky pilot pass the gate by construction.

The rule, as the plan states it: N <= 1,000 documents means the full plan.
Otherwise one of the plan's fallbacks applies, chosen mechanically here so it is
fixed before the pilot is read -- N <= 2,000 (the whole corpus) keeps the 8B
arms but on fewer settings; N > 2,000 means even the whole corpus is
underpowered, so weight shifts to the 110M model and C2.  The gate is taken on
the largest N over the arms and both primary contrasts.

The pilot is too small to halve, so the field term uses leave-one-document-out
calibrators (as ``analyze_traces.py --split kfold-doc`` does).
"""

from __future__ import annotations

import numpy as np

from . import DEFAULT_KNOTS, DELTA_MIN, N_BOOT, N_POOLS, POOL_SIZES, RETAIN
from .assemble import position_and_pool_frames
from .calibrators import crossfit_predictions
from .endpoints.e1 import delta_m
from .endpoints.e2 import decompose
from .metrics import ece_equal_mass
from .power import required_documents_sd
from .stats import doc_boot_counts, percentile_ci

G1_MAX_DOCS = 1000
"""Plan v2 T2 / Sec. 6: the full plan runs if N <= 1,000 documents."""

G1_CORPUS_DOCS = 2000
"""The calibration corpus: the most documents any fallback can use."""

FALLBACKS = {
    "full_plan": "N <= 1,000: run the full plan",
    "fewer_8b_settings": "1,000 < N <= 2,000: use all 2,000 documents on fewer 8B settings",
    "shift_to_110m_and_c2": "N > 2,000: even the whole corpus is underpowered; shift weight "
                            "to the 110M model and C2",
}


def _contrast_summary(est, delta_min: float) -> dict:
    hi, lo = max(POOL_SIZES), min(POOL_SIZES)
    point, draws = est.contrast(hi, lo)
    per_doc = est.per_doc_contrast(hi, lo)
    sigma = float(np.std(per_doc, ddof=1)) if per_doc.size > 1 else float("nan")
    return {"name": f"({hi}) - ({lo})", "value": point, "ci": list(percentile_ci(draws)),
            "n_docs": int(per_doc.size), "sigma_d": sigma,
            "n_required": required_documents_sd(sigma, delta_min) if np.isfinite(sigma) else None}


def g1_arm(df_arm, delta_min: float = DELTA_MIN, n_boot: int = N_BOOT, seed: int = 0,
           knots: int = DEFAULT_KNOTS) -> dict:
    """The G1 quantities for one mask arm of a pilot trace."""
    pos, pools = position_and_pool_frames(df_arm, pool_sizes=POOL_SIZES,
                                          n_pools_per_size=N_POOLS, seed=seed, retain=RETAIN,
                                          min_state_size=max(POOL_SIZES))
    codes = np.unique(pos.doc_id)
    counts = doc_boot_counts(codes.size, n_boot, seed=seed, label="gate-g1")
    delta = delta_m(pools, pos, codes, counts)
    pi_own, pi_ctx, _ = crossfit_predictions(pos, {int(d): int(d) for d in codes},
                                             n_knots=knots)
    dec = decompose(pools, pos, pi_own, pi_ctx, codes, counts)
    ece = ece_equal_mass(pos.conf, pos.agree, 15, np.searchsorted(codes, pos.doc_id), counts)
    return {"n_docs": int(codes.size), "n_states": int(np.unique(pos.state_id).size),
            "delta_contrast": _contrast_summary(delta, delta_min),
            "field_contrast": _contrast_summary(dec.field, delta_min),
            "ece": {"value": ece.ece, "ci": list(ece.ci)}}


def g1_decision(arms: dict, floor: float, delta_min: float = DELTA_MIN,
                max_docs: int = G1_MAX_DOCS, corpus_docs: int = G1_CORPUS_DOCS) -> dict:
    """Apply the G1 rule to per-arm results; records the binding arm and endpoint."""
    worst, binding = -1, None
    for arm, r in arms.items():
        for key in ("delta_contrast", "field_contrast"):
            n = r[key]["n_required"]
            if n is None:
                return {"decision": "undetermined", "reason": f"{arm}/{key}: sigma_d undefined "
                        "(fewer than two documents with both m=1 and m=32)"}
            if n > worst:
                worst, binding = n, (arm, key)
    if worst <= max_docs:
        decision = "full_plan"
    elif worst <= corpus_docs:
        decision = "fewer_8b_settings"
    else:
        decision = "shift_to_110m_and_c2"
    return {"decision": decision, "meaning": FALLBACKS[decision], "n_required_max": int(worst),
            "binding": {"arm": binding[0], "endpoint": binding[1]},
            "max_docs": max_docs, "delta_min": delta_min, "floor": floor,
            # Sec. 5 declares an effect only above 2x the floor; if delta_min is
            # below that bar, an effect of the size N is powered for cannot be declared.
            "delta_min_above_2x_floor": bool(delta_min > 2 * abs(floor))}
