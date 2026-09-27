"""E2: the field effect -- the real winner's curse (plan v2 T6, Prop. 2).

For winners of pools of size m, with pi(l) the ordinary calibration curve and
pi(l, s) the context-aware one:

    Delta(m) = E_m[c - pi(l)]            ordinary miscalibration seen through selection
             + E_m[pi(l) - pi(l, s)]     the field term
             + E_m[pi(l, s) - a]         zero in expectation under Prop. 1

Each term is computed on the same winners with the same document weights and
the same resamples, so the three add up to Delta(m) exactly, point estimate and
every bootstrap draw alike.  The plug-in pi_hat(l) and pi_hat(l, s) come from
``c3.calibrators`` and must be out-of-sample for the documents being reported.

Intervals treat the fitted calibrators as fixed (they are not refitted inside
each resample), so they are conditional on the fit documents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import POOL_SIZES
from ..frame import AnalysisFrame, PositionFrame
from ..stats import GroupedMean, percentile_ci
from .e1 import winner_gap, winner_gap_exact


@dataclass
class FieldDecomposition:
    delta: GroupedMean        # E_m[c - a]
    delta_own: GroupedMean    # E_m[c - pi(l)]      what an own-confidence calibrator predicts
    field: GroupedMean        # E_m[pi(l) - pi(l, s)]
    residual: GroupedMean     # E_m[pi(l, s) - a]
    delta_ctx: GroupedMean    # E_m[c - pi(l, s)]   what a context-aware calibrator predicts

    def check_identity(self, atol: float = 1e-9) -> None:
        total = self.delta_own.point + self.field.point + self.residual.point
        if not np.allclose(total, self.delta.point, atol=atol, equal_nan=True):
            raise AssertionError("Delta != Delta_own + field + residual (point estimates)")
        tot_d = self.delta_own.draws + self.field.draws + self.residual.draws
        if not np.allclose(tot_d, self.delta.draws, atol=atol, equal_nan=True):
            raise AssertionError("Delta != Delta_own + field + residual (bootstrap draws)")

    def to_records(self) -> list[dict]:
        out = []
        for g, m in enumerate(self.delta.levels):
            row = {"m": int(m)}
            for name in ("delta", "delta_own", "field", "residual", "delta_ctx"):
                est: GroupedMean = getattr(self, name)
                lo, hi = est.ci()[g]
                row[name] = float(est.point[g])
                row[f"{name}_ci"] = [float(lo), float(hi)]
            row["n_docs"] = int(self.delta.n_docs[g])
            out.append(row)
        return out


def decompose(pools: AnalysisFrame | None, pos: PositionFrame, pi_own: np.ndarray,
              pi_ctx: np.ndarray, codes: np.ndarray, counts: np.ndarray,
              pos_mask: np.ndarray | None = None, exact_m1: bool = True, method: str = "mc",
              pool_sizes=POOL_SIZES, min_state_size: int = 1) -> FieldDecomposition:
    """The Prop. 2 decomposition of Delta(m), term by term.

    ``method="exact"`` uses the closed-form expectation over all pools
    (``pools`` may then be None); ``"mc"`` the drawn pools.
    """
    c, a = pos.conf, pos.agree

    def gap(q, t):
        if method == "exact":
            return winner_gap_exact(pos, q, t, codes, counts, pos_mask, pool_sizes, min_state_size)
        return winner_gap(pools, pos, q, t, codes, counts, pos_mask, exact_m1)

    out = FieldDecomposition(
        delta=gap(c, a),
        delta_own=gap(c, pi_own),
        field=gap(pi_own, pi_ctx),
        residual=gap(pi_ctx, a),
        delta_ctx=gap(c, pi_ctx),
    )
    out.check_identity()
    return out


def icc_residuals(pos: PositionFrame, pi_ctx: np.ndarray, codes: np.ndarray,
                  counts: np.ndarray, pos_mask: np.ndarray | None = None,
                  alpha: float = 0.05) -> dict:
    """ICC(1) of the residuals a - pi_hat(l, s) within masked states (T6(4)).

    Prop. 1 assumes positions are independent given the state; residual
    correlation within a state is the direct symptom of shared errors (the
    "misreading" scenario).  One-way ANOVA with unequal group sizes (the k0
    correction); the document bootstrap reweights whole documents, and each
    state inherits its document's weight.
    """
    sel = np.ones(len(pos), bool) if pos_mask is None else np.asarray(pos_mask, bool)
    p = pos.select(sel)
    r = p.agree - np.asarray(pi_ctx)[sel]
    s_idx, G = p.state_index
    n_g = np.bincount(s_idx, minlength=G).astype(np.float64)
    sum_g = np.bincount(s_idx, weights=r, minlength=G)
    ss_g = np.bincount(s_idx, weights=r * r, minlength=G)
    mean_g = sum_g / n_g
    ssw_g = ss_g - n_g * mean_g ** 2
    doc_of_state = np.zeros(G, dtype=np.int64)
    doc_of_state[s_idx] = np.searchsorted(codes, p.doc_id)

    def icc(w):  # w: (B, G) state weights
        N = w @ n_g
        Geff = w.sum(axis=1)
        grand = (w @ sum_g) / N
        ssb = (w * n_g) @ (mean_g ** 2) - N * grand ** 2
        ssw = w @ ssw_g
        msb = ssb / np.maximum(Geff - 1, 1)
        msw = ssw / np.maximum(N - Geff, 1)
        k0 = (N - (w @ (n_g ** 2)) / N) / np.maximum(Geff - 1, 1)
        return (msb - msw) / (msb + (k0 - 1) * msw)

    point = float(icc(np.ones((1, G)))[0])
    draws = icc(counts[:, doc_of_state].astype(np.float64))
    lo, hi = percentile_ci(draws, alpha)
    return {"icc": point, "ci": [lo, hi], "n_states": int(G), "n_positions": int(len(p))}
