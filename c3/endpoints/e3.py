"""E3: are the states a decoder visits less calibrated than random ones? (plan v2 T7)

A confidence decoder fills easy positions first, so its later states hold the
hard ones, bunched in mask patterns that random-mask training rarely shows.
Comparing raw accuracy would only restate that decoder states are harder; the
comparison is made *at the same confidence*, which removes difficulty that the
confidence already reports.  What is left is whether a given confidence means
less in a decoder-visited state.

Model, on the positions of one decoder arm and one reference arm::

    Pr(a = 1) = sigma( alpha_b + f(ell) + type * (g_0 + f'(ell)) )

``type`` = 1 in decoder states, ``alpha_b`` an intercept per mask-rate bin
(0.1 wide, centred on 0.1, ..., 0.9; rates outside [0.05, 0.95) are dropped),
``f`` and ``f'`` restricted cubic splines on the same knots.  The reference arm
is either the matched uniform states (each the size of a decoder state, so rates
match state by state) or the uniform grid, whose rate-bin mix is reweighted to
the decoder arm's.  The type terms (g_0, f') are what is tested.

**Scale.** A full arm is tens of millions of positions, and the document
bootstrap refits the model on every resample.  Positions are therefore reduced
to per-document sufficient statistics over cells (type x rate bin x one of
``n_ell_bins`` equal-mass ell bins): weight, weighted agreement, weighted ell.
A resample is then one matrix product and a grouped logistic fit on a few
thousand cells.  The cell's mean ell stands in for its positions' ell; with 100
equal-mass bins the spline varies negligibly within a cell.

Weights: within each arm, every document carries total weight 1, every state
in it the same share, and positions in a state equal shares (the pool weights
of ``c3.calibrators.pool_weights``), so the arms are weighted alike.

Reported, all on one set of document resamples:

* ``type_effect``: E over decoder positions of pi_ref(l, b) - pi_dec(l, b), in
  probability units.  Positive: at the same confidence, a decoder-state token is
  right less often (decoder states are more overconfident).
* ``type_test``: the joint test of the type terms (g_0, f') = 0, as a sup-t test
  that the difference curve is zero on its whole grid (see :func:`sup_t`).
* ``difference_curve``: pi_ref(l) - pi_dec(l) on a grid of ell, with pointwise
  CIs and a simultaneous 95% band.
* ``by_rate``: the type effect within each rate bin, from separate fits
  (exploratory: the plan expects it largest at high mask rates).
* ``reliability``: equal-mass reliability bins of both arms on shared confidence
  bins, and their per-bin accuracy difference, for side-by-side diagrams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import DEFAULT_KNOTS
from ..calibrators import pool_weights
from ..frame import PositionFrame
from ..logit import _sigmoid, fit_fe_logit
from ..splines import Spline, default_knots
from ..stats import boot_pvalue, percentile_ci

RATE_EDGES = np.round(np.arange(0.05, 1.0, 0.1), 2)
"""Mask-rate bin edges: [0.05, 0.15), ..., [0.85, 0.95), centred on the uniform grid."""
RATE_CENTRES = np.round((RATE_EDGES[:-1] + RATE_EDGES[1:]) / 2, 2)


def rate_bin(rate: np.ndarray) -> np.ndarray:
    """Bin index 0..8 of each mask rate, -1 outside [0.05, 0.95)."""
    r = np.asarray(rate, dtype=np.float64)
    b = np.digitize(r, RATE_EDGES) - 1
    b[(r < RATE_EDGES[0]) | (r >= RATE_EDGES[-1])] = -1
    return b


@dataclass
class E3Nuisance:
    """Choices frozen on the fit half: spline knots, ell-bin edges, confidence bins."""

    knots: np.ndarray
    ell_edges: np.ndarray       # interior edges of the equal-mass ell bins
    conf_edges: np.ndarray      # interior edges of the reliability-diagram bins

    @classmethod
    def from_positions(cls, ell: np.ndarray, conf: np.ndarray, n_knots: int = DEFAULT_KNOTS,
                       n_ell_bins: int = 100, n_conf_bins: int = 15) -> "E3Nuisance":
        ell = np.asarray(ell, dtype=np.float64)
        conf = np.asarray(conf, dtype=np.float64)
        q = np.linspace(0, 1, n_ell_bins + 1)[1:-1]
        qc = np.linspace(0, 1, n_conf_bins + 1)[1:-1]
        return cls(knots=default_knots(ell, n_knots), ell_edges=np.unique(np.quantile(ell, q)),
                   conf_edges=np.unique(np.quantile(conf, qc)))

    def to_dict(self) -> dict:
        return {"knots": self.knots.tolist(), "n_ell_bins": int(self.ell_edges.size + 1),
                "conf_edges": self.conf_edges.tolist()}


def _weights_by_arm(dec: PositionFrame, ref: PositionFrame, match_rates: bool):
    """Pool weights in each arm; optionally the reference reweighted to the decoder's rate mix."""
    wd, wr = pool_weights(dec), pool_weights(ref)
    bd, br = rate_bin(dec.mask_rate), rate_bin(ref.mask_rate)
    wd = np.where(bd >= 0, wd, 0.0)
    wr = np.where(br >= 0, wr, 0.0)
    factors = np.ones(RATE_CENTRES.size)
    if match_rates:
        for b in range(RATE_CENTRES.size):
            sd, sr = wd[bd == b].sum(), wr[br == b].sum()
            factors[b] = sd / sr if sr > 0 else 0.0
        wr = wr * np.where(br >= 0, factors[np.maximum(br, 0)], 0.0)
    return wd, wr, bd, br, factors


@dataclass
class _Cells:
    """Per-document sums over cells (type, rate bin, ell bin)."""

    S_w: np.ndarray      # (n_docs, n_cells)
    S_wa: np.ndarray
    S_wl: np.ndarray
    t: np.ndarray        # (n_cells,) type of each cell
    b: np.ndarray        # (n_cells,) rate bin of each cell

    def totals(self, C: np.ndarray):
        """Cell totals under document weights ``C`` (n_resamples, n_docs)."""
        return C @ self.S_w, C @ self.S_wa, C @ self.S_wl


def _cells(dec, ref, wd, wr, bd, br, nu: E3Nuisance, codes, kind_filter=None) -> _Cells:
    J = nu.ell_edges.size + 1
    B = RATE_CENTRES.size
    n_cells = 2 * B * J
    n_docs = codes.size
    S = [np.zeros(n_docs * n_cells) for _ in range(3)]
    for t, (p, w, b) in enumerate(((ref, wr, br), (dec, wd, bd))):
        keep = (b >= 0) & (w > 0)
        if kind_filter is not None:
            keep &= kind_filter(t, b)
        j = np.searchsorted(nu.ell_edges, p.ell[keep], side="right")
        cell = (t * B + b[keep]) * J + j
        di = np.searchsorted(codes, p.doc_id[keep])
        flat = di * n_cells + cell
        ww = w[keep]
        S[0] += np.bincount(flat, weights=ww, minlength=n_docs * n_cells)
        S[1] += np.bincount(flat, weights=ww * p.agree[keep], minlength=n_docs * n_cells)
        S[2] += np.bincount(flat, weights=ww * p.ell[keep], minlength=n_docs * n_cells)
    t_of = np.repeat(np.arange(2), B * J)
    b_of = np.tile(np.repeat(np.arange(B), J), 2)
    return _Cells(*(s.reshape(n_docs, n_cells) for s in S), t=t_of, b=b_of)


def _fit_cells(w, wa, wl, t, b, spline: Spline, ridge: float, rate_fe: bool):
    """Grouped logistic on cells with positive weight; returns (fit, design pieces)."""
    keep = w > 0
    w, t, b = w[keep], t[keep], b[keep]
    y = wa[keep] / w
    ell = wl[keep] / w
    F = spline.transform(ell)
    X = np.column_stack([F, t[:, None], F * t[:, None]])
    if rate_fe:
        _, g = np.unique(b, return_inverse=True)
        bins_present = np.unique(b)
    else:
        g = np.zeros(b.size, dtype=np.int64)
        bins_present = np.array([-1])
    fit = fit_fe_logit(y, X, g, n_groups=int(g.max()) + 1, weights=w * (w.size / w.sum()),
                       ridge_fe=1e-8, ridge_beta=ridge)
    alpha_by_bin = {int(bb): float(fit.alpha[i]) for i, bb in enumerate(bins_present)}
    return fit, keep, F, t, b, w, alpha_by_bin


def _effects(fit, F, t, b, w, alpha_by_bin, rate_fe: bool, n_f: int):
    """Average type effect over decoder cells, overall and by rate bin."""
    beta = fit.beta
    a = np.array([alpha_by_bin[int(x)] if rate_fe else alpha_by_bin[-1] for x in b])
    eta_ref = a + F @ beta[:n_f]
    eta_dec = eta_ref + beta[n_f] + F @ beta[n_f + 1:]
    diff = _sigmoid(eta_ref) - _sigmoid(eta_dec)
    dec = t == 1
    overall = float(np.sum(w[dec] * diff[dec]) / np.sum(w[dec]))
    return overall


def e3_compare(dec: PositionFrame, ref: PositionFrame, codes: np.ndarray, counts: np.ndarray,
               nuisance: E3Nuisance, match_rates: bool = False, rate_fe: bool = True,
               grid_size: int = 25, ridge: float = 1e-6, chunk: int = 500,
               alpha: float = 0.05, by_rate: bool = True) -> dict:
    """The E3 comparison of one decoder arm against one reference arm.

    ``dec`` and ``ref`` hold only the positions to be reported (e.g. the report
    half); ``nuisance`` must have been built on other documents (the fit half).
    ``codes``/``counts`` are the run's document index and resample matrix.
    """
    if len(dec) == 0 or len(ref) == 0:
        raise ValueError("both arms need positions")
    wd, wr, bd, br, factors = _weights_by_arm(dec, ref, match_rates)
    cells = _cells(dec, ref, wd, wr, bd, br, nuisance, codes)
    spline = Spline(nuisance.knots)
    n_f = spline.n_columns
    ones = np.ones((1, codes.size))
    grid = np.quantile(dec.ell[wd > 0], np.linspace(0.025, 0.975, grid_size))
    Fg = spline.transform(grid)

    def one(Cw, Cwa, Cwl):
        fit, keep, F, t, b, w, abin = _fit_cells(Cw, Cwa, Cwl, cells.t, cells.b, spline,
                                                 ridge, rate_fe)
        eff = _effects(fit, F, t, b, w, abin, rate_fe, n_f)
        # The curve at the decoder cells' weight-averaged intercept.
        dec_c = t == 1
        a_bar = float(np.average([abin[int(x)] if rate_fe else abin[-1] for x in b[dec_c]],
                                 weights=w[dec_c]))
        base = a_bar + Fg @ fit.beta[:n_f]
        curve = _sigmoid(base) - _sigmoid(base + fit.beta[n_f] + Fg @ fit.beta[n_f + 1:])
        return eff, curve, fit.beta[n_f:].copy(), fit.converged

    pw, pwa, pwl = cells.totals(ones)
    p_eff, p_curve, p_theta, p_conv = one(pw[0], pwa[0], pwl[0])

    effs, curves, thetas, n_fail = [], [], [], 0
    for s in range(0, counts.shape[0], chunk):
        Cw, Cwa, Cwl = cells.totals(counts[s:s + chunk].astype(np.float64))
        for i in range(Cw.shape[0]):
            try:
                e, c, th, _ = one(Cw[i], Cwa[i], Cwl[i])
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                n_fail += 1
                continue
            effs.append(e); curves.append(c); thetas.append(th)
    effs = np.asarray(effs)
    curves = np.vstack(curves) if curves else np.empty((0, grid.size))
    thetas = np.vstack(thetas) if thetas else np.empty((0, p_theta.size))

    res = {
        "type_effect": {"value": p_eff, "ci": list(percentile_ci(effs, alpha)),
                        "p_boot": boot_pvalue(effs),
                        "meaning": "E_decoder positions[pi_ref(l) - pi_dec(l)]; > 0: at equal "
                                   "confidence decoder-state tokens are right less often"},
        "type_main_effect": {"value": float(p_theta[0]),
                             "ci": list(percentile_ci(thetas[:, 0], alpha)) if len(thetas) else
                             [float("nan")] * 2},
        "difference_curve": {"ell": grid.tolist(), "conf": _sigmoid(grid).tolist(),
                             "value": p_curve.tolist(),
                             "ci": [list(percentile_ci(curves[:, j], alpha))
                                    for j in range(grid.size)],
                             "simultaneous": sup_t(p_curve, curves, alpha)},
        "type_test": {k: v for k, v in sup_t(p_curve, curves, alpha).items() if k != "band"},
        "reliability": reliability_pair(dec, ref, wd, wr, nuisance, codes, counts, alpha),
        "n_boot_ok": int(effs.size), "n_boot_failed": int(n_fail),
        "point_converged": bool(p_conv),
        "n_positions": {"decoder": int((wd > 0).sum()), "reference": int((wr > 0).sum())},
        "n_docs": {"decoder": int(np.unique(dec.doc_id[wd > 0]).size),
                   "reference": int(np.unique(ref.doc_id[wr > 0]).size)},
        "rate_fe": bool(rate_fe), "match_rates": bool(match_rates),
        "rate_weight_factors": factors.tolist(),
    }
    if by_rate:
        res["by_rate"] = _by_rate(cells, spline, counts, ridge, chunk, alpha)
    return res


def _by_rate(cells: _Cells, spline: Spline, counts, ridge, chunk, alpha) -> list[dict]:
    """The type effect within each rate bin, from a separate fit per bin."""
    n_f = spline.n_columns
    out = []
    ones = np.ones((1, cells.S_w.shape[0]))
    for b in range(RATE_CENTRES.size):
        sel = cells.b == b
        sub = _Cells(cells.S_w[:, sel], cells.S_wa[:, sel], cells.S_wl[:, sel],
                     cells.t[sel], cells.b[sel])
        pw, pwa, pwl = sub.totals(ones)
        have = [np.sum(pw[0][sub.t == tt]) > 0 for tt in (0, 1)]
        if not all(have):
            out.append({"rate": float(RATE_CENTRES[b]), "value": None, "ci": [None, None]})
            continue

        def one(w, wa, wl):
            fit, keep, F, t, bb, ww, abin = _fit_cells(w, wa, wl, sub.t, sub.b, spline,
                                                       ridge, False)
            return _effects(fit, F, t, bb, ww, abin, False, n_f)

        try:
            point = one(pw[0], pwa[0], pwl[0])
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            out.append({"rate": float(RATE_CENTRES[b]), "value": None, "ci": [None, None]})
            continue
        draws = []
        for s in range(0, counts.shape[0], chunk):
            Cw, Cwa, Cwl = sub.totals(counts[s:s + chunk].astype(np.float64))
            for i in range(Cw.shape[0]):
                if not (np.sum(Cw[i][sub.t == 0]) > 0 and np.sum(Cw[i][sub.t == 1]) > 0):
                    continue
                try:
                    draws.append(one(Cw[i], Cwa[i], Cwl[i]))
                except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                    continue
        draws = np.asarray(draws)
        out.append({"rate": float(RATE_CENTRES[b]), "value": point,
                    "ci": list(percentile_ci(draws, alpha)), "p_boot": boot_pvalue(draws),
                    "n_boot_ok": int(draws.size)})
    return out


def sup_t(point: np.ndarray, draws: np.ndarray, alpha: float = 0.05) -> dict:
    """Simultaneous band and joint test that a curve is zero everywhere on its grid.

    T = max_j |curve_j| / sd_j, referred to the bootstrap distribution of
    max_j |curve*_j - curve_j| / sd_j (the sup-t band of Montiel Olea and
    Plagborg-Moller, 2019).  This is the test of the type terms: (g_0, f') = 0
    if and only if the difference curve is zero.  A Wald test on the
    coefficients themselves needs their covariance, and the bootstrap one is
    near-singular for collinear spline columns: on simulated nulls it rejected
    far too often, which this does not.

    Small samples: on ``c3.sim.trajectories`` 'calibrated' it rejected 5 of 20
    nulls at 40 report documents and 0 of 12 at 150, so treat it as unreliable
    below ~100 documents (the pilot); the main arms have ~1,000.
    """
    d = draws[np.all(np.isfinite(draws), axis=1)] if draws.size else draws
    if d.shape[0] < 20:
        return {"stat": None, "p": None, "band": None, "note": "too few resamples"}
    sd = d.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)
    t_obs = float(np.max(np.abs(point) / sd))
    t_boot = np.max(np.abs(d - point) / sd, axis=1)
    crit = float(np.quantile(t_boot, 1 - alpha))
    p = float((1 + np.sum(t_boot >= t_obs)) / (t_boot.size + 1))
    band = np.column_stack([point - crit * sd, point + crit * sd])
    return {"stat": t_obs, "p": p, "critical": crit,
            "band": [[_f(lo), _f(hi)] for lo, hi in band],
            "test": "sup-t: the difference curve (type terms) is zero on the whole grid"}


def reliability_pair(dec: PositionFrame, ref: PositionFrame, wd: np.ndarray, wr: np.ndarray,
                     nuisance: E3Nuisance, codes: np.ndarray, counts: np.ndarray,
                     alpha: float = 0.05) -> dict:
    """Weighted reliability bins of both arms on shared confidence bins, with the
    per-bin accuracy difference (reference - decoder) and document-bootstrap CIs."""
    nb = nuisance.conf_edges.size + 1
    n_docs = codes.size
    out = {}
    sums = {}
    for name, p, w in (("decoder", dec, wd), ("reference", ref, wr)):
        keep = w > 0
        bi = np.searchsorted(nuisance.conf_edges, p.conf[keep], side="right")
        di = np.searchsorted(codes, p.doc_id[keep])
        flat = di * nb + bi
        ww = w[keep]
        Sw = np.bincount(flat, weights=ww, minlength=n_docs * nb).reshape(n_docs, nb)
        Sc = np.bincount(flat, weights=ww * p.conf[keep], minlength=n_docs * nb).reshape(n_docs, nb)
        Sa = np.bincount(flat, weights=ww * p.agree[keep], minlength=n_docs * nb).reshape(n_docs, nb)
        sums[name] = (Sw, Sc, Sa)
        tw, tc, ta = Sw.sum(0), Sc.sum(0), Sa.sum(0)
        C = counts.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            acc_draws = (C @ Sa) / (C @ Sw)
            conf_pt, acc_pt = tc / tw, ta / tw
            gap = np.nansum(tw * np.abs(conf_pt - acc_pt)) / tw.sum()
            gap_draws = np.nansum((C @ Sw) * np.abs((C @ Sc) / (C @ Sw) - acc_draws), axis=1) \
                / (C @ Sw).sum(axis=1)
        out[name] = {"bins": [{"conf": _f(conf_pt[j]), "acc": _f(acc_pt[j]),
                               "acc_ci": list(percentile_ci(acc_draws[:, j], alpha)),
                               "weight": float(tw[j])} for j in range(nb)],
                     "weighted_ece": float(gap), "weighted_ece_ci": list(percentile_ci(gap_draws, alpha))}
    (Swd, _, Sad), (Swr, _, Sar) = sums["decoder"], sums["reference"]
    C = counts.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff_pt = Sar.sum(0) / Swr.sum(0) - Sad.sum(0) / Swd.sum(0)
        diff_draws = (C @ Sar) / (C @ Swr) - (C @ Sad) / (C @ Swd)
    out["difference"] = [{"bin": j, "value": _f(diff_pt[j]),
                          "ci": list(percentile_ci(diff_draws[:, j], alpha))} for j in range(nb)]
    out["note"] = ("shared equal-mass confidence bins (frozen on the fit half); pool weights, "
                   "reference rate mix matched when requested; difference = acc_ref - acc_dec")
    return out


def _f(x) -> float | None:
    return float(x) if np.isfinite(x) else None


def contrast_of_contrasts(g_dec, g_ref, hi: int = 32, lo: int = 1, alpha: float = 0.05) -> dict:
    """[est(hi) - est(lo)] in the decoder arm minus the same in the reference arm.

    Both ``GroupedMean`` objects must share the run's resample matrix, so their
    draws pair up.
    """
    p1, d1 = g_dec.contrast(hi, lo)
    p2, d2 = g_ref.contrast(hi, lo)
    return {"value": p1 - p2, "ci": list(percentile_ci(d1 - d2, alpha)),
            "p_boot": boot_pvalue(d1 - d2), "decoder": p1, "reference": p2}
