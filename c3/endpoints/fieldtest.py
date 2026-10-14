"""B: a proper-score test of the field effect over every masked position.

E2 (plan v2 T6) asks whether the state a token sits in carries information about
its correctness beyond its own confidence, but it sees that only through pool
winners and their 0/1 correctness.  The quantity a context-aware recalibration
(C3a) exploits can be tested directly, and on every masked position: does
pi_hat(l, s) predict correctness better than pi_hat(l)?  A proper scoring rule
answers it.  Both calibrators are fitted on one set of documents and scored on
another, and the held-out log-loss (and Brier score) gain of the context model is
averaged within documents first and bootstrapped over documents, as every
endpoint is (``c3.stats``).

The primary pair is the *same model family*: a spline-only logistic in l against
the same spline plus the field features, so the difference isolates the
features.  Isotonic pi_hat(l), the calibrator E2 uses, is reported as a second
row; it differs from the logistic in family as well as in features, and it can
predict exactly 0 or 1, so log-loss clips probabilities to [eps, 1 - eps].
"""

from __future__ import annotations

import numpy as np

from .. import DEFAULT_KNOTS
from ..calibrators import ContextLogistic, IsotonicOwn, _weights, pool_weights
from ..frame import FIELD_FEATURES, PositionFrame
from ..stats import boot_pvalue, declare, grouped_doc_mean, percentile_ci

EPS = 1e-6
DECODE_FEATURES = ("median_ell", "mask_rate", "log_n_masked")
"""Field features a decoder can compute at inference time (``step`` is the
trajectory index of the teacher-forced traces and has no counterpart when
decoding a span)."""


def clip_prob(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def log_loss(p: np.ndarray, agree: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Per-position negative log-likelihood of correctness, in nats."""
    q = clip_prob(p, eps)
    a = np.asarray(agree, dtype=np.float64)
    return -(a * np.log(q) + (1.0 - a) * np.log1p(-q))


def brier(p: np.ndarray, agree: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=np.float64) - np.asarray(agree, dtype=np.float64)) ** 2


_LOSSES = {"log_loss": lambda p, a, eps: log_loss(p, a, eps),
           "brier": lambda p, a, eps: brier(p, a)}


def paired_score_test(p_base: np.ndarray, p_alt: np.ndarray, agree: np.ndarray,
                      doc_idx: np.ndarray, n_docs: int, counts: np.ndarray,
                      weights: np.ndarray | None = None, metric: str = "log_loss",
                      eps: float = EPS, floor: float = 0.0, alpha: float = 0.05) -> dict:
    """Held-out score of ``p_base`` minus that of ``p_alt`` (positive: alt is better).

    Each document's score is the (weighted) mean over its positions; documents
    count equally, and resample b reuses ``counts[b]`` for both models, so the
    difference is paired.  ``declared`` follows plan v2 Sec. 5 (the interval
    excludes 0 and the gain exceeds twice the floor); ``helps`` additionally
    requires the gain to be positive.
    """
    if metric not in _LOSSES:
        raise ValueError(f"metric must be one of {sorted(_LOSSES)}")
    loss = _LOSSES[metric]
    zeros = np.zeros(np.asarray(agree).size, dtype=np.int64)
    g_base = grouped_doc_mean(loss(p_base, agree, eps), doc_idx, n_docs, zeros, counts,
                              alpha, weights)
    g_alt = grouped_doc_mean(loss(p_alt, agree, eps), doc_idx, n_docs, zeros, counts,
                             alpha, weights)
    point = float(g_base.point[0] - g_alt.point[0])
    draws = g_base.draws[:, 0] - g_alt.draws[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_draws = draws / g_base.draws[:, 0]
    per_doc = g_base.per_doc[:, 0] - g_alt.per_doc[:, 0]
    per_doc = per_doc[np.isfinite(per_doc)]
    sigma_d = float(np.std(per_doc, ddof=1)) if per_doc.size > 1 else float("nan")
    ci = percentile_ci(draws, alpha)
    is_declared = declare(point, ci, floor)
    return {
        "metric": metric,
        "base": float(g_base.point[0]), "alt": float(g_alt.point[0]),
        "gain": point, "ci": list(ci), "p_boot": boot_pvalue(draws),
        "rel_gain": point / float(g_base.point[0]) if g_base.point[0] else float("nan"),
        "rel_ci": list(percentile_ci(rel_draws, alpha)),
        "declared": is_declared, "helps": bool(is_declared and point > 0),
        "sigma_d": sigma_d,
        # Smallest gain detectable at 5% two-sided with 80% power on these documents.
        "mde": 2.8 * sigma_d / np.sqrt(per_doc.size) if per_doc.size > 1 else float("nan"),
        "n_docs": int(per_doc.size), "n_positions": int(np.asarray(agree).size),
        "floor": float(floor),
    }


def field_effect_test(pos: PositionFrame, fit_mask: np.ndarray, report_mask: np.ndarray,
                      codes: np.ndarray, counts: np.ndarray, features=FIELD_FEATURES,
                      n_knots: int = DEFAULT_KNOTS, interactions: bool = True,
                      ridge: float = 1e-4, weighting: str = "pool", eps: float = EPS,
                      floor_ll: float = 0.0, floor_brier: float = 0.0,
                      alpha: float = 0.05) -> dict:
    """Fit own and context calibrators on ``fit_mask``, score them on ``report_mask``.

    ``codes`` are the sorted document ids indexing the columns of ``counts``; every
    reported document must be among them.  Returns the log-loss and Brier
    contrasts for the same-family pair (``spline_vs_context``) and for isotonic
    against the context model (``isotonic_vs_context``), plus fit diagnostics.
    """
    fit_mask = np.asarray(fit_mask, dtype=bool)
    report_mask = np.asarray(report_mask, dtype=bool)
    if (fit_mask & report_mask).any():
        raise ValueError("fit and report positions overlap; the test must be out of sample")
    tr, pr = pos.select(fit_mask), pos.select(report_mask)
    if len(tr) == 0 or len(pr) == 0:
        raise ValueError("empty fit or report selection")
    missing = np.setdiff1d(np.unique(pr.doc_id), codes)
    if missing.size:
        raise ValueError(f"{missing.size} reported documents are not in the bootstrap codes")

    w_fit = _weights(tr, weighting)
    own = ContextLogistic(features=(), n_knots=n_knots, ridge=ridge,
                          interactions=False).fit(tr, w_fit)
    ctx = ContextLogistic(features=tuple(features), n_knots=n_knots, ridge=ridge,
                          interactions=interactions).fit(tr, w_fit)
    iso = IsotonicOwn().fit(tr.ell, tr.agree, w_fit)

    p_own, p_ctx, p_iso = own.predict(pr), ctx.predict(pr), iso.predict(pr.ell)
    doc_idx = np.searchsorted(codes, pr.doc_id)
    w_rep = pool_weights(pr) if weighting == "pool" else None
    common = dict(agree=pr.agree, doc_idx=doc_idx, n_docs=codes.size, counts=counts,
                  weights=w_rep, eps=eps, alpha=alpha)
    out = {"features": list(ctx.used), "requested_features": list(features),
           "n_knots": n_knots, "interactions": interactions, "ridge": ridge,
           "weighting": weighting, "eps": eps,
           "n_fit_positions": int(len(tr)), "n_report_positions": int(len(pr)),
           "own_converged": own.converged, "context_converged": ctx.converged,
           "isotonic_clipped_share": float(np.mean((p_iso <= eps) | (p_iso >= 1 - eps)))}
    for name, base in (("spline_vs_context", p_own), ("isotonic_vs_context", p_iso)):
        out[name] = {
            "log_loss": paired_score_test(base, p_ctx, metric="log_loss", floor=floor_ll, **common),
            "brier": paired_score_test(base, p_ctx, metric="brier", floor=floor_brier, **common),
        }
    return out
