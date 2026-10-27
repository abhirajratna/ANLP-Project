"""Logistic regression with high-cardinality fixed effects.

Eqs. 2-3 carry one intercept per masked state.  At full scale that is ~2,000
documents times several states each, i.e. tens of thousands of dummy columns
against millions of rows, which rules out a dense design matrix.

The structure rescues it: the fixed-effect block of ``X'WX`` is *diagonal*
(states partition the rows), so each IRLS step reduces by Schur complement to a
``p x p`` solve, where ``p`` is just the spline columns plus the selection terms.
Cost per iteration is O(N*p + G) with no ``G x G`` matrix ever formed.

A small ridge is applied to the fixed effects only.  States in which every
position agrees (or none do) are perfectly separated and their intercept would
diverge; the ridge keeps them finite without touching the reported coefficients,
which live in the ``p``-dimensional block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogitFit:
    beta: np.ndarray           # coefficients on the dense covariates
    alpha: np.ndarray          # per-group intercepts
    n_iter: int
    converged: bool
    loglik: float
    names: list[str]

    def coef(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])

    def as_dict(self) -> dict[str, float]:
        return {n: float(b) for n, b in zip(self.names, self.beta)}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _loglik(y: np.ndarray, cw: np.ndarray, eta: np.ndarray) -> float:
    mu = np.clip(_sigmoid(eta), 1e-12, 1 - 1e-12)
    return float(np.sum(cw * (y * np.log(mu) + (1 - y) * np.log1p(-mu))))


def _penalty(alpha: np.ndarray, beta: np.ndarray, ridge_fe: float, ridge_beta: float) -> float:
    """The ridge terms the IRLS normal equations correspond to (x 1/2)."""
    return 0.5 * (ridge_fe * float(alpha @ alpha) + ridge_beta * float(beta @ beta))


def fit_fe_logit(
    y: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    n_groups: int | None = None,
    names: list[str] | None = None,
    ridge_fe: float = 1e-3,
    ridge_beta: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-10,
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    init_beta: np.ndarray | None = None,
    init_alpha: np.ndarray | None = None,
) -> LogitFit:
    """IRLS for ``logit(P(y=1)) = alpha[groups] + X @ beta``.

    Parameters
    ----------
    y : (N,) binary outcome.
    X : (N, p) dense covariates, no intercept column (the fixed effects supply it).
    groups : (N,) integer group index in ``[0, n_groups)``.
    weights : optional (N,) non-negative case weights (used by the cluster
        bootstrap, which resamples clusters by reweighting rather than by
        materialising a new design matrix).
    offset : optional (N,) term entering the linear predictor with coefficient
        fixed at 1.  Cross-fitting uses it to carry the confidence-accuracy
        curve ``f(ell)`` estimated on the fit half into the eval half, so that
        ``f`` cannot be flexed on the same data that reports ``beta_z``.
    init_beta, init_alpha : optional warm start, ``beta`` on the original
        (unstandardised) covariate scale and one intercept per group.  Either
        may be given alone; the other starts at zero.  ``None`` (default) is the
        cold start from zero.  The estimator, tolerance, ``max_iter`` and the
        step-halving rule are unchanged; only the starting point differs.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim == 1:
        X = X[:, None]
    groups = np.asarray(groups, dtype=np.int64).ravel()
    N, p = X.shape
    if y.size != N or groups.size != N:
        raise ValueError(f"shape mismatch: y={y.size}, X={X.shape}, groups={groups.size}")
    G = int(n_groups if n_groups is not None else (groups.max() + 1 if N else 0))
    if names is None:
        names = [f"x{j}" for j in range(p)]
    if len(names) != p:
        raise ValueError(f"got {len(names)} names for {p} columns")

    # Standardise the covariate columns before fitting.  The spline basis spans
    # two orders of magnitude in scale, which left the Schur complement at
    # cond ~ 3e7 and cost most of the available precision.  Coefficients are
    # rescaled back at the end, so nothing downstream sees the change.
    col_scale = X.std(axis=0)
    col_scale[~np.isfinite(col_scale) | (col_scale <= 0)] = 1.0
    X = X / col_scale

    cw = np.ones(N) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if cw.size != N:
        raise ValueError("weights must have one entry per row")
    off = np.zeros(N) if offset is None else np.asarray(offset, dtype=np.float64).ravel()
    if off.size != N:
        raise ValueError("offset must have one entry per row")

    beta = np.zeros(p) if init_beta is None else np.array(init_beta, dtype=np.float64).ravel() * col_scale
    alpha = np.zeros(G) if init_alpha is None else np.array(init_alpha, dtype=np.float64).ravel()
    if beta.size != p:
        raise ValueError(f"init_beta has {beta.size} entries for {p} columns")
    if alpha.size != G:
        raise ValueError(f"init_alpha has {alpha.size} entries for {G} groups")
    converged = False
    prev_ll = -np.inf
    it = 0
    for it in range(1, max_iter + 1):
        eta = alpha[groups] + X @ beta + off
        mu = _sigmoid(eta)
        w = cw * np.clip(mu * (1.0 - mu), 1e-10, None)
        # Working response, formed as eta + (y-mu)/w_unweighted to keep the
        # case weights strictly in w.
        resid = (y - mu) / np.clip(mu * (1.0 - mu), 1e-10, None)
        z = eta + resid

        D = np.bincount(groups, weights=w, minlength=G) + ridge_fe
        zc = z - off  # solve for (alpha, beta) on the offset-removed working response
        rhs_a = np.bincount(groups, weights=w * zc, minlength=G)
        Xw = X * w[:, None]
        B = np.stack([np.bincount(groups, weights=Xw[:, j], minlength=G) for j in range(p)], axis=1)
        C = X.T @ Xw + (np.eye(p) * ridge_beta if ridge_beta else 0.0)
        rhs_b = Xw.T @ zc

        Dinv = 1.0 / D
        S = C - B.T @ (B * Dinv[:, None])
        rhs = rhs_b - B.T @ (Dinv * rhs_a)
        try:
            new_beta = np.linalg.solve(S, rhs)
        except np.linalg.LinAlgError:
            new_beta = np.linalg.lstsq(S, rhs, rcond=None)[0]
        new_alpha = Dinv * (rhs_a - B @ new_beta)

        # Step-halving.  Under quasi-separation (e.g. every high-confidence
        # position correct, so the spline's tail coefficients are pushed towards
        # infinity) a full Newton step can overshoot and *lower* the likelihood,
        # and the iteration then oscillates or diverges.  Accept a step only if
        # it does not decrease the (penalised) log-likelihood; otherwise halve it.
        # Well-behaved fits never trigger this, so their results are unchanged.
        ll = _loglik(y, cw, new_alpha[groups] + X @ new_beta + off)
        if np.isfinite(prev_ll):
            pen_prev = prev_ll - _penalty(alpha, beta, ridge_fe, ridge_beta)
            h = 1.0
            cand_a, cand_b = new_alpha, new_beta
            while (not np.isfinite(ll) or ll - _penalty(cand_a, cand_b, ridge_fe, ridge_beta)
                   < pen_prev - 1e-12 * (abs(pen_prev) + 1.0)) and h > 1e-10:
                h *= 0.5
                cand_b = beta + h * (new_beta - beta)
                cand_a = alpha + h * (new_alpha - alpha)
                ll = _loglik(y, cw, cand_a[groups] + X @ cand_b + off)
            new_alpha, new_beta = cand_a, cand_b

        step = float(np.max(np.abs(new_beta - beta))) if p else 0.0
        beta, alpha = new_beta, new_alpha
        if not np.all(np.isfinite(beta)):
            raise FloatingPointError("IRLS diverged: non-finite coefficients")

        # Relative criteria, not absolute.  An absolute threshold on the
        # coefficient step is unreachable once coefficients are O(10): round-off
        # keeps the step jittering near 1e-8 long after the log-likelihood has
        # gone flat, so the fit is reported as non-converged while being exact.
        rel_step = step / (1.0 + float(np.max(np.abs(beta)))) if p else 0.0
        rel_ll = abs(ll - prev_ll) / (abs(ll) + 1.0)
        prev_ll = ll
        if rel_step < tol or rel_ll < tol:
            converged = True
            break

    eta = alpha[groups] + X @ beta + off
    mu = np.clip(_sigmoid(eta), 1e-12, 1 - 1e-12)
    loglik = float(np.sum(cw * (y * np.log(mu) + (1 - y) * np.log1p(-mu))))
    return LogitFit(beta=beta / col_scale, alpha=alpha, n_iter=it,
                    converged=converged, loglik=loglik, names=list(names))
