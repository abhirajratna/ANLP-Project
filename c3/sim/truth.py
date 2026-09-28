"""The exact calibration functions of the Gaussian simulation (``c3.sim.positions``).

T0 asks whether the field term grows with m *in the model*, which is a question
about the true pi(l, s) and pi(l), not about how well a fitted calibrator
recovers them.  In the calibrated, uniform and context scenarios both are
available in closed form up to one-dimensional integrals:

    state s:   d ~ N(0, 1),  mean_s = mu - shift * d,  sig_s = sigma * exp(dn * d)
    position:  l* ~ N(mean_s, sd^2),  l = l* + sig_s * eps,  a ~ Bernoulli(sigmoid(l*))

Given (l, s), l* is Gaussian with

    v = sd^2 sig_s^2 / (sd^2 + sig_s^2),   m = mean_s + sd^2 / (sd^2 + sig_s^2) * (l - mean_s)

so pi(l, s) = E[sigmoid(m + sqrt(v) Z)], a Gauss-Hermite sum.  pi(l) mixes pi(l, s)
over states with weights p(l | d) phi(d), p(l | d) = N(l; mean_d, sd^2 + sig_d^2),
another Gauss-Hermite sum over d.  Every state has the same number of positions,
so this position-level mixture is also the pool-weighted population the endpoints
average over, and E[pi(l) - pi(l, s)] = 0 at m = 1 exactly.

The misreading scenario has no such form (the hidden event is not a function of
anything observed), so it has no truth here.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

TRUTH_SCENARIOS = ("calibrated", "uniform", "context")


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _nodes(n: int):
    x, w = hermegauss(n)
    return x, w / w.sum()          # expectations under N(0, 1)


def state_params(scenario: str, difficulty, mu: float = 0.5, sigma: float = 1.0,
                 difficulty_shift: float = 1.0, difficulty_noise: float = 0.6):
    """(mean_s, sig_s) of the field in each state, as ``simulate_positions`` sets them."""
    d = np.asarray(difficulty, dtype=np.float64)
    if scenario == "context":
        return mu - difficulty_shift * d, sigma * np.exp(difficulty_noise * d)
    if scenario == "uniform":
        return np.full_like(d, mu), np.full_like(d, sigma)
    if scenario == "calibrated":
        return np.full_like(d, mu), np.zeros_like(d)
    raise ValueError(f"no closed-form truth for scenario {scenario!r}; known: {TRUTH_SCENARIOS}")


def pi_given_state(ell, mean_s, sig_s, sd: float = 1.5, n_nodes: int = 48) -> np.ndarray:
    """pi(l, s) = Pr(a = 1 | l, s), elementwise over positions."""
    ell = np.asarray(ell, dtype=np.float64)
    mean_s = np.broadcast_to(np.asarray(mean_s, dtype=np.float64), ell.shape)
    sig2 = np.broadcast_to(np.asarray(sig_s, dtype=np.float64) ** 2, ell.shape)
    shrink = sd ** 2 / (sd ** 2 + sig2)
    m = mean_s + shrink * (ell - mean_s)
    v = sig2 * shrink                      # = sd^2 sig^2 / (sd^2 + sig^2)
    x, w = _nodes(n_nodes)
    return _sigmoid(m[..., None] + np.sqrt(v)[..., None] * x) @ w


def pi_marginal(ell, scenario: str, sd: float = 1.5, n_state_nodes: int = 64,
                n_nodes: int = 48, grid_size: int = 4001, **params) -> np.ndarray:
    """pi(l) = Pr(a = 1 | l), the ordinary calibration curve, mixed over states.

    Computed on a grid spanning the data and interpolated: it is a smooth
    function of one variable, and evaluating the double sum per position would
    cost ~10^9 sigmoids at the T0 sizes.
    """
    ell = np.asarray(ell, dtype=np.float64)
    lo, hi = float(ell.min()) - 1.0, float(ell.max()) + 1.0
    grid = np.linspace(lo, hi, grid_size)
    d, wd = _nodes(n_state_nodes)
    mean_d, sig_d = state_params(scenario, d, **params)
    var_l = sd ** 2 + sig_d ** 2                                      # (S,)
    logp = -0.5 * (grid[:, None] - mean_d) ** 2 / var_l - 0.5 * np.log(var_l)
    wts = np.exp(logp - logp.max(axis=1, keepdims=True)) * wd         # (G, S)
    pis = np.stack([pi_given_state(grid, mean_d[j], sig_d[j], sd, n_nodes)
                    for j in range(d.size)], axis=1)                  # (G, S)
    curve = (wts * pis).sum(axis=1) / wts.sum(axis=1)
    return np.interp(ell, grid, curve)


def true_calibrators(df, scenario: str, sd: float = 1.5, **params) -> tuple[np.ndarray, np.ndarray]:
    """(pi(l), pi(l, s)) for every row of a ``simulate_positions`` frame."""
    mean_s, sig_s = state_params(scenario, df["difficulty"].to_numpy(), **params)
    ell = df["ell"].to_numpy(np.float64)
    return pi_marginal(ell, scenario, sd, **params), pi_given_state(ell, mean_s, sig_s, sd)
