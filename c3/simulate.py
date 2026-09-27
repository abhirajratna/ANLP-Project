"""Synthetic generators with known truth, for validating the estimator.

Proposal week 3-4 requires the Eq. 2-3 machinery to be checked on a toy where
the answer is known before it is pointed at a real model.  Four data-generating
processes are provided, each pinning a different part of the claim:

``homoskedastic``
    ``ell = ell* + eps`` with a single noise scale.  Correctness depends on the
    *true* log-odds, so reported confidence is noisy.  ``beta_z = 0`` is the
    null the audit is built to test.

``heteroskedastic``
    Same, but the per-position noise scale is lognormal with dispersion ``tau``.
    The proposal (Sec. 5.1) expected power here; plan v2's Prop. 1 shows that
    position-level noise, however unequal, leaves beta_z with state FE at zero
    and moves Delta(m) only through ordinary miscalibration.  None of these DGPs
    has a nonzero field term: the state latent shifts correctness, never the
    confidences.  The v2 scenarios live in ``c3.sim``.

``direct_rank``
    Correctness is generated to depend on ``z_{r,m}`` with a known coefficient
    ``delta``.  This is not a realistic generative story -- it is the positive
    control that shows the estimator recovers a planted coefficient.

``curvature``
    Homoskedastic, but the confidence-accuracy relation is strongly non-linear.
    Used to show what a linear (rather than spline) control for ``f(ell)`` does
    to ``beta_z``, which is the reason Eq. 2 specifies a flexible control.

In every DGP ``agree_i`` is drawn **once per position** and reused across every
pool that position falls into (except in ``direct_rank``, where by construction
the outcome depends on the pool).  Drawing it per row instead would destroy the
within-document dependence the clustered bootstrap exists to handle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import POOL_SIZES
from .design import draw_pools
from .frame import AnalysisFrame
from .order_stats import z_vector
from .rng import generator


@dataclass
class Truth:
    dgp: str
    beta_z: float          # the coefficient the estimator should recover
    tau: float             # dispersion of log noise scale (0 = homoskedastic)
    sigma: float           # base noise scale on the log-odds scale
    n_docs: int
    n_positions: int

    def __str__(self) -> str:
        return (f"DGP={self.dgp} beta_z_true={self.beta_z:+.3f} tau={self.tau} "
                f"sigma={self.sigma} docs={self.n_docs} pos/state={self.n_positions}")


def simulate(
    n_docs: int = 200,
    n_states_per_doc: int = 4,
    n_positions: int = 64,
    dgp: str = "heteroskedastic",
    sigma: float = 1.0,
    tau: float = 0.8,
    delta: float = -0.30,
    positional_noise: bool = False,
    mu_ell: float = 1.0,
    sd_ell: float = 1.5,
    sd_state: float = 0.5,
    pool_sizes=POOL_SIZES,
    n_pools_per_size: int = 16,
    retain: int | str = "all",
    seed: int = 0,
    pool_stream: str = "per_m",
) -> tuple[AnalysisFrame, Truth]:
    """Generate an :class:`AnalysisFrame` together with the truth that made it."""
    if dgp not in {"homoskedastic", "heteroskedastic", "direct_rank", "curvature"}:
        raise ValueError(f"unknown dgp {dgp!r}")
    if dgp == "homoskedastic":
        tau = 0.0

    rng = generator(seed, "sim", dgp)
    docs, states, poss, ells, agrees, ms, rs = [], [], [], [], [], [], []

    for d in range(n_docs):
        for s in range(n_states_per_doc):
            state_id = d * n_states_per_doc + s
            alpha = rng.normal(0.0, sd_state)
            ell_star = rng.normal(mu_ell, sd_ell, n_positions)

            # Per-position noise scale: lognormal, mean of the variance held at
            # sigma^2 so that tau changes dispersion without changing overall scale.
            # With positional_noise the scale is a smooth function of position, so
            # it tracks an observable and is estimable; otherwise it is idiosyncratic
            # and no stratifier can recover it (see c3.dispersion).
            if positional_noise:
                u = np.arange(n_positions) / max(1, n_positions - 1)
                eta = np.sqrt(2.0) * np.sin(2 * np.pi * u)
            else:
                eta = rng.normal(size=n_positions)
            sig_i = sigma * np.exp(tau * eta - 0.5 * tau ** 2)
            ell = ell_star + rng.normal(0.0, 1.0, n_positions) * sig_i

            pd = draw_pools(ell, pool_sizes, n_pools_per_size, seed=seed,
                            state_key=f"{dgp}/{state_id}", retain=retain,
                            stream=pool_stream)
            if len(pd) == 0:
                continue

            if dgp == "direct_rank":
                z = z_vector(pd.r, pd.m)
                p = 1.0 / (1.0 + np.exp(-(alpha + ell_star[pd.state_row] + delta * z)))
                a = (rng.random(p.size) < p).astype(np.float64)
            else:
                # sign * |x|^1.4: a plain power is NaN for negative log-odds.
                lin = (0.6 * np.sign(ell_star) * np.abs(ell_star) ** 1.4
                       if dgp == "curvature" else ell_star)
                p_pos = 1.0 / (1.0 + np.exp(-(alpha + lin)))
                agree_pos = (rng.random(n_positions) < p_pos).astype(np.float64)
                a = agree_pos[pd.state_row]

            n = len(pd)
            docs.append(np.full(n, d, dtype=np.int64))
            states.append(np.full(n, state_id, dtype=np.int64))
            poss.append(pd.state_row)
            ells.append(ell[pd.state_row])
            agrees.append(a)
            ms.append(pd.m)
            rs.append(pd.r)

    frame = AnalysisFrame(
        doc_id=np.concatenate(docs), state_id=np.concatenate(states),
        position=np.concatenate(poss), ell=np.concatenate(ells),
        agree=np.concatenate(agrees), m=np.concatenate(ms), r=np.concatenate(rs),
    )
    truth = Truth(dgp=dgp, beta_z=(delta if dgp == "direct_rank" else 0.0),
                  tau=tau, sigma=sigma, n_docs=n_docs, n_positions=n_positions)
    return frame, truth
