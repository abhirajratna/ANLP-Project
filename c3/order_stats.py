"""Expected normal order statistics, the regressor ``z_{r,m}`` of Eq. 2.

``z_{r,m}`` is the expectation of the r-th *largest* of ``m`` i.i.d. standard
normal draws (so ``r = 1`` is the winner of the pool).  It enters Eq. 2 as the
single-degree-of-freedom summary of selection intensity: under the proposal's
error model a winner's curse implies ``beta_z < 0`` and calibration given
selection implies ``beta_z = 0``.

The standard normal is used as the reference distribution, so the noise scale is
absorbed into ``beta_z`` rather than being separately identified.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import integrate
from scipy.special import betaln
from scipy.stats import norm


def _log_order_density(x: np.ndarray, k: int, m: int) -> np.ndarray:
    """log density of the k-th *smallest* of m standard normals."""
    return (
        -betaln(k, m - k + 1)
        + (k - 1) * norm.logcdf(x)
        + (m - k) * norm.logsf(x)
        + norm.logpdf(x)
    )


@lru_cache(maxsize=4096)
def expected_order_statistic(r: int, m: int) -> float:
    """E[Z_(r):m] where r=1 is the largest of m standard normal draws.

    Computed by quadrature on the density of the corresponding order statistic.
    Exact reference values: z(1,1)=0, z(1,2)=1/sqrt(pi), z(1,3)=3/(2*sqrt(pi)).
    """
    if m < 1:
        raise ValueError(f"pool size m must be >= 1, got {m}")
    if not 1 <= r <= m:
        raise ValueError(f"rank r must satisfy 1 <= r <= m, got r={r}, m={m}")
    if m == 1:
        return 0.0

    k = m - r + 1  # k-th smallest == r-th largest

    def integrand(x: float) -> float:
        return x * float(np.exp(_log_order_density(np.asarray(x), k, m)))

    # The density concentrates near the corresponding normal quantile; split the
    # range there so the quadrature does not miss a narrow peak for large m.
    centre = float(norm.ppf(k / (m + 1)))
    lo, hi = centre - 12.0, centre + 12.0
    value, _ = integrate.quad(
        integrand, lo, hi, points=[centre], limit=400, epsabs=1e-12, epsrel=1e-11
    )
    return float(value)


def z_table(pool_sizes) -> dict[tuple[int, int], float]:
    """Precompute ``{(r, m): z_{r,m}}`` for every rank in every pool size."""
    return {
        (r, m): expected_order_statistic(r, m)
        for m in pool_sizes
        for r in range(1, m + 1)
    }


def z_vector(ranks: np.ndarray, pool_sizes: np.ndarray) -> np.ndarray:
    """Vectorised lookup of ``z_{r,m}`` for arrays of ranks and pool sizes."""
    ranks = np.asarray(ranks, dtype=np.int64)
    pool_sizes = np.asarray(pool_sizes, dtype=np.int64)
    if ranks.shape != pool_sizes.shape:
        raise ValueError("ranks and pool_sizes must have the same shape")
    out = np.empty(ranks.shape, dtype=np.float64)
    for (r, m) in {(int(a), int(b)) for a, b in zip(ranks.ravel(), pool_sizes.ravel())}:
        mask = (ranks == r) & (pool_sizes == m)
        out[mask] = expected_order_statistic(r, m)
    return out
