"""Restricted cubic spline basis for ``f(ell_i)`` in Eqs. 2-3.

``f`` is the flexible non-linear control for reported confidence.  Its job is to
stop a linear term from absorbing confidence-accuracy curvature and thereby
leaking into ``beta_z``.  A restricted (natural) cubic spline is the right shape
here because it is linear beyond the boundary knots, so the sparse tails of the
log-odds scale cannot drive wild extrapolation.

Harrell's parameterisation: K knots give K-1 basis columns (the linear term plus
K-2 cubic terms).
"""

from __future__ import annotations

import numpy as np


def default_knots(x: np.ndarray, n_knots: int = 5) -> np.ndarray:
    """Knots at evenly spaced quantiles of the observed confidence log-odds.

    Uses interior quantiles rather than min/max so a single extreme value cannot
    set a boundary knot.
    """
    if n_knots < 3:
        raise ValueError(f"need at least 3 knots, got {n_knots}")
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("no finite values to place knots on")
    # Harrell's recommended outer quantiles for 3-7 knots.
    lo, hi = (0.10, 0.90) if n_knots <= 4 else (0.05, 0.95)
    qs = np.linspace(lo, hi, n_knots)
    knots = np.quantile(x, qs)
    knots = np.unique(knots)
    if knots.size < 3:
        raise ValueError(
            f"confidence log-odds are too degenerate to place {n_knots} knots "
            f"(only {knots.size} distinct quantile(s))"
        )
    return knots


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis, shape ``(n, len(knots) - 1)``.

    Column 0 is ``x`` itself; the remaining columns are the constrained cubic
    terms.  An intercept is *not* included (state fixed effects supply it).
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(knots, dtype=np.float64)
    K = t.size
    if K < 3:
        raise ValueError(f"need at least 3 knots, got {K}")
    if not np.all(np.diff(t) > 0):
        raise ValueError("knots must be strictly increasing")

    def cube_plus(u: np.ndarray) -> np.ndarray:
        return np.where(u > 0, u, 0.0) ** 3

    denom = (t[-1] - t[0]) ** 2
    tail = t[-1] - t[-2]
    out = np.empty((x.size, K - 1), dtype=np.float64)
    out[:, 0] = x
    for j in range(K - 2):
        term = (
            cube_plus(x - t[j])
            - cube_plus(x - t[-2]) * (t[-1] - t[j]) / tail
            + cube_plus(x - t[-1]) * (t[-2] - t[j]) / tail
        )
        out[:, j + 1] = term / denom
    return out


class Spline:
    """A fitted spline basis: knots chosen once, then applied to new data.

    Knots must be frozen on the fit half and reused on the eval half, otherwise
    the two halves are not comparable.
    """

    def __init__(self, knots: np.ndarray):
        self.knots = np.asarray(knots, dtype=np.float64)

    @classmethod
    def fit(cls, x: np.ndarray, n_knots: int = 5) -> "Spline":
        return cls(default_knots(x, n_knots=n_knots))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return rcs_basis(x, self.knots)

    @property
    def n_columns(self) -> int:
        return self.knots.size - 1

    def names(self, prefix: str = "f_ell") -> list[str]:
        return [f"{prefix}[{j}]" for j in range(self.n_columns)]
