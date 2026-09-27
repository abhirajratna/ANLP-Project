"""Exact pool expectations: the average over *every* pool, not 20 random ones.

Plan v2 notes that at a fixed state each position's (ell_i, a_i) are fixed
numbers, so drawing random pools only re-weights positions already observed.
The re-weighting can be written down.  Sort the n masked positions of a state by
ell, highest first.  A uniformly random pool of size m is won by the position at
sorted rank j exactly when it is in the pool and the m-1 others all come from the
n-j positions below it:

    P(j wins) = C(n-j, m-1) / C(n, m)

and more generally the position at sorted rank j sits at pool rank r with

    P(j has rank r) = C(j-1, r-1) C(n-j, m-r) / C(n, m).

So E_m[v_winner] at a state is sum_j P(j wins) v_j -- the limit of the Monte Carlo
pool average as the number of pools grows, with no sampling noise, m = 1 as the
plain state mean, and one row per (position, m) instead of 20 per (pool, m).
Positions tied in ell share their weights equally, which is the expectation of
the random tie-break ``design.rank_within_pools`` uses.

This is a deviation from the plan's literal "20 random pools"; it computes the
quantity those pools estimate.  The Monte Carlo design remains available and is
the robustness check.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from . import POOL_SIZES


def _log_comb(n, k):
    n = np.asarray(n, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    out = np.full(np.broadcast(n, k).shape, -np.inf)
    ok = (k >= 0) & (k <= n)
    nn, kk = np.broadcast_arrays(n, k)
    out[ok] = gammaln(nn[ok] + 1) - gammaln(kk[ok] + 1) - gammaln(nn[ok] - kk[ok] + 1)
    return out


def rank_weights(ell: np.ndarray, m: int, r: int = 1) -> np.ndarray:
    """P(position i sits at rank r of a uniformly random size-m pool), for every i.

    Returned in the input order.  Sums to 1 over positions when m <= n.  Ties in
    ``ell`` share the probability of the ranks they jointly occupy.
    """
    ell = np.asarray(ell, dtype=np.float64).ravel()
    n = ell.size
    if m < 1 or r < 1 or r > m:
        raise ValueError(f"need 1 <= r <= m, got r={r}, m={m}")
    if m > n:
        return np.zeros(n)
    order = np.argsort(-ell, kind="stable")
    j = np.arange(1, n + 1)
    logw = _log_comb(j - 1, r - 1) + _log_comb(n - j, m - r) - _log_comb(n, m)
    w_sorted = np.exp(logw)
    # Average within runs of tied ell (in sorted order).
    s = ell[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    ends = np.r_[starts[1:], n]
    if starts.size < n:
        for a, b in zip(starts, ends):
            if b - a > 1:
                w_sorted[a:b] = w_sorted[a:b].mean()
    w = np.empty(n)
    w[order] = w_sorted
    return w


def win_weights(ell: np.ndarray, m: int) -> np.ndarray:
    """P(position i wins a uniformly random pool of size m)."""
    return rank_weights(ell, m, 1)


def long_weights(state_id: np.ndarray, ell: np.ndarray, pool_sizes=POOL_SIZES,
                 min_state_size: int = 1, r: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank-r weights for every (position, m) with m <= |state|, in long format.

    Returns ``(row, m, weight)``: ``row`` indexes the input arrays.  States with
    fewer than ``min_state_size`` positions are skipped, so every m averages over
    the same states (see ``assemble.frame_from_arrays``).
    """
    state_id = np.asarray(state_id)
    ell = np.asarray(ell, dtype=np.float64)
    order = np.argsort(state_id, kind="stable")
    bounds = np.flatnonzero(np.diff(state_id[order])) + 1
    rows, ms, ws = [], [], []
    for idx in np.split(order, bounds):
        n = idx.size
        if n == 0 or n < min_state_size:
            continue
        for m in pool_sizes:
            if m > n or r > m:
                continue
            w = rank_weights(ell[idx], m, r)
            keep = w > 0
            rows.append(idx[keep])
            ms.append(np.full(int(keep.sum()), m, dtype=np.int64))
            ws.append(w[keep])
    if not rows:
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0))
    return np.concatenate(rows), np.concatenate(ms), np.concatenate(ws)
