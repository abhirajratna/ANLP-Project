"""Pool construction: the randomisation that makes selection intensity exogenous.

One forward pass at a fixed masked state yields, for every masked position i, a
confidence log-odds ``ell_i``, a top-1 token ``v_i`` and correctness ``agree_i``
against the true token.  Pools are then drawn *post hoc* from that single pass:
random subsets ``P`` of size ``m``, ranked by ``ell``.  Because the forward pass
is shared across every draw, the model state is strictly constant and only pool
composition varies (proposal Fig. 1).

Ranks are assigned with an explicit random tie-break rather than by argsort
order.  Reported confidence saturates in low precision -- several positions can
share the clipped maximum -- and letting position index decide the winner would
manufacture a positional artefact in exactly the coefficient under audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rng import generator


@dataclass
class PoolDraw:
    """Long-format pool membership table: one row per (pool, retained member)."""

    state_row: np.ndarray   # index into the state's position arrays
    pool_id: np.ndarray     # pool identifier, unique within this draw
    m: np.ndarray           # pool size
    r: np.ndarray           # rank within pool, 1 = highest confidence

    def __len__(self) -> int:
        return int(self.state_row.size)


def rank_within_pools(ell_pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rank each row of ``ell_pool`` in descending order, ties broken at random.

    Returns an array of the same shape whose entries are ranks in ``1..m``.
    Implemented as two stable sorts, which is a lexicographic sort on
    ``(-ell, uniform)`` and so breaks ties uniformly at random.
    """
    n_pools, m = ell_pool.shape
    u = rng.random((n_pools, m))
    o1 = np.argsort(u, axis=1, kind="stable")
    primary = np.take_along_axis(-ell_pool, o1, axis=1)
    o2 = np.argsort(primary, axis=1, kind="stable")
    order = np.take_along_axis(o1, o2, axis=1)  # order[:, k] = member at rank k+1
    ranks = np.empty((n_pools, m), dtype=np.int64)
    np.put_along_axis(ranks, order, np.arange(1, m + 1)[None, :].repeat(n_pools, 0), axis=1)
    return ranks


def draw_pools(
    ell: np.ndarray,
    pool_sizes=(2, 4, 8, 16, 32),
    n_pools_per_size: int = 64,
    seed: int = 0,
    state_key: str = "",
    retain: int | str = "all",
) -> PoolDraw:
    """Draw random pools at one fixed masked state.

    Parameters
    ----------
    ell : (n_positions,) confidence log-odds from the single shared forward pass.
    n_pools_per_size : pools drawn per size ``m``.
    retain : ``"all"`` records every pool member with its rank (the rank profile
        of Eq. 3 needs ranks ``k >= 2``); an integer records only the top ``k``.
    """
    ell = np.asarray(ell, dtype=np.float64).ravel()
    n = ell.size
    rng = generator(seed, "pools", state_key)

    rows, pools, ms, rs = [], [], [], []
    next_pool_id = 0
    for m in pool_sizes:
        if m > n:
            # Not enough masked positions at this state to form a pool of size m.
            continue
        keys = rng.random((n_pools_per_size, n))
        idx = np.argsort(keys, axis=1, kind="stable")[:, :m]      # membership
        ell_pool = ell[idx]
        ranks = rank_within_pools(ell_pool, rng)

        if retain == "all":
            keep = np.ones_like(ranks, dtype=bool)
        else:
            k = int(retain)
            if k < 1:
                raise ValueError(f"retain must be >= 1 or 'all', got {retain}")
            keep = ranks <= k

        pid = (next_pool_id + np.arange(n_pools_per_size))[:, None].repeat(m, axis=1)
        rows.append(idx[keep])
        pools.append(pid[keep])
        ms.append(np.full(int(keep.sum()), m, dtype=np.int64))
        rs.append(ranks[keep])
        next_pool_id += n_pools_per_size

    if not rows:
        empty = np.empty(0, dtype=np.int64)
        return PoolDraw(empty, empty.copy(), empty.copy(), empty.copy())

    return PoolDraw(
        state_row=np.concatenate(rows).astype(np.int64),
        pool_id=np.concatenate(pools).astype(np.int64),
        m=np.concatenate(ms),
        r=np.concatenate(rs),
    )
