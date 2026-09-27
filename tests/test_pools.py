"""Exact pool expectations against brute-force enumeration and Monte Carlo pools."""
from itertools import combinations

import numpy as np
import pytest

from c3 import N_POOLS, POOL_SIZES, RETAIN
from c3.assemble import position_and_pool_frames
from c3.endpoints.e1 import delta_m, doc_codes_of
from c3.pools import long_weights, rank_weights, win_weights
from c3.sim.positions import simulate_positions
from c3.stats import doc_boot_counts


def _brute(ell, m, r):
    n = ell.size
    w = np.zeros(n)
    subsets = list(combinations(range(n), m))
    for sub in subsets:
        order = sorted(sub, key=lambda i: -ell[i])
        w[order[r - 1]] += 1
    return w / len(subsets)


@pytest.mark.parametrize("n,m,r", [(6, 1, 1), (6, 3, 1), (7, 4, 2), (8, 8, 1), (8, 5, 4), (5, 2, 2)])
def test_rank_weights_match_enumeration(n, m, r):
    ell = np.random.default_rng(n * 10 + m).normal(size=n)
    assert np.allclose(rank_weights(ell, m, r), _brute(ell, m, r), atol=1e-12)


def test_weights_sum_to_one_and_m1_is_uniform():
    ell = np.random.default_rng(0).normal(size=40)
    for m in (1, 2, 8, 32, 40):
        assert win_weights(ell, m).sum() == pytest.approx(1.0)
    assert np.allclose(win_weights(ell, 1), 1 / 40)
    assert win_weights(ell, 41).sum() == 0


def test_ties_share_their_weight():
    ell = np.array([2.0, 1.0, 1.0, 0.0])
    w = win_weights(ell, 2)
    assert w[1] == pytest.approx(w[2])
    assert w.sum() == pytest.approx(1.0)
    # Averaging over tie orders equals enumerating both orders.
    a = _brute(np.array([2.0, 1.0 + 1e-9, 1.0, 0.0]), 2, 1)
    b = _brute(np.array([2.0, 1.0, 1.0 + 1e-9, 0.0]), 2, 1)
    assert np.allclose(w, (a + b) / 2)


def test_long_weights_respect_min_state_size():
    state = np.r_[np.zeros(10, int), np.ones(40, int)]
    ell = np.random.default_rng(1).normal(size=50)
    row, m, w = long_weights(state, ell, POOL_SIZES, min_state_size=32)
    assert np.all(state[row] == 1)
    for mm in POOL_SIZES:
        assert w[m == mm].sum() == pytest.approx(1.0)


def test_exact_and_monte_carlo_delta_agree():
    df = simulate_positions("uniform", n_docs=80, seed=3)
    pos, pools = position_and_pool_frames(df, pool_sizes=POOL_SIZES, n_pools_per_size=200,
                                          seed=0, retain=RETAIN, min_state_size=32)
    codes = doc_codes_of(pos)
    counts = doc_boot_counts(codes.size, 200)
    mc = delta_m(pools, pos, codes, counts)
    ex = delta_m(None, pos, codes, counts, method="exact", min_state_size=32)
    assert np.array_equal(mc.levels, ex.levels)
    assert np.allclose(mc.point, ex.point, atol=0.004)
    # m=1 is the same exact quantity under both.
    assert mc.point[0] == pytest.approx(ex.point[0], abs=1e-12)
