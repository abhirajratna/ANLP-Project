"""Pool randomisation: exogenous composition, unbiased ranks, fair tie-breaks."""
import numpy as np

from c3.design import draw_pools, rank_within_pools


def test_every_rank_appears_once_per_pool():
    ell = np.random.default_rng(0).normal(size=64)
    d = draw_pools(ell, (8,), n_pools_per_size=100, seed=1, state_key="s")
    for pid in np.unique(d.pool_id):
        assert sorted(d.r[d.pool_id == pid]) == list(range(1, 9))


def test_pool_membership_is_exogenous_in_confidence():
    # Mean confidence over *all* members must not vary with pool size; only the
    # winner's confidence should. That is what makes selection intensity exogenous.
    ell = np.random.default_rng(2).normal(size=128)
    d = draw_pools(ell, (2, 32), n_pools_per_size=400, seed=3, state_key="s")
    means = [ell[d.state_row[d.m == m]].mean() for m in (2, 32)]
    assert abs(means[0] - means[1]) < 0.06
    winners = [ell[d.state_row[(d.m == m) & (d.r == 1)]].mean() for m in (2, 32)]
    assert winners[1] > winners[0] + 0.5


def test_ties_broken_uniformly_not_positionally():
    r = rank_within_pools(np.zeros((8000, 4)), np.random.default_rng(0))
    share = np.bincount(np.argmax(r == 1, axis=1), minlength=4) / 8000
    assert np.all(np.abs(share - 0.25) < 0.02)


def test_pool_larger_than_state_is_skipped():
    d = draw_pools(np.random.default_rng(0).normal(size=4), (2, 8), 10, seed=0, state_key="s")
    assert set(d.m.tolist()) == {2}


def test_retain_top_k():
    ell = np.random.default_rng(0).normal(size=32)
    d = draw_pools(ell, (8,), 50, seed=0, state_key="s", retain=1)
    assert np.all(d.r == 1)


def test_draw_is_reproducible():
    ell = np.random.default_rng(0).normal(size=40)
    a = draw_pools(ell, (4,), 20, seed=9, state_key="x")
    b = draw_pools(ell, (4,), 20, seed=9, state_key="x")
    assert np.array_equal(a.r, b.r) and np.array_equal(a.state_row, b.state_row)
