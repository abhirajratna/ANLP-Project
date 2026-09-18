"""z_{r,m} against closed forms and the identities it must satisfy."""
import math

import numpy as np
import pytest

from c3.order_stats import expected_order_statistic as z
from c3.order_stats import z_vector


def test_closed_forms():
    assert z(1, 1) == pytest.approx(0.0, abs=1e-12)
    assert z(1, 2) == pytest.approx(1 / math.sqrt(math.pi), abs=1e-10)
    assert z(2, 2) == pytest.approx(-1 / math.sqrt(math.pi), abs=1e-10)
    assert z(1, 3) == pytest.approx(3 / (2 * math.sqrt(math.pi)), abs=1e-10)


@pytest.mark.parametrize("m", [2, 4, 8, 16, 32])
def test_ranks_sum_to_zero(m):
    assert sum(z(r, m) for r in range(1, m + 1)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("m", [2, 4, 8, 16, 32])
def test_symmetry(m):
    for r in range(1, m + 1):
        assert z(r, m) == pytest.approx(-z(m - r + 1, m), abs=1e-9)


def test_winner_expectation_grows_with_pool_size():
    vals = [z(1, m) for m in (2, 4, 8, 16, 32)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_ranks_decrease_within_pool():
    vals = [z(r, 16) for r in range(1, 17)]
    assert all(b < a for a, b in zip(vals, vals[1:]))


def test_vectorised_lookup_matches_scalar():
    r = np.array([1, 2, 1, 5]); m = np.array([2, 4, 32, 8])
    assert np.allclose(z_vector(r, m), [z(a, b) for a, b in zip(r, m)])


def test_rejects_impossible_rank():
    with pytest.raises(ValueError):
        z(5, 4)
