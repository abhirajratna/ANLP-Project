"""The restricted cubic spline must be linear outside the boundary knots."""
import numpy as np

from c3.splines import Spline, default_knots, rcs_basis


def test_basis_shape():
    x = np.linspace(-3, 3, 100)
    k = default_knots(x, 5)
    assert rcs_basis(x, k).shape == (100, 4)


def test_linear_beyond_boundary_knots():
    knots = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x = np.array([3.0, 4.0, 5.0, 6.0])
    B = rcs_basis(x, knots)
    # A linear function of x has constant second differences of zero in every column.
    assert np.allclose(np.diff(B, n=2, axis=0), 0.0, atol=1e-9)


def test_knots_frozen_on_fit_half():
    rng = np.random.default_rng(0)
    sp = Spline.fit(rng.normal(size=5000), n_knots=5)
    other = rng.normal(loc=10.0, size=100)
    # Transforming shifted data must reuse the original knots, not re-fit them.
    assert np.allclose(sp.transform(other), rcs_basis(other, sp.knots))


def test_degenerate_input_rejected():
    import pytest
    with pytest.raises(ValueError):
        default_knots(np.zeros(100), 5)
