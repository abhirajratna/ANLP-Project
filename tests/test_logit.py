"""The fixed-effect IRLS solver against a dense reference."""
import numpy as np
import pytest

from c3.logit import fit_fe_logit

sm = pytest.importorskip("statsmodels.api")


def _data(seed=0, N=3000, G=20, p=3):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, G, N)
    X = rng.normal(size=(N, p))
    eta = rng.normal(scale=0.7, size=G)[g] + X @ np.array([0.6, -0.4, 0.2])[:p]
    y = (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(float)
    return y, X, g, G


def _dense(y, X, g, G, offset=None):
    D = np.zeros((y.size, G)); D[np.arange(y.size), g] = 1.0
    return sm.GLM(y, np.hstack([X, D]), family=sm.families.Binomial(),
                  offset=offset).fit()


def test_matches_dense_glm():
    y, X, g, G = _data()
    ours = fit_fe_logit(y, X, g, n_groups=G, ridge_fe=0.0)
    ref = _dense(y, X, g, G)
    assert np.allclose(ours.beta, ref.params[:X.shape[1]], atol=1e-8)
    assert ours.loglik == pytest.approx(ref.llf, abs=1e-6)


def test_offset_matches_dense_glm():
    y, X, g, G = _data(seed=2)
    off = np.random.default_rng(3).normal(scale=0.5, size=y.size)
    ours = fit_fe_logit(y, X, g, n_groups=G, offset=off, ridge_fe=0.0)
    ref = _dense(y, X, g, G, offset=off)
    assert np.allclose(ours.beta, ref.params[:X.shape[1]], atol=1e-7)


def test_integer_weights_equal_row_duplication():
    y, X, g, G = _data(seed=5, N=800, G=8)
    w = np.random.default_rng(1).integers(0, 4, y.size).astype(float)
    weighted = fit_fe_logit(y, X, g, n_groups=G, weights=w, ridge_fe=1e-8)
    rep = np.repeat(np.arange(y.size), w.astype(int))
    dup = fit_fe_logit(y[rep], X[rep], g[rep], n_groups=G, ridge_fe=1e-8)
    assert np.allclose(weighted.beta, dup.beta, atol=1e-6)


def test_separated_group_stays_finite():
    # A group where every outcome is 1 has a divergent intercept without the ridge.
    y = np.array([1.0] * 20 + [0.0, 1.0] * 10)
    X = np.random.default_rng(0).normal(size=(40, 1))
    g = np.array([0] * 20 + [1] * 20)
    fit = fit_fe_logit(y, X, g, n_groups=2, ridge_fe=1e-4)
    assert np.all(np.isfinite(fit.alpha)) and np.all(np.isfinite(fit.beta))
