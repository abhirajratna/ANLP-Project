"""The estimator must return a null when there is no effect and find a planted one."""
import numpy as np
import pytest

from c3.regression import fit_and_report, fit_once
from c3.simulate import simulate
from c3.splines import Spline

BASE = dict(n_docs=40, n_states_per_doc=3, n_positions=256, n_pools_per_size=8)


def _beta(dgp, knots=9, seed=11, **kw):
    fr, truth = simulate(dgp=dgp, seed=seed, **BASE, **kw)
    return fit_once(fr, Spline.fit(fr.ell, n_knots=knots), "eq2").coef("beta_z"), truth


def test_null_under_homoskedastic_noise():
    """The central guarantee: no manufactured effect when none exists."""
    b, truth = _beta("homoskedastic")
    assert truth.beta_z == 0.0
    assert abs(b) < 0.03, f"estimator manufactured beta_z={b:+.4f} under the null"


def test_recovers_planted_rank_effect():
    b, truth = _beta("direct_rank", tau=0.0, delta=-0.30)
    assert b < 0
    assert 0.5 * 0.30 < abs(b) < 1.5 * 0.30, f"planted -0.30, recovered {b:+.4f}"


def test_planted_effect_scales_with_its_size():
    small, _ = _beta("direct_rank", tau=0.0, delta=-0.10)
    large, _ = _beta("direct_rank", tau=0.0, delta=-0.40)
    assert large < small < 0


def test_heteroskedastic_bias_shrinks_with_spline_flexibility():
    """Documented failure mode: too few knots leak curvature into beta_z."""
    coarse, _ = _beta("heteroskedastic", knots=5, tau=1.0)
    fine, _ = _beta("heteroskedastic", knots=17, tau=1.0)
    assert abs(fine) < abs(coarse)


def test_eq3_rank_profile_is_flat_under_the_null():
    fr, _ = simulate(dgp="homoskedastic", seed=11, **BASE)
    fit = fit_once(fr, Spline.fit(fr.ell, n_knots=9), "eq3")
    gammas = np.array([v for n, v in fit.as_dict().items() if n.startswith("gamma_")])
    assert gammas.size > 0 and np.abs(gammas).max() < 0.25


def test_cross_fitted_path_runs_and_brackets_the_null():
    fr, _ = simulate(dgp="homoskedastic", seed=11, **BASE)
    est = fit_and_report(fr, "eq2", n_knots=9, n_boot=150, cross_fit=True)["beta_z"]
    assert np.isfinite(est.ci_low) and np.isfinite(est.ci_high)
    assert est.ci_low < 0 < est.ci_high, "null DGP should not exclude zero"


def test_bootstrap_interval_orders_correctly():
    fr, _ = simulate(dgp="direct_rank", tau=0.0, delta=-0.35, seed=11, **BASE)
    est = fit_and_report(fr, "eq2", n_knots=9, n_boot=200, cross_fit=False)["beta_z"]
    assert est.ci_low < est.value < est.ci_high
    assert est.excludes_zero and est.value < 0
