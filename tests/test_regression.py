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


# ---- Eq. 2/3 bootstrap warm start -------------------------------------------------

def _boot_frames():
    fr, _ = simulate(dgp="direct_rank", tau=0.0, delta=-0.3, seed=5, n_docs=24,
                     n_states_per_doc=3, n_positions=128, n_pools_per_size=6)
    return {"eq2": fr.winners(), "eq3": fr}


def _boots(frame, eq, fe, warm, n_boot=12, **kw):
    return fit_and_report(frame, eq, n_knots=5, n_boot=n_boot, cross_fit=False, fe=fe,
                          warm_start=warm, max_rank=4, **kw)


@pytest.mark.parametrize("eq", ["eq2", "eq3"])
@pytest.mark.parametrize("fe", [False, True])
def test_warm_start_matches_cold_and_is_faster(eq, fe):
    frame = _boot_frames()[eq]
    cold, warm = _boots(frame, eq, fe, False), _boots(frame, eq, fe, True)
    assert cold.keys() == warm.keys()
    worst = 0.0
    for k in cold:
        assert cold[k].boot.shape == warm[k].boot.shape       # same draws survive
        worst = max(worst, float(np.max(np.abs(cold[k].boot - warm[k].boot))))
        assert cold[k].n_nonconverged_draws == 0 and warm[k].n_nonconverged_draws == 0
    print(f"{eq} fe={fe}: max |warm - cold| = {worst:.2e}")
    assert worst < 1e-8
    k0 = next(iter(cold))
    assert warm[k0].n_iter_median < cold[k0].n_iter_median
    assert warm[k0].warm_start is True and cold[k0].warm_start is False


def test_cold_start_reproduces_frozen_worker_exactly():
    """warm_start=False must be bit-identical to the pre-change per-resample fit."""
    from c3.logit import fit_fe_logit
    from c3.regression import build_design
    from c3.rng import generator

    frame = _boot_frames()["eq2"]
    est = _boots(frame, "eq2", True, False, n_boot=6)["beta_z"]
    spline = Spline.fit(frame.ell, n_knots=5)
    X, names = build_design(frame, spline, "eq2")
    doc_idx, n_docs = frame.doc_index
    ref = []
    for b in range(6):
        counts = generator(0, "bootstrap", str(b)).multinomial(n_docs, np.full(n_docs, 1.0 / n_docs))
        w = counts[doc_idx].astype(float)
        keep = w > 0
        _, g = np.unique(frame.state_index[0][keep], return_inverse=True)
        ref.append(fit_fe_logit(frame.agree[keep], X[keep], g, n_groups=int(g.max()) + 1,
                                names=names, offset=np.zeros(keep.sum()), weights=w[keep]).coef("beta_z"))
    assert np.array_equal(est.boot, np.array(ref))


def test_nonconverged_draws_are_reported():
    frame = _boot_frames()["eq2"]
    ok = _boots(frame, "eq2", True, True)["beta_z"]
    assert ok.n_nonconverged_draws == 0
    bad = _boots(frame, "eq2", True, False, max_iter=2)["beta_z"]
    assert bad.n_nonconverged_draws > 0 and bad.n_iter_max <= 2


def test_library_default_is_cold_and_bit_identical_to_explicit_cold():
    frame = _boot_frames()["eq2"]
    dflt = fit_and_report(frame, "eq2", n_knots=5, n_boot=6, cross_fit=False, fe=True, max_rank=4)["beta_z"]
    cold = _boots(frame, "eq2", True, False, n_boot=6)["beta_z"]
    assert dflt.warm_start is False and dflt.point_restarted is False
    assert np.array_equal(dflt.boot, cold.boot) and dflt.value == cold.value
    # ...and the cold path itself is the frozen worker (see test above).


def test_unconverged_point_fit_is_restarted_once_when_warm():
    frame = _boot_frames()["eq2"]
    cold = _boots(frame, "eq2", True, False, max_iter=5)["beta_z"]
    warm = _boots(frame, "eq2", True, True, max_iter=5)["beta_z"]
    assert cold.converged is False and cold.point_restarted is False
    assert warm.converged is True and warm.point_restarted is True
    assert warm.point_iters_total > 5 and warm.n_iter <= 5
    ref = _boots(frame, "eq2", True, False)["beta_z"]       # fully converged cold fit
    assert abs(warm.value - ref.value) < 1e-8


def test_unconverged_draws_are_retried_once_and_remaining_counted():
    frame = _boot_frames()["eq2"]
    cold = _boots(frame, "eq2", True, False, max_iter=4)["beta_z"]
    warm = _boots(frame, "eq2", True, True, max_iter=4)["beta_z"]
    assert cold.n_nonconverged_draws > 0
    assert warm.n_nonconverged_draws == 0 and warm.n_iter_max > 4       # retried draws
    ref = _boots(frame, "eq2", True, False)["beta_z"]
    assert np.max(np.abs(warm.boot - ref.boot)) < 1e-6
    hopeless = _boots(frame, "eq2", True, True, max_iter=1)["beta_z"]
    assert hopeless.n_nonconverged_draws > 0


def test_paper_marker_flags_nonconverged_draws():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import c1_paper_outputs as P
    base = {"value": 0.1, "ci": [0.0, 0.2], "point_converged": True}

    def reg(**kw):
        return {"regression": {"eq2_fe": {"beta_z": {**base, **kw}}}}
    assert P.beta(reg(n_nonconverged_draws=0), "eq2_fe")[1] == ""
    assert P.beta(reg(n_nonconverged_draws=3), "eq2_fe")[1] == "\\S"
    assert P.beta(reg(point_converged=False, n_nonconverged_draws=0), "eq2_fe")[1] == "\\S"
