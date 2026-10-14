"""B, the proper-score field test, on positions with known truth."""
import numpy as np
import pytest

from c3.endpoints.fieldtest import (DECODE_FEATURES, brier, field_effect_test, log_loss,
                                    paired_score_test)
from c3.frame import PositionFrame, hash_halves
from c3.sim.positions import simulate_positions
from c3.sim.truth import true_calibrators
from c3.stats import doc_boot_counts


def _setup(scenario, n_docs=160, seed=1, n_boot=400):
    df = simulate_positions(scenario, n_docs=n_docs, seed=seed)
    pos = PositionFrame.from_frame(df)
    codes = np.unique(pos.doc_id)
    return df, pos, codes, doc_boot_counts(codes.size, n_boot, seed=0, label="test-fieldtest")


def test_true_context_calibrator_beats_the_marginal_one_only_under_context():
    # The oracle: with the exact pi(l) and pi(l, s), the log-loss gain is the
    # expected KL between them, positive under 'context' and zero under 'uniform'.
    for scenario, expect in (("context", True), ("uniform", False)):
        df, pos, codes, counts = _setup(scenario)
        pi_l, pi_ls = true_calibrators(df, scenario)
        res = paired_score_test(pi_l, pi_ls, pos.agree, np.searchsorted(codes, pos.doc_id),
                                codes.size, counts)
        assert res["helps"] is expect, (scenario, res)
        if not expect:
            assert abs(res["gain"]) < 1e-3


def test_fitted_calibrators_find_the_field_effect_out_of_sample():
    for scenario, expect in (("context", True), ("uniform", False), ("calibrated", False)):
        _, pos, codes, counts = _setup(scenario)
        half = hash_halves(pos.doc_id, seed=0)
        res = field_effect_test(pos, half == 0, half == 1, codes, counts,
                                features=("median_ell", "mask_rate", "log_n_masked"))
        ll = res["spline_vs_context"]["log_loss"]
        assert ll["helps"] is expect, (scenario, ll)
        assert res["context_converged"] and res["own_converged"]
        assert np.isfinite(res["isotonic_vs_context"]["log_loss"]["gain"])


def test_decode_features_exclude_step():
    assert "step" not in DECODE_FEATURES


def test_clipping_keeps_log_loss_finite_for_hard_predictions():
    p = np.array([0.0, 1.0, 1.0, 0.0])
    a = np.array([1, 0, 1, 0])
    ll = log_loss(p, a)
    assert np.all(np.isfinite(ll))
    assert ll[2] < 1e-5 and ll[3] < 1e-5 and ll[0] > 10
    assert np.allclose(brier(p, a), [1, 1, 0, 0])


def test_pairing_is_document_first_and_matches_a_hand_computation():
    # 3 documents with 1, 2 and 3 positions: documents count equally.
    doc_idx = np.array([0, 1, 1, 2, 2, 2])
    agree = np.array([1, 0, 1, 1, 1, 0])
    p0 = np.full(6, 0.5)
    p1 = np.array([0.9, 0.2, 0.7, 0.8, 0.6, 0.3])
    counts = doc_boot_counts(3, 50, seed=0)
    res = paired_score_test(p0, p1, agree, doc_idx, 3, counts)
    d = log_loss(p0, agree) - log_loss(p1, agree)
    by_doc = [d[doc_idx == j].mean() for j in range(3)]
    assert res["gain"] == pytest.approx(np.mean(by_doc))
    again = paired_score_test(p0, p1, agree, doc_idx, 3, counts)
    assert res["ci"] == again["ci"]           # same counts, same draws


def test_overlapping_fit_and_report_are_refused():
    _, pos, codes, counts = _setup("uniform", n_docs=20)
    everything = np.ones(len(pos), dtype=bool)
    with pytest.raises(ValueError, match="overlap"):
        field_effect_test(pos, everything, everything, codes, counts)
