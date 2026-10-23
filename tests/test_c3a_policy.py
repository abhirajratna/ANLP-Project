"""C3a (plan v2 T11) as a policy, without the decode loop: runs without torch."""
import json

import numpy as np
import pandas as pd
import pytest

from c3.calibrators import DECODE_FEATURES, ContextLogistic, state_field_features
from c3.decode.policies import C3a, StepContext
from c3.frame import PositionFrame


def _calibrator():
    """A context calibrator fitted on simulated decoder states."""
    from c3.sim.trajectories import simulate_design

    df = simulate_design("calibrated", n_docs=30, doc_len=64, ks=(4,), matched=False, seed=1)
    df = df[df["mask_kind"] == "decoder"].reset_index(drop=True)
    return ContextLogistic(features=DECODE_FEATURES).fit(PositionFrame.from_frame(df))


STATE = {"median_ell": 0.5, "mask_rate": 0.5, "log_n_masked": float(np.log(16))}


def _ctx(state=STATE, step=0, n=4):
    return StepContext(step=step, positions=np.arange(n), state=dict(state) if state else None)


def test_c3a_counts_by_c_hat_and_commits_in_ell_order():
    cal = _calibrator()
    stats = {"ell": np.array([4.0, -2.0, 1.0, 6.0])}
    c = cal.predict_arrays(stats["ell"], STATE)
    for alpha in (0.01, 0.2, 0.5, 0.9):
        order, n = C3a(alpha, cal).select(stats, _ctx())
        assert n == max(1, int(np.sum(c >= 1 - alpha)))
        assert list(order) == [3, 0, 2, 1]                 # the model's own ranking, always
    with pytest.raises(ValueError):
        C3a(0.1, cal).select(stats, _ctx(state=None))
    with pytest.raises(ValueError):
        C3a(0.1, None).select(stats, _ctx())


class _NonMonotone:
    """c_hat falls with ell: ranking by c_hat would commit the least confident first."""
    feature_ranges = None

    def predict_arrays(self, ell, field):
        return 1.0 - 1.0 / (1.0 + np.exp(-np.asarray(ell)))


def test_a_non_monotone_calibrator_changes_the_count_not_the_order():
    stats = {"ell": np.array([3.0, -3.0, 0.0])}
    order, n = C3a(0.6, _NonMonotone()).select(stats, _ctx(n=3))
    assert list(order[:n]) == [0, 2][:n] and order[0] == 0


def test_out_of_range_states_are_counted_logged_and_optionally_clamped():
    cal = _calibrator()
    cal.feature_ranges = {"mask_rate": {"min": 0.1, "max": 0.6},
                          "log_n_masked": {"min": 0.0, "max": float(np.log(100))},
                          "median_ell": {"min": -3.0, "max": 3.0}}
    far = {"median_ell": 0.5, "mask_rate": 1.0, "log_n_masked": float(np.log(256))}
    stats = {"ell": np.linspace(-3, 6, 9)}
    log = []
    C3a(0.3, cal, log=log).select(stats, _ctx(far, step=2, n=9))
    assert log == [{"step": 2, **far, "n_eligible": 9, "n_commit": log[0]["n_commit"],
                    "outside": ["mask_rate", "log_n_masked"]}]
    # clamping evaluates the calibrator at the range's edge
    clamped = C3a(0.3, cal, clamp=True)
    st, outside = clamped.state_features(_ctx(far))
    assert outside == ["mask_rate", "log_n_masked"]
    assert st["mask_rate"] == 0.6 and st["log_n_masked"] == pytest.approx(np.log(100))
    edge = cal.predict_arrays(stats["ell"], st)
    assert np.allclose(clamped.calibrated(stats, _ctx(far, n=9))[0], edge)
    # in range: nothing is flagged and clamping changes nothing
    _, outside = clamped.state_features(_ctx())
    assert outside == []


def test_calibrator_round_trips_with_its_ranges_and_predicts_like_the_trace_frame(tmp_path):
    cal = _calibrator()
    cal.feature_ranges = {"mask_rate": {"min": 0.05, "max": 1.0}}
    (tmp_path / "c.json").write_text(json.dumps({"calibrator": cal.to_dict(), "other": 1}))
    back = ContextLogistic.load(tmp_path / "c.json")
    assert back.feature_ranges == cal.feature_ranges
    # one trace state: 6 masked positions of a 12-position maskable region
    ell = np.array([2.0, -1.0, 0.3, 3.0, 1.1, -0.2])
    df = pd.DataFrame({"doc_id": 0, "state_id": 0, "position": np.arange(6), "ell": ell,
                       "agree": 1, "mask_rate": 6 / 12, "n_masked": 6, "step": 0})
    st = state_field_features(ell, 6, 12)
    assert np.allclose(back.predict_arrays(ell, st), cal.predict(PositionFrame.from_frame(df)))
