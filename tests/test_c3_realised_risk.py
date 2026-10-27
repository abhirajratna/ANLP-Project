"""Realised risk of C3a vs the raw 1 - alpha rule on report-half decoder states (exploratory).

scripts/c3_realised_risk.py re-plays one commit step per teacher-forced decoder state of the
report half.  Its vectorised simulation must commit exactly what ``C3a.select`` and
``Threshold.select`` would at that state, and its report-half filter must never let a
fit-half document through.  numpy / pandas / pyarrow only (no torch).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import c3_realised_risk as rr  # noqa: E402

from c3.calibrators import DECODE_FEATURES, ContextLogistic, state_field_features  # noqa: E402
from c3.decode.policies import C3a, StepContext, Threshold  # noqa: E402
from c3.frame import PositionFrame  # noqa: E402
from c3.stats import doc_boot_counts  # noqa: E402

ALPHAS = (0.01, 0.05, 0.10, 0.20, 0.40, 0.60)


def _calibrator():
    """A context calibrator fitted on simulated decoder states (as tests/test_c3a_policy.py)."""
    from c3.sim.trajectories import simulate_design

    df = simulate_design("calibrated", n_docs=30, doc_len=64, ks=(4,), matched=False, seed=1)
    df = df[df["mask_kind"] == "decoder"].reset_index(drop=True)
    return ContextLogistic(features=DECODE_FEATURES).fit(PositionFrame.from_frame(df))


def _states(n_states=50, seed=0):
    """Synthetic decoder states: |state| 1..70 over a region starting past 0 so that 32-blocks
    straddle, tied ell, saturated conf (= 1.0), random agree, a few documents."""
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("doc_id", "state_id", "position", "span_start", "span_len",
                            "n_masked", "ell", "conf", "agree")}
    for s in range(n_states):
        n = int(rng.integers(1, 71)) if s > 2 else (1, 70, 33)[s]
        span_len = int(rng.integers(max(n, 8), 120))
        start = int(rng.integers(5, 40))
        pos = np.sort(rng.choice(span_len, size=n, replace=False)) + start
        ell = np.round(rng.normal(1.5, 2.5, n), 1)                 # one decimal: many ties
        conf = (1.0 / (1.0 + np.exp(-ell))).astype(np.float32)     # float32, as the traces
        conf[rng.random(n) < 0.1] = np.float32(1.0)                # saturated
        for k, v in (("doc_id", np.full(n, s % 7)), ("state_id", np.full(n, s)), ("position", pos),
                     ("span_start", np.full(n, start)), ("span_len", np.full(n, span_len)),
                     ("n_masked", np.full(n, n)), ("ell", ell), ("conf", conf),
                     ("agree", rng.integers(0, 2, n))):
            cols[k].append(v)
    return rr.make_rows(**{k: np.concatenate(v) for k, v in cols.items()})


def _policy_inputs(rows, s, variant):
    """What the decode loop hands a policy for state s: the eligible rows' statistics, a
    StepContext with the state's features over every masked row, and the eligible positions."""
    sl = slice(rows.state_start[s], rows.state_start[s + 1])
    pos, ell = rows.position[sl], rows.ell[sl]
    keep = (np.ones(pos.size, bool) if variant == "region"
            else (pos - rows.span_start[sl]) // rr.BLOCK == ((pos - rows.span_start[sl]) // rr.BLOCK).min())
    state = state_field_features(ell, int(rows.n_masked[sl][0]), int(rows.span_len[sl][0]))
    ctx = StepContext(step=0, positions=pos[keep], gen_start=int(rows.span_start[sl][0]), state=state)
    return {"ell": ell[keep], "conf": rows.conf[sl][keep]}, ctx, keep, sl


@pytest.mark.parametrize("variant", ["block32", "region"])
def test_simulated_c3a_commits_what_C3a_select_returns(variant):
    rows, cal = _states(), _calibrator()
    chat = rr.chat_for(rows, cal)
    sim = rr.simulate(rows, chat, variant, ALPHAS)
    assert any(sim.size[s] < rows.n_rows_state[s] for s in range(rows.n_states)) or variant == "region"
    for s in range(rows.n_states):
        stats, ctx, keep, sl = _policy_inputs(rows, s, variant)
        for a, alpha in enumerate(ALPHAS):
            order, n = C3a(alpha, cal).select(stats, ctx)
            want = set(ctx.positions[order[:n]].tolist())
            assert sim.n["c3a"][a, s] == n
            assert set(rows.position[sim.committed(s, "c3a", a)].tolist()) == want
            n_pass = int(np.sum(cal.predict_arrays(stats["ell"], ctx.state) >= 1.0 - alpha))
            assert sim.forced["c3a"][a, s] == (n_pass == 0)
            assert sim.correct["c3a"][a, s] == rows.agree[sim.committed(s, "c3a", a)].sum()


@pytest.mark.parametrize("variant", ["block32", "region"])
def test_simulated_raw_rule_commits_what_Threshold_select_returns(variant):
    rows, cal = _states(seed=3), _calibrator()
    sim = rr.simulate(rows, rr.chat_for(rows, cal), variant, ALPHAS)
    for s in range(rows.n_states):
        stats, ctx, _, _ = _policy_inputs(rows, s, variant)
        for a, alpha in enumerate(ALPHAS):
            order, n = Threshold(tau=1.0 - alpha).select(stats, ctx)
            assert sim.n["raw"][a, s] == n
            assert (set(rows.position[sim.committed(s, "raw", a)].tolist())
                    == set(ctx.positions[order[:n]].tolist()))
            assert sim.forced["raw"][a, s] == (not np.any(stats["conf"] >= 1.0 - alpha))
    # the saturated conf = 1.0 rows pass at every alpha, as in T12
    assert (sim.n["raw"][0] >= 1).all()


def test_the_in_script_check_agrees_on_synthetic_states_and_catches_a_wrong_calibrator():
    rows, cal = _states(seed=5), _calibrator()
    chat = rr.chat_for(rows, cal)
    rr.verify_against_policies(rows, chat, cal, ALPHAS, n_states=40, seed=0)
    other = _calibrator()
    other.intercept += 3.0                          # commits different counts
    with pytest.raises(AssertionError):
        rr.verify_against_policies(rows, chat, other, ALPHAS, n_states=60, seed=0)


def test_vectorised_state_features_equal_state_field_features_and_the_fit_path():
    rows, cal = _states(seed=7), _calibrator()
    feats = rr.state_features(rows)
    for s in range(rows.n_states):
        sl = slice(rows.state_start[s], rows.state_start[s + 1])
        ref = state_field_features(rows.ell[sl], int(rows.n_masked[sl][0]), int(rows.span_len[sl][0]))
        for name in DECODE_FEATURES:
            assert feats[name][s] == pytest.approx(ref[name], rel=0, abs=1e-12)
    # the PositionFrame path that the fit used gives the same c_hat
    import pandas as pd
    df = pd.DataFrame({"doc_id": rows.doc_id, "state_id": rows.state_id, "position": rows.position,
                       "ell": rows.ell, "conf": rows.conf, "agree": rows.agree,
                       "mask_kind": "decoder", "mask_rate": rows.n_masked / rows.span_len,
                       "n_masked": rows.n_masked, "step": 0})
    assert np.allclose(rr.chat_for(rows, cal), cal.predict(PositionFrame.from_frame(df)), atol=1e-12)


def test_singleton_state_is_forced_exactly_when_c_hat_is_below_the_bar():
    cal = _calibrator()
    rows = rr.make_rows(doc_id=[0], state_id=[0], position=[10], span_start=[4], span_len=[20],
                        n_masked=[1], ell=[0.2], conf=np.float32([0.55]), agree=[1])
    chat = rr.chat_for(rows, cal)
    c = float(chat[0])
    for alpha, forced in ((1.0 - c - 0.05, True), (1.0 - c + 0.05, False)):
        sim = rr.simulate(rows, chat, "block32", (alpha,))
        assert sim.n["c3a"][0, 0] == 1 and sim.forced["c3a"][0, 0] == forced
        assert sim.size[0] == 1


# ---- bootstrap ---------------------------------------------------------------------------

def test_ratio_bootstrap_is_deterministic_and_the_point_is_the_pooled_ratio():
    rng = np.random.default_rng(0)
    num, den = rng.integers(0, 5, 40).astype(float), rng.integers(1, 9, 40).astype(float)
    a = rr.ratio_ci(num, den, doc_boot_counts(40, 500, seed=0, label="t11-realised-risk-toy"))
    b = rr.ratio_ci(num, den, doc_boot_counts(40, 500, seed=0, label="t11-realised-risk-toy"))
    assert a["value"] == b["value"] == pytest.approx(num.sum() / den.sum())
    assert a["ci"] == b["ci"] and np.array_equal(a["draws"], b["draws"])
    c = rr.ratio_ci(num, den, doc_boot_counts(40, 500, seed=1, label="t11-realised-risk-toy"))
    assert c["ci"] != a["ci"]


def test_ratio_bootstrap_on_a_toy_with_known_counts():
    num, den = np.array([1.0, 0.0, 3.0]), np.array([2.0, 2.0, 2.0])
    counts = np.array([[3, 0, 0], [0, 3, 0], [1, 1, 1], [0, 0, 3]])
    r = rr.ratio_ci(num, den, counts)
    assert r["value"] == pytest.approx(4 / 6)
    assert np.allclose(r["draws"], [0.5, 0.0, 4 / 6, 1.5])


def test_paired_difference_uses_the_same_draws():
    num, den = np.array([1.0, 2.0, 0.0, 1.0]), np.array([2.0, 2.0, 1.0, 1.0])
    num2 = np.array([2.0, 2.0, 1.0, 1.0])
    counts = doc_boot_counts(4, 200, seed=0, label="t11-realised-risk-toy")
    a, b = rr.ratio_ci(num, den, counts), rr.ratio_ci(num2, den, counts)
    d = rr.diff_ci(a, b)
    assert d["value"] == pytest.approx(a["value"] - b["value"])
    assert np.allclose(d["draws"], a["draws"] - b["draws"])


# ---- the report-half guard ---------------------------------------------------------------

def _trace(tmp_path):
    import pandas as pd

    rows = []
    for doc in (0, 1, 2, 3):                        # docs 0, 2: fit half; 1, 3: report half
        for k in (2, 4):
            for state in range(2):
                for p in range(5):
                    rows.append({"doc_id": doc, "state_id": doc * 100 + k * 10 + state, "position": 10 + p,
                                 "ell": 0.1 * p, "conf": 0.5, "agree": p % 2, "mask_kind": "decoder",
                                 "policy": "topk_conf", "k": k, "mask_rate": 0.5, "n_masked": 5,
                                 "step": state, "span_start": 10, "span_len": 10})
    rows.append({**rows[0], "mask_kind": "uniform", "doc_id": 1})
    path = tmp_path / "part0.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_report_loader_drops_fit_half_documents_and_the_post_read_check_raises_if_bypassed(tmp_path):
    path = _trace(tmp_path)
    df = rr.read_report_rows([path], report_docs=[1, 3], ks=(2, 4))
    assert set(df["doc_id"].unique()) == {1, 3} and set(df["k"].unique()) == {2, 4}
    assert (df["mask_kind"] == "decoder").all()
    # bypass the read filter: a fit-half document in a report-half run must raise
    import pyarrow.parquet as pq
    unfiltered = pq.read_table(path).to_pandas()
    with pytest.raises(SystemExit, match="non-report"):
        rr.check_report_only(unfiltered["doc_id"].to_numpy(), [1, 3])
    rr.check_report_only(df["doc_id"].to_numpy(), [1, 3])        # the filtered read passes


def test_report_docs_come_from_value_one_of_the_manifest():
    assert rr.report_docs({"3": 1, "4": 0, "5": 1}) == [3, 5]
    with pytest.raises(SystemExit):
        rr.report_docs({"3": 0})
