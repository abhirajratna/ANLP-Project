"""Plan v2 endpoints on traces with known truth: E1, E2, Eq. 2 with and without FE."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from c3 import N_POOLS, POOL_SIZES, RETAIN
from c3.assemble import position_and_pool_frames
from c3.calibrators import crossfit_predictions, fit_on_half
from c3.endpoints.e1 import delta_m, doc_codes_of, native_gap, winner_gap
from c3.endpoints.e2 import decompose, icc_residuals
from c3.frame import PositionFrame, hash_halves
from c3.regression import fit_once
from c3.sim.positions import simulate_positions
from c3.splines import Spline
from c3.stats import doc_boot_counts, percentile_ci

ROOT = Path(__file__).resolve().parents[1]


def _frames(scenario, n_docs=120, seed=1):
    df = simulate_positions(scenario, n_docs=n_docs, seed=seed)
    pos, pools = position_and_pool_frames(df, pool_sizes=POOL_SIZES, n_pools_per_size=N_POOLS,
                                          seed=0, retain=RETAIN, min_state_size=max(POOL_SIZES))
    codes = doc_codes_of(pos)
    return df, pos, pools, codes, doc_boot_counts(codes.size, 400, seed=0)


def test_pool_frame_links_back_to_positions():
    _, pos, pools, _, _ = _frames("uniform", n_docs=10)
    assert np.array_equal(pos.ell[pools.src_row], pools.ell)
    assert np.array_equal(pos.doc_id[pools.src_row], pools.doc_id)


def test_delta_sign_is_conf_minus_agree_and_m1_is_exact():
    _, pos, pools, codes, counts = _frames("uniform", n_docs=60)
    d = delta_m(pools, pos, codes, counts)
    # m=1: equal-weight states within docs, equal-weight docs (all states equal size here)
    exact = np.mean(pos.conf - pos.agree)
    assert d.point[d.index(1)] == pytest.approx(exact, abs=1e-12)
    # Noisy confidence makes committed tokens overconfident: positive under the v2 sign.
    assert d.point[d.index(8)] > 0.02


def test_delta_near_zero_when_calibrated():
    _, pos, pools, codes, counts = _frames("calibrated")
    d = delta_m(pools, pos, codes, counts)
    assert np.all(np.abs(d.point) < 0.015)


def test_decomposition_is_exact_and_field_term_zero_at_m1():
    _, pos, pools, codes, counts = _frames("context")
    pi_own, pi_ctx, _ = fit_on_half(pos, codes)            # in-sample: sums are preserved
    dec = decompose(pools, pos, pi_own, pi_ctx, codes, counts)   # asserts the identity
    # With equal-size states and in-sample fits, both calibrators reproduce the
    # overall accuracy, so the m=1 field term vanishes up to IRLS convergence.
    assert abs(dec.field.point[dec.field.index(1)]) < 1e-6


def test_pool_weighted_calibrators_zero_the_m1_field_term_with_unequal_states():
    """States of different sizes: only calibrators fitted with the pools' own
    weighting (1 / (|state| x states per doc)) make the m=1 field term vanish."""
    import pandas as pd

    from c3.calibrators import pool_weights

    df = simulate_positions("context", n_docs=80, seed=4)
    rng = np.random.default_rng(0)
    keep = (df.state_id % 2 == 0) | (rng.random(len(df)) < 0.4)   # odd states shrink to ~40%
    df = df[keep].reset_index(drop=True)
    pos, pools = position_and_pool_frames(df, pool_sizes=POOL_SIZES, n_pools_per_size=N_POOLS,
                                          seed=0, retain=RETAIN, min_state_size=32)
    codes = doc_codes_of(pos)
    counts = doc_boot_counts(codes.size, 100)
    w = pool_weights(pos)
    assert np.allclose(np.bincount(np.searchsorted(codes, pos.doc_id), weights=w), 1.0)
    po, pc, _ = fit_on_half(pos, codes, weighting="pool")
    dec = decompose(pools, pos, po, pc, codes, counts)
    assert abs(dec.field.point[dec.field.index(1)]) < 1e-5
    assert abs(dec.residual.point[dec.residual.index(1)]) < 1e-5


def test_field_term_detects_context_dependent_miscalibration():
    _, pos, pools, codes, counts = _frames("context", n_docs=200)
    half = hash_halves(codes, 0)
    report = np.isin(pos.doc_id, codes[half == 1])
    pi_own, pi_ctx, _ = fit_on_half(pos, codes[half == 0])
    dec = decompose(pools, pos, pi_own, pi_ctx, codes, counts, pos_mask=report)
    f8, draws = dec.field.contrast(8, 1)
    assert f8 > 0.012 and percentile_ci(draws)[0] > 0


def test_residual_exposes_a_misspecified_context_calibrator():
    """State-varying noise changes the slope of pi(l, s); an additive calibrator
    cannot follow it, and the Prop. 1 residual at large m shows the misfit."""
    _, pos, pools, codes, counts = _frames("context", n_docs=200)
    half = hash_halves(codes, 0)
    report = np.isin(pos.doc_id, codes[half == 1])
    res = {}
    for inter in (True, False):
        pi_own, pi_ctx, _ = fit_on_half(pos, codes[half == 0], interactions=inter)
        dec = decompose(pools, pos, pi_own, pi_ctx, codes, counts, pos_mask=report)
        res[inter] = abs(dec.residual.point[dec.residual.index(32)])
    assert res[True] < 0.5 * res[False]


def test_field_term_near_zero_when_miscalibration_is_the_same_everywhere():
    _, pos, pools, codes, counts = _frames("uniform", n_docs=200)
    half = hash_halves(codes, 0)
    pi_own, pi_ctx, _ = fit_on_half(pos, codes[half == 0])
    report = np.isin(pos.doc_id, codes[half == 1])
    dec = decompose(pools, pos, pi_own, pi_ctx, codes, counts, pos_mask=report)
    assert abs(dec.field.point[dec.field.index(32)]) < 0.006


def test_leave_one_document_out_never_sees_the_predicted_document():
    df, pos, _, codes, _ = _frames("context", n_docs=12)
    folds = {int(d): int(d) for d in codes}
    own_a, ctx_a, _ = crossfit_predictions(pos, folds, n_knots=5)
    df2 = df.copy()
    target = int(codes[0])
    df2.loc[df2.doc_id == target, "agree"] = 1 - df2.loc[df2.doc_id == target, "agree"]
    pos2 = PositionFrame.from_frame(df2)
    own_b, ctx_b, _ = crossfit_predictions(pos2, folds, n_knots=5)
    sel = pos.doc_id == target
    assert np.allclose(own_a[sel], own_b[sel]) and np.allclose(ctx_a[sel], ctx_b[sel])
    assert not np.allclose(own_a[~sel], own_b[~sel])


def test_icc_detects_shared_errors_only_when_present():
    for scenario, expect_positive in (("uniform", False), ("misreading", True)):
        _, pos, _, codes, counts = _frames(scenario, n_docs=200)
        _, pi_ctx, _ = fit_on_half(pos, codes)
        r = icc_residuals(pos, pi_ctx, codes, counts)
        assert (r["ci"][0] > 0) == expect_positive, (scenario, r)


def test_beta_z_without_fe_is_negative_under_a_field_effect():
    _, _, pools, _, _ = _frames("context", n_docs=200)
    w = pools.winners()
    sp = Spline.fit(w.ell, n_knots=9)
    no_fe = fit_once(w, sp, "eq2", fe=False).coef("beta_z")
    fe = fit_once(w, sp, "eq2", fe=True).coef("beta_z")
    assert no_fe < -0.1
    assert fe > no_fe + 0.1   # state intercepts absorb most of the field term


def test_no_fe_fit_matches_statsmodels_glm():
    sm = pytest.importorskip("statsmodels.api")
    _, _, pools, _, _ = _frames("uniform", n_docs=30)
    w = pools.winners()
    sp = Spline.fit(w.ell, n_knots=5)
    ours = fit_once(w, sp, "eq2", fe=False)
    from c3.regression import build_design
    X, _ = build_design(w, sp, "eq2")
    glm = sm.GLM(w.agree, sm.add_constant(X), family=sm.families.Binomial()).fit()
    assert np.allclose(ours.beta, glm.params[1:], atol=1e-5)


def test_native_gap_rebuilds_the_commit_set():
    # Two decoder states; with k=2 and min_masked=1 the two most confident
    # positions of each state are the committed ones.
    import pandas as pd
    df = pd.DataFrame({
        "doc_id": [0] * 8, "state_id": [0] * 4 + [1] * 4, "position": [0, 1, 2, 3] * 2,
        "ell": np.zeros(8), "conf": [0.9, 0.2, 0.8, 0.1, 0.3, 0.95, 0.4, 0.6],
        "agree": [1, 0, 0, 0, 0, 1, 0, 1], "mask_kind": ["decoder"] * 8,
        "mask_rate": [0.5] * 8, "n_masked": [4] * 8, "step": [0] * 4 + [1] * 4})
    pos = PositionFrame.from_frame(df)
    codes = np.array([0])
    g = native_gap(pos, codes, doc_boot_counts(1, 5), k=2, min_masked=1)
    # committed: state 0 -> pos 0 (.9,a1), pos 2 (.8,a0); state 1 -> pos 1 (.95,a1), pos 3 (.6,a1)
    expected = np.mean([0.9 - 1, 0.8 - 0, 0.95 - 1, 0.6 - 1])
    assert g.point[0] == pytest.approx(expected)


def test_winner_gap_restricted_to_report_half():
    _, pos, pools, codes, counts = _frames("uniform", n_docs=40)
    half = hash_halves(codes, 0)
    mask = np.isin(pos.doc_id, codes[half == 1])
    g = winner_gap(pools, pos, pos.conf, pos.agree, codes, counts, pos_mask=mask)
    assert g.n_docs.max() == int((half == 1).sum())


def test_hash_halves_do_not_depend_on_which_documents_are_present():
    ids = np.arange(100)
    full = hash_halves(ids, 7)
    part = hash_halves(ids[::3], 7)
    assert np.array_equal(full[::3], part)
    assert 30 < full.sum() < 70


def test_analyze_traces_end_to_end(tmp_path):
    out = tmp_path / "res"
    cmd = [sys.executable, str(ROOT / "scripts" / "analyze_traces.py"),
           "--traces", "sim:context,n_docs=40", "--split", "kfold-doc", "--n-boot", "200",
           "--eq-boot", "20", "--no-smooth-ece", "--macro-prefix", "Sim", "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT)
    res = json.loads((out / "results.json").read_text())
    r = res["by_mask_kind"]["uniform"]
    assert [row["level"] for row in r["delta"]] == list(POOL_SIZES)
    assert {"eq2_fe", "eq2_nofe", "eq3_fe", "eq3_nofe"} <= set(r["regression"])
    assert res["provenance"]["git_commit"]
    macros = (out / "macros.tex").read_text()
    names = [line.split("{\\")[1].split("}")[0] for line in macros.splitlines()
             if line.startswith("\\newcommand")]
    assert names and not any(ch.isdigit() for n in names for ch in n)
    assert (out / "fig_gap_uniform.png").exists()


def test_winner_gap_leaves_the_callers_mask_alone():
    from c3 import POOL_SIZES
    from c3.assemble import position_and_pool_frames
    from c3.endpoints.e1 import winner_gap
    from c3.sim.trajectories import simulate_design
    from c3.stats import doc_boot_counts

    df = simulate_design("noisy", n_docs=20, ks=(4,), seed=0)
    df = df[df.mask_kind == "uniform"].reset_index(drop=True)
    pos, pools = position_and_pool_frames(df, pool_sizes=POOL_SIZES, seed=0, retain=1,
                                          min_state_size=max(POOL_SIZES))
    mask = pos.doc_id % 2 == 1
    before = mask.copy()
    codes = np.unique(pos.doc_id[mask])
    winner_gap(pools, pos, pos.conf, pos.agree, codes, doc_boot_counts(codes.size, 20, 0), mask)
    assert np.array_equal(mask, before)
