"""E3 (T7): decoder-visited vs random states at equal confidence."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from c3.calibrators import pool_weights
from c3.endpoints import e3
from c3.frame import PositionFrame
from c3.logit import _sigmoid, fit_fe_logit
from c3.sim.trajectories import distance_to_visible, simulate_design
from c3.splines import Spline
from c3.stats import doc_boot_counts

ROOT = Path(__file__).resolve().parents[1]


def _arms(scenario, n_docs=80, seed=1, **kw):
    df = simulate_design(scenario, n_docs=n_docs, ks=(4,), seed=seed, **kw)
    codes = np.unique(df.doc_id)
    fit, rep = codes[codes % 2 == 0], codes[codes % 2 == 1]
    sel = df.mask_kind.isin(["decoder", "uniform_matched"])
    f = df[df.doc_id.isin(fit) & sel]
    nu = e3.E3Nuisance.from_positions(f.ell.to_numpy(), f.conf.to_numpy(), n_ell_bins=50)
    D = PositionFrame.from_frame(df[(df.mask_kind == "decoder") & df.doc_id.isin(rep)]
                                 .reset_index(drop=True))
    R = PositionFrame.from_frame(df[(df.mask_kind == "uniform_matched") & df.doc_id.isin(rep)]
                                 .reset_index(drop=True))
    G = PositionFrame.from_frame(df[(df.mask_kind == "uniform") & df.doc_id.isin(rep)]
                                 .reset_index(drop=True))
    return D, R, G, nu, np.unique(D.doc_id)


def test_rate_bins_are_centred_on_the_uniform_grid():
    b = e3.rate_bin(np.array([0.04, 0.05, 0.1, 0.149, 0.15, 0.9, 0.949, 0.95, 1.0]))
    assert b.tolist() == [-1, 0, 0, 0, 1, 8, 8, -1, -1]
    assert np.allclose(e3.RATE_CENTRES, np.arange(1, 10) / 10)


def test_distance_to_visible():
    m = np.array([1, 1, 0, 1, 1, 1, 0, 1], bool)
    assert distance_to_visible(m).tolist() == [2, 1, 1, 2, 1, 1]
    assert distance_to_visible(np.ones(4, bool)).tolist() == [4, 4, 4, 4]


def test_cells_reproduce_the_position_level_fit():
    """The sufficient-statistic reduction is what makes 10k refits affordable; it must
    not change the estimate."""
    D, R, _, nu, codes = _arms("noisy")
    r = e3.e3_compare(D, R, codes, doc_boot_counts(codes.size, 20, seed=0), nu, by_rate=False)
    wd, wr, bd, br, _ = e3._weights_by_arm(D, R, False)
    ell = np.r_[D.ell, R.ell]
    y = np.r_[D.agree, R.agree]
    t = np.r_[np.ones(len(D)), np.zeros(len(R))]
    w = np.r_[wd, wr]
    b = np.r_[bd, br]
    k = w > 0
    sp = Spline(nu.knots)
    F = sp.transform(ell[k])
    n_f = sp.n_columns
    X = np.column_stack([F, t[k, None], F * t[k, None]])
    _, g = np.unique(b[k], return_inverse=True)
    fit = fit_fe_logit(y[k], X, g, weights=w[k] * k.sum() / w[k].sum(), ridge_fe=1e-8,
                       ridge_beta=1e-6)
    eta_r = fit.alpha[g] + F @ fit.beta[:n_f]
    eta_d = eta_r + fit.beta[n_f] + F @ fit.beta[n_f + 1:]
    dd = t[k] == 1
    direct = np.sum(w[k][dd] * (_sigmoid(eta_r[dd]) - _sigmoid(eta_d[dd]))) / w[k][dd].sum()
    assert abs(r["type_effect"]["value"] - direct) < 2e-3


def test_a_known_decoder_shift_is_recovered():
    D, R, G, nu, codes = _arms("decoder_shift", shift=1.0)
    counts = doc_boot_counts(codes.size, 200, seed=0)
    w = np.where(e3.rate_bin(D.mask_rate) >= 0, pool_weights(D), 0.0)
    truth = np.sum(w * (_sigmoid(D.ell) - _sigmoid(D.ell - 1.0))) / w.sum()
    for ref, match in ((R, False), (G, True)):
        r = e3.e3_compare(D, ref, codes, counts, nu, match_rates=match, by_rate=False)
        te = r["type_effect"]
        assert te["ci"][0] > 0 and abs(te["value"] - truth) < 0.03
        assert r["type_test"]["p"] < 0.01
        assert all(v > 0 for v in r["difference_curve"]["value"][3:-3])


def test_calibrated_states_show_no_type_effect():
    D, R, _, nu, codes = _arms("calibrated", n_docs=200, seed=3)
    r = e3.e3_compare(D, R, codes, doc_boot_counts(codes.size, 200, seed=0), nu, by_rate=True)
    te = r["type_effect"]
    assert abs(te["value"]) < 0.015
    assert te["ci"][0] < te["value"] < te["ci"][1]
    assert len(r["by_rate"]) == 9
    rel = r["reliability"]
    assert len(rel["difference"]) == len(rel["decoder"]["bins"]) == 15


def test_grid_reference_is_reweighted_to_the_decoder_rate_mix():
    D, _, G, nu, codes = _arms("calibrated")
    wd, wr, bd, br, f = e3._weights_by_arm(D, G, True)
    for b in range(9):
        if wr[br == b].sum() > 0:
            assert np.isclose(wr[br == b].sum(), wd[bd == b].sum())


def test_sup_t_band_and_null():
    rng = np.random.default_rng(0)
    cov = 0.5 ** np.abs(np.subtract.outer(np.arange(10), np.arange(10)))
    draws = rng.multivariate_normal(np.zeros(10), cov, size=2000)
    r = e3.sup_t(np.zeros(10), draws)
    assert r["p"] > 0.5 and r["critical"] > 1.96
    r = e3.sup_t(np.full(10, 4.0), draws + 4.0)     # bootstrap draws centre on the estimate
    assert r["p"] < 0.01
    assert all(lo < 4 < hi for lo, hi in r["band"])


def _run(*args, check=True):
    return subprocess.run([sys.executable, *map(str, args)], check=check, capture_output=True,
                          text=True, cwd=ROOT)


def test_cli_requires_an_explicit_half_and_writes_results(tmp_path):
    common = ["scripts/analyze_e3.py", "--traces", "sim-design:decoder_shift,n_docs=30,shift=0.5",
              "--split", "half", "--n-boot", "60", "--no-by-rate"]
    r = _run(*common, "--out", tmp_path / "x", check=False)
    assert r.returncode != 0 and "--half" in (r.stderr + r.stdout)
    _run(*common, "--half", "report", "--out", tmp_path / "a")
    res = json.loads((tmp_path / "a" / "results.json").read_text())
    assert set(res["comparisons"]) == {"decoder:topk_conf:k4_vs_matched",
                                       "decoder:topk_conf:k4_vs_grid"}
    assert res["comparisons"]["decoder:topk_conf:k4_vs_matched"]["type_effect"]["value"] > 0
    macros = (tmp_path / "a" / "macros.tex").read_text()
    assert "\\EThreeTypeEffectTopkConfKFourMatched}" in macros


def test_development_mode_never_reads_the_report_half(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_e3

    codes = np.arange(40)
    man = tmp_path / "m.json"
    man.write_text(json.dumps({str(i): int(i % 2) for i in codes}))
    nu, est, scope = analyze_e3.halves(codes, "half", "fit", str(man), 0)
    assert set(nu) | set(est) <= set(codes[codes % 2 == 0])
    assert not set(nu) & set(est) and "DEVELOPMENT" in scope
    nu, est, _ = analyze_e3.halves(codes, "half", "report", str(man), 0)
    assert set(nu) == set(codes[codes % 2 == 0]) and set(est) == set(codes[codes % 2 == 1])
    with pytest.raises(SystemExit):
        analyze_e3.halves(codes, "half", None, str(man), 0)
