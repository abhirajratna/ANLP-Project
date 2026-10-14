"""Set-level calibration on decoder states with known joint truth."""
import numpy as np
import pytest

from c3.calibrators import ContextLogistic, state_field_features
from c3.endpoints.fieldtest import paired_score_test
from c3.frame import PositionFrame, hash_halves
from c3.setcal import SET_GRID, SetCalibrator, build_set_table, prefix_features
from c3.sim.trajectories import set_truth, simulate_design
from c3.stats import doc_boot_counts


def _decoder(err_corr=0.0, n_docs=120, seed=0, scenario="calibrated", mu=3.0):
    # mu=3: mostly confident tokens, as in committed sets, so joint errors matter.
    df = simulate_design(scenario, n_docs=n_docs, doc_len=64, ks=(4,), matched=False,
                         seed=seed, err_corr=err_corr, mu=mu)
    df = df[df["mask_kind"] == "decoder"].reset_index(drop=True)
    c = df["conf"].to_numpy()
    df["entropy"] = -(c * np.log(c) + (1 - c) * np.log1p(-c))     # a stand-in entropy
    return df


def test_vectorised_builder_matches_the_per_state_features():
    df = _decoder(n_docs=6)
    c_hat = df["conf"].to_numpy()
    table = build_set_table(df, c_hat)
    for sid in df["state_id"].unique()[:12]:
        rows = df[df["state_id"] == sid]
        n = len(rows)
        _, f = prefix_features(rows["ell"], rows["entropy"], rows["position"],
                               rows["conf"], np.median(rows["ell"]), rows["mask_rate"].iloc[0],
                               np.log(n))
        t = table[table["state_id"] == sid].sort_values("j")
        js = t["j"].to_numpy() - 1
        assert t["j"].max() <= n and set(t["j"]) <= set(SET_GRID)
        for name in ("u", "min_ell", "eb_gap", "log_size", "adj_frac", "median_ell",
                     "log_n_masked"):
            assert np.allclose(t[name].to_numpy(), f[name][js]), (sid, name)


def test_label_is_all_of_the_top_j_correct():
    df = _decoder(n_docs=4)
    table = build_set_table(df, df["conf"].to_numpy())
    for sid in df["state_id"].unique()[:10]:
        rows = df[df["state_id"] == sid]
        order = np.lexsort((rows["position"].to_numpy(), -rows["conf"].to_numpy()))
        agree = rows["agree"].to_numpy()[order]
        t = table[table["state_id"] == sid]
        for j, y in zip(t["j"], t["y"]):
            assert y == float(agree[:j].min())


def test_documents_carry_equal_weight():
    df = _decoder(n_docs=5)
    table = build_set_table(df, df["conf"].to_numpy())
    assert np.allclose(table.groupby("doc_id")["weight"].sum(), 1.0)


def _fit_and_score(err_corr):
    df = _decoder(err_corr=err_corr, n_docs=400)   # enough documents to resolve the gain
    table = build_set_table(df, df["conf"].to_numpy())     # sim conf is calibrated
    half = hash_halves(table["doc_id"].to_numpy(), 0)
    fit, rep = table[half == 0], table[half == 1]
    cal = SetCalibrator(use_field=False).fit(fit, fit["weight"].to_numpy())
    p_set = cal.predict(rep)
    p_ind = np.exp(rep["log_p_ind"].to_numpy())
    codes = np.unique(rep["doc_id"])
    counts = doc_boot_counts(codes.size, 400, seed=0, label="test-setcal")
    res = paired_score_test(p_ind, p_set, rep["y"].to_numpy(),
                            np.searchsorted(codes, rep["doc_id"].to_numpy()), codes.size,
                            counts, weights=rep["weight"].to_numpy())
    return df, rep, cal, p_set, p_ind, res


@pytest.mark.parametrize("err_corr, sign", [(0.9, 1)])
def test_set_calibrator_learns_correlated_errors(err_corr, sign):
    # Errors of position pairs move together: independence underestimates the chance
    # that a set holding such a pair is all right, and the set model corrects it.
    # (Anticorrelated errors barely move that chance at high confidence, so the
    # negative case is not a useful test of the mechanism.)
    df, rep, cal, p_set, p_ind, res = _fit_and_score(err_corr)
    assert res["helps"], res
    assert np.sign(cal.coef()["adj_frac"]) == sign
    # Closer to the exact joint truth than independence, on prefixes with pairs.
    sub = rep[rep["adj_frac"] > 0].sample(150, random_state=0)
    err_set, err_ind = [], []
    for (_, r), ps, pi in zip(sub.iterrows(), cal.predict(sub), np.exp(sub["log_p_ind"])):
        rows = df[df["state_id"] == r["state_id"]]
        order = np.lexsort((rows["position"].to_numpy(), -rows["conf"].to_numpy()))
        top = order[: int(r["j"])]
        truth = set_truth(rows["position"].to_numpy()[top], rows["truth_p"].to_numpy()[top],
                          err_corr)
        err_set.append(abs(ps - truth))
        err_ind.append(abs(pi - truth))
    assert np.mean(err_set) < np.mean(err_ind)


def test_no_gain_is_declared_when_errors_are_independent():
    *_, res = _fit_and_score(0.0)
    assert not res["helps"], res


def test_calibrators_round_trip_through_json(tmp_path):
    df = _decoder(n_docs=20)
    table = build_set_table(df, df["conf"].to_numpy())
    cal = SetCalibrator(use_field=True).fit(table, table["weight"].to_numpy())
    cal.save(tmp_path / "set.json")
    again = SetCalibrator.load(tmp_path / "set.json")
    assert np.allclose(cal.predict(table), again.predict(table))

    pos = PositionFrame.from_frame(df)
    tok = ContextLogistic(features=("median_ell", "mask_rate", "log_n_masked")).fit(pos)
    tok.save(tmp_path / "tok.json")
    back = ContextLogistic.load(tmp_path / "tok.json")
    assert np.allclose(tok.predict(pos), back.predict(pos))
    # predict_arrays with per-state features equals the frame-based prediction.
    sid = df["state_id"].iloc[0]
    rows = df["state_id"].to_numpy() == sid
    feats = state_field_features(df["ell"].to_numpy()[rows], int(rows.sum()), 64)
    assert np.allclose(back.predict_arrays(df["ell"].to_numpy()[rows], feats),
                       tok.predict(pos.select(rows)))
