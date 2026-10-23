"""T13 ablation 2: the recalibration methods (c3.recal) and their comparison script."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from c3 import recal
from c3.calibrators import ContextLogistic
from c3.frame import PositionFrame
from c3.logit import _sigmoid

ROOT = Path(__file__).resolve().parents[1]


def _pos(ell, agree, rate=None, n_per_state=8):
    n = ell.size
    rate = np.full(n, 0.5) if rate is None else rate
    st = np.arange(n) // n_per_state
    return PositionFrame.from_frame(pd.DataFrame({
        "doc_id": st // 4, "state_id": st, "position": np.arange(n) % n_per_state,
        "ell": ell, "agree": agree, "mask_rate": rate,
        "n_masked": np.full(n, n_per_state), "step": np.zeros(n, int)}))


def _draw(n=40_000, T=2.0, seed=0):
    rng = np.random.default_rng(seed)
    ell = rng.normal(1.0, 2.5, n)
    return ell, (rng.random(n) < _sigmoid(ell / T)).astype(float)


def test_identity_is_the_raw_confidence():
    ell, y = _draw(1000)
    p = recal.make("none").fit(_pos(ell, y)).predict(_pos(ell, y))
    assert np.allclose(p, _sigmoid(ell))


def test_global_temperature_recovers_a_known_t():
    ell, y = _draw(T=2.0)
    cal = recal.make("temperature").fit(_pos(ell, y))
    assert abs(cal.T - 2.0) < 0.1


def test_rate_temperature_recovers_one_t_per_mask_rate_bin():
    e1, y1 = _draw(T=0.7, seed=1)
    e2, y2 = _draw(T=2.5, seed=2)
    ell, y = np.r_[e1, e2], np.r_[y1, y2]
    rate = np.r_[np.full(e1.size, 0.2), np.full(e2.size, 0.8)]
    cal = recal.make("temperature_rate").fit(_pos(ell, y, rate))
    from c3.controls import temperature_bin
    assert abs(cal.T[temperature_bin(np.array([0.2]))[0]] - 0.7) < 0.05
    assert abs(cal.T[temperature_bin(np.array([0.8]))[0]] - 2.5) < 0.2
    with pytest.raises(KeyError):
        cal.predict_arrays(np.zeros(3), {"median_ell": 0.0})


def test_histogram_bins_hold_equal_mass_and_their_accuracy():
    ell, y = _draw(15_000)
    cal = recal.make("histogram").fit(_pos(ell, y))
    assert cal.edges.size == 14
    b = np.searchsorted(cal.edges, ell, side="right")
    share = np.bincount(b) / ell.size
    assert np.allclose(share, 1 / 15, atol=0.002)
    acc = np.array([y[b == j].mean() for j in range(15)])
    assert np.allclose(cal.predict_arrays(ell), acc[b])


def test_histogram_with_heavy_ties_never_leaves_an_empty_bin():
    rng = np.random.default_rng(0)
    ell = np.where(rng.random(5000) < 0.7, 9.0, rng.normal(size=5000))   # 70% saturated
    y = (rng.random(5000) < _sigmoid(ell)).astype(float)
    cal = recal.make("histogram").fit(_pos(ell, y))
    assert cal.edges.size < 14 and np.isfinite(cal.values).all()
    assert cal.predict_arrays(np.array([9.0]))[0] == pytest.approx(y[ell == 9.0].mean())


@pytest.mark.parametrize("method", recal.METHODS)
def test_every_method_round_trips_through_json(tmp_path, method):
    ell, y = _draw(6000)
    rng = np.random.default_rng(3)
    pos = _pos(ell, y, rate=rng.choice([0.2, 0.5, 0.8], ell.size))
    cal = recal.make(method).fit(pos)
    recal.save(cal, tmp_path / "tok.json")
    back = recal.load(tmp_path / "tok.json")
    assert np.allclose(back.predict(pos), cal.predict(pos))
    p = cal.predict(pos)
    assert p.shape == ell.shape and np.all((p >= 0) & (p <= 1))
    # the decode-time interface, with the features a decoder has at a step
    field = {"median_ell": 0.5, "mask_rate": 0.5, "log_n_masked": np.log(8)}
    q = back.predict_arrays(ell[:50], field)
    assert q.shape == (50,) and np.all((q >= 0) & (q <= 1))


def test_context_files_written_before_t13_still_load(tmp_path):
    ell, y = _draw(4000)
    ctx = ContextLogistic(features=("mask_rate",)).fit(_pos(ell, y, np.linspace(0.1, 0.9, ell.size)))
    ctx.save(tmp_path / "tok.json")
    back = recal.load(tmp_path / "tok.json")
    assert isinstance(back, ContextLogistic)


def test_c3a_runs_on_any_method():
    from c3.decode.policies import StepContext
    from c3.infill import build_policy

    ell, y = _draw(4000)
    pos = _pos(ell, y)
    stats = {"ell": np.array([3.0, -1.0, 0.5]), "entropy": np.array([0.1, 2.0, 1.0])}
    ctx = StepContext(step=0, positions=np.array([4, 5, 6]), n_masked_total=3, seq_len=10)
    for m in recal.METHODS:
        pol = build_policy({"policy": "c3a", "alpha": 0.3}, recal.make(m).fit(pos))
        order, n = pol.select(stats, ctx)
        assert order[0] == 0 and 1 <= n <= 3


def test_cli_compares_every_method_and_saves_calibrators(tmp_path):
    from c3.sim.trajectories import simulate_design

    df = simulate_design("noisy", n_docs=40, ks=(4,), seed=4)
    trace = tmp_path / "trace.parquet"
    df.to_parquet(trace, index=False)
    manifest = tmp_path / "dev.manifest.json"
    manifest.write_text(json.dumps({str(d): int(d % 2) for d in sorted(df.doc_id.unique())}))
    out = tmp_path / "recal.json"
    subprocess.run([sys.executable, "scripts/compare_recalibration.py", "--traces", str(trace),
                    "--manifest", str(manifest), "--arms", "uniform", "decoder",
                    "--save-arms", "decoder", "--save-calibrators", str(tmp_path / "cal"),
                    "--n-boot", "50", "--out", str(out)], check=True, cwd=ROOT,
                   capture_output=True, text=True)
    res = json.loads(out.read_text())
    for arm in ("uniform", "decoder"):
        ms = res["arms"][arm]["methods"]
        assert set(ms) == set(recal.METHODS)
        assert "vs_none" not in ms["none"] and "vs_context" in ms["none"]
        # no recalibration leaves the raw gap (the simulator stores conf = sigma(ell))
        assert abs(ms["none"]["gap_hi_vs_raw"]["value"]) < 1e-9
    for m in recal.METHODS:
        assert (tmp_path / "cal" / m / "tok.json").exists()
    assert "\\begin{tabular}" in out.with_suffix(".tex").read_text()
