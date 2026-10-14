"""Batched FineWeb infilling and the calibrated commit policies, on the CPU toy."""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from c3.calibrators import ContextLogistic
from c3.decode.loop import decode
from c3.decode.policies import (C3a, C3Indep, C3Set, StepContext, TopKConfidence)
from c3.frame import PositionFrame
from c3.infill import (StepLog, build_policy, config_key, infill_batch, interp_at, interp_draws,
                       load_configs, select_docs)
from c3.setcal import SetCalibrator, build_set_table
from c3.sim.trajectories import simulate_design
from c3.traces.models import ToyDiffusionLM

L, SPAN = 64, (16, 48)


class MultiRefToy(ToyDiffusionLM):
    """The toy, but each row learns its own document from its first 8 tokens."""

    def logits(self, input_ids):
        outs = []
        for row in input_ids:
            self.reference = self.refs[row[:8].cpu().numpy().tobytes()]
            outs.append(ToyDiffusionLM.logits(self, row[None]))
        return torch.cat(outs)


def _docs(n=5, seed=0):
    rng = np.random.default_rng(seed)
    docs = [(100 + i, rng.integers(0, 60, L)) for i in range(n)]
    model = MultiRefToy(noise_dispersion=0.3)
    model.refs = {ids[:8].tobytes(): ids % model.vocab_size for _, ids in docs}
    return model, docs


@pytest.fixture(scope="module")
def calibrators():
    df = simulate_design("calibrated", n_docs=30, doc_len=64, ks=(4,), matched=False, seed=1)
    df = df[df["mask_kind"] == "decoder"].reset_index(drop=True)
    c = df["conf"].to_numpy()
    df["entropy"] = -(c * np.log(c) + (1 - c) * np.log1p(-c))
    tok = ContextLogistic(features=("median_ell", "mask_rate", "log_n_masked")).fit(
        PositionFrame.from_frame(df))
    table = build_set_table(df, tok.predict(PositionFrame.from_frame(df)))
    return tok, SetCalibrator(use_field=True).fit(table, table["weight"].to_numpy())


CONFIGS = [{"policy": "topk_conf", "k": 2}, {"policy": "margin", "k": 3},
           {"policy": "pc_sampler", "k": 2}, {"policy": "threshold", "tau": 0.5},
           {"policy": "entropy_bounded", "gamma": 0.8},
           {"policy": "klass", "tau": 0.3, "eps_kl": 0.5},
           {"policy": "c3a", "alpha": 0.4}, {"policy": "c3indep", "alpha": 0.6},
           {"policy": "c3set", "alpha": 0.6}]


@pytest.mark.parametrize("cfg", CONFIGS, ids=config_key)
def test_batched_infill_is_the_single_sequence_decode(cfg, calibrators):
    model, docs = _docs()
    tok, setcal = calibrators
    policy = build_policy(cfg, tok, setcal)
    got = infill_batch(model, docs, policy, span=SPAN)
    for (doc_id, ids), res in zip(docs, got):
        x = ids.copy()
        x[SPAN[0]:SPAN[1]] = model.mask_token_id
        model.reference = model.refs[ids[:8].tobytes()]
        single = decode(_Single(model, ids), x, build_policy(cfg, tok, setcal), record=False)
        assert np.array_equal(single.tokens[SPAN[0]:SPAN[1]], res["tokens"]), (cfg, doc_id)
        assert single.n_forward == res["n_forward"]
        assert res["acc"] == pytest.approx(np.mean(res["tokens"] == ids[SPAN[0]:SPAN[1]]))
        assert np.array_equal(single.tokens[:SPAN[0]], ids[:SPAN[0]])


class _Single:
    """One document's view of the multi-reference toy, for ``decode``."""

    def __init__(self, toy, ids):
        self.toy, self.ref = toy, ids
        self.mask_token_id, self.input_device, self.name = toy.mask_token_id, None, "toy"

    def logits(self, t):
        return self.toy.logits(t)


@pytest.mark.parametrize("k", [1, 3, 8])
def test_top_k_infill_speed_is_k(k):
    model, docs = _docs(n=3)
    n = SPAN[1] - SPAN[0]
    for res in infill_batch(model, docs, TopKConfidence(k), span=SPAN):
        assert res["n_forward"] == math.ceil(n / k)
        assert res["tokens_per_step"] == pytest.approx(n / math.ceil(n / k))
        assert 0.0 <= res["step_all_right"] <= 1.0


@pytest.mark.parametrize("name", ["c3a", "c3indep", "c3set"])
def test_calibrated_policies_span_one_per_step_to_all_at_once(name, calibrators):
    model, docs = _docs(n=2)
    tok, setcal = calibrators
    n = SPAN[1] - SPAN[0]
    slow = infill_batch(model, docs, build_policy({"policy": name, "alpha": 1e-15}, tok, setcal),
                        span=SPAN)
    fast = infill_batch(model, docs, build_policy({"policy": name, "alpha": 1.0}, tok, setcal),
                        span=SPAN)
    assert all(r["n_forward"] >= n // 2 for r in slow)        # nearly one token per step
    assert all(r["n_forward"] == 1 for r in fast)             # everything in one step


class _Fixed:
    """A stand-in calibrator returning fixed probabilities."""

    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float64)

    def predict_arrays(self, ell, field):
        return self.p

    def predict(self, feats):
        return self.p


def _stats(n):
    return {"ell": np.linspace(1, 2, n), "entropy": np.full(n, 0.1), "conf": np.full(n, 0.8)}


def test_c3_indep_takes_the_largest_prefix_with_product_above_one_minus_alpha():
    ctx = StepContext(step=0, positions=np.arange(3), n_masked_total=3, seq_len=10)
    pol = C3Indep(alpha=0.06, calibrator=_Fixed([0.9, 0.99, 0.95]))
    order, n = pol.select(_stats(3), ctx)
    assert order.tolist() == [1, 2, 0] and n == 2           # 0.99, 0.9405 >= 0.94 > 0.846


def test_c3_set_uses_the_running_minimum_of_the_set_probability():
    ctx = StepContext(step=0, positions=np.arange(4), n_masked_total=4, seq_len=10)
    for alpha, want in [(0.04, 1), (0.06, 3), (0.5, 4)]:
        pol = C3Set(alpha=alpha, calibrator=_Fixed([0.99, 0.98, 0.97, 0.96]),
                    set_calibrator=_Fixed([0.99, 0.95, 0.97, 0.80]))
        assert pol.select(_stats(4), ctx)[1] == want, alpha
    pol = C3a(alpha=0.025, calibrator=_Fixed([0.99, 0.98, 0.97, 0.96]))
    assert pol.select(_stats(4), ctx)[1] == 2
    pol = C3a(alpha=1e-9, calibrator=_Fixed([0.5] * 4))
    assert pol.select(_stats(4), ctx)[1] == 1                # always at least one


def test_calibrated_policies_need_the_sequence_length():
    pol = C3a(alpha=0.1, calibrator=_Fixed([0.9]))
    with pytest.raises(ValueError, match="seq_len"):
        pol.select(_stats(1), StepContext(step=0, positions=np.arange(1)))


def test_interpolation_is_exact_on_log_linear_curves_and_nan_outside():
    tps = np.array([1.0, 2.0, 4.0, 8.0])
    acc = 0.9 - 0.1 * np.log2(tps)
    for t in (1.0, 1.5, 2.0, 3.0, 4.0, 8.0):
        assert interp_at(tps, acc, t) == pytest.approx(0.9 - 0.1 * np.log2(t))
    assert math.isnan(interp_at(tps, acc, 0.5)) and math.isnan(interp_at(tps, acc, 16.0))
    # Two configurations at one speed are averaged.
    assert interp_at(np.array([2.0, 2.0, 4.0]), np.array([0.5, 0.7, 0.4]), 2.0) == pytest.approx(0.6)
    draws = interp_draws(np.tile(tps, (3, 1)), np.tile(acc, (3, 1)), 3.0)
    assert np.allclose(draws, 0.9 - 0.1 * np.log2(3.0))


def test_dev_halves_partition_the_fit_half():
    split = {str(d): int(d % 3 == 0) for d in range(300)}
    fit, rep = select_docs(split, "fit"), select_docs(split, "report")
    dfit, deval = select_docs(split, "dev-fit"), select_docs(split, "dev-eval")
    assert not (fit & rep) and fit | rep == set(range(300))
    assert dfit | deval == fit and not (dfit & deval)
    assert 0.35 < len(dfit) / len(fit) < 0.65


def test_on_policy_log_builds_a_set_table(calibrators):
    model, docs = _docs(n=3)
    log = StepLog.empty()
    infill_batch(model, docs, TopKConfidence(4), span=SPAN, log=log)
    df = log.frame()
    assert df.groupby("state_id")["committed"].sum().eq(4).all()
    assert df["state_id"].nunique() == 3 * 8
    tok, _ = calibrators
    table = build_set_table(df, tok.predict(PositionFrame.from_frame(df)))
    assert np.allclose(table.groupby("doc_id")["weight"].sum(), 1.0)


def test_decode_reports_every_step_to_on_step():
    model, docs = _docs(n=1)
    ids = docs[0][1]
    x = ids.copy()
    x[SPAN[0]:SPAN[1]] = model.mask_token_id
    model.reference = model.refs[ids[:8].tobytes()]
    seen = []
    res = decode(_Single(model, ids), x, TopKConfidence(4),
                 on_step=lambda step, stats, ctx, chosen: seen.append((step, ctx.seq_len,
                                                                       ctx.n_masked_total)))
    assert [s[0] for s in seen] == list(range(res.n_forward))
    assert seen[0][1:] == (L, SPAN[1] - SPAN[0])


def test_configs_load_and_refuse_duplicates(tmp_path):
    p = tmp_path / "grid.json"
    p.write_text(json.dumps({"configs": CONFIGS}))
    assert len(load_configs(p)) == len(CONFIGS)
    p.write_text(json.dumps([CONFIGS[0], CONFIGS[0]]))
    with pytest.raises(ValueError, match="duplicate"):
        load_configs(p)
