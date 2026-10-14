"""Set-level C3 on the JSON probe: exact joint correctness and the knob rules."""
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from c3.probe import RHOS
from c3.probe.instances import ProbeSpec, eval_instances, sample_values
from c3.probe.truth import MASKED, exact_joint_correct, marginals
from tests.test_probe import brute_conditional, toy_spec

ROOT = Path(__file__).resolve().parents[1]


def test_exact_joint_correctness_equals_brute_force():
    spec = toy_spec(seed=3)
    rng = np.random.default_rng(0)
    for trial in range(60):
        sch = spec.schemas[trial % len(spec.schemas)]
        rho = float(RHOS[trial % len(RHOS)])
        x = sample_values(sch, rho, rng)
        n = len(sch.fields)
        obs = np.where(rng.random(n) < 0.4, x, MASKED)
        masked = np.flatnonzero(obs == MASKED)
        if masked.size == 0:
            continue
        S = sorted(rng.choice(masked, size=int(rng.integers(1, masked.size + 1)), replace=False).tolist())
        vals = [int(rng.integers(sch.fields[i].n)) for i in S]
        cond = brute_conditional(sch, rho, obs, S)
        assert exact_joint_correct(sch, rho, obs, S, vals) == pytest.approx(cond[tuple(vals)], abs=1e-12)
        marg = marginals(sch, rho, obs, S)
        assert exact_joint_correct(sch, rho, obs, S, vals) <= min(marg[i][v] for i, v in zip(S, vals)) + 1e-12
        assert exact_joint_correct(sch, rho, obs, S, [-1] + vals[1:]) == 0.0


@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from c3.traces.models import LocalMDLM
    from c3.training.model import MaskedDiffusionLM, ModelConfig
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as exc:                              # offline without a cache
        pytest.skip(f"gpt2 tokenizer unavailable: {exc}")
    torch.manual_seed(0)
    m = LocalMDLM(path="", device="cpu")
    m.model = MaskedDiffusionLM(ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64)).eval()
    m.tokenizer = tok
    return m


def test_every_rule_decodes_every_field_and_the_summary_matches_speeds(tiny):
    from c3.probe.grid import prepare, run_trajectories, summarise, summarise_matched
    from scripts.probe_c3set import calibrate
    from c3.infill import build_policy

    spec = ProbeSpec.load(ROOT / "artifacts" / "probe" / "spec.json")
    args = SimpleNamespace(model="local-110m", n_calib=40, calib_ks=[1, 2], seed=0, batch_size=8,
                           token_features=["median_ell", "mask_rate", "log_n_masked"], no_field=False)
    calib_df, table, tok, setcal = calibrate(tiny, spec, args, log=lambda *_: None)
    assert np.allclose(table.groupby("doc_id")["weight"].sum(), 1.0)     # instances weigh equally
    ev = eval_instances(spec, per_rho=200)
    pick = [ev[i] for i in (0, 1, 2, 200, 201, 400, 600, 800)]             # rho 0 .. 1
    prepared = prepare(spec, pick, tiny.tokenizer, "gpt2")
    rows = []
    cfgs = [("random", 2, None), ("confidence", 2, None), ("confidence", 4, None)]
    cfgs += [(c["policy"], next(v for k, v in c.items() if k != "policy"), build_policy(c, tok, setcal))
             for c in ({"policy": "threshold", "tau": 0.01}, {"policy": "entropy_bounded", "gamma": 5.0},
                       {"policy": "c3a", "alpha": 0.9}, {"policy": "c3indep", "alpha": 0.9},
                       {"policy": "c3set", "alpha": 0.9}, {"policy": "c3set", "alpha": 0.999})]
    for rule, knob, pol in cfgs:
        got = run_trajectories(tiny, spec, prepared, knob, rule, 0, 8, max_cells=200_000, policy=pol)
        for p in prepared:
            mine = [r for r in got if r["instance_id"] == p.inst.instance_id]
            assert sum(r["n_commit"] for r in mine) == len(p.inst.values)      # every field, once
            assert all(r["n_commit"] >= 1 for r in mine)
            assert all(0.0 <= r["J"] <= 1.0 for r in mine)
        if rule in ("c3indep", "c3set"):
            assert all(np.isfinite(r["P_ind"]) for r in got)
        if rule == "c3set":
            assert all(np.isfinite(r["P_set"]) for r in got)
        rows += got
    ident = summarise(rows, n_boot=100)["identity"]
    assert ident["max_abs_err_nats"] < 1e-6
    m = summarise_matched(rows, speeds=(2.0,), n_boot=100)
    assert m["max_A_at_rho0"] < 1e-9
    assert {"c3set", "confidence", "threshold"} <= set(m["matched"])
    assert "c3set-confidence" in m["contrasts"]
    assert set(m["calibration"]) == {"c3set", "c3indep", "c3a"}
