"""T8 controls: position, token type, temperature, precision -- one ablation table."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from c3 import POOL_SIZES
from c3 import controls as C
from c3.assemble import position_and_pool_frames
from c3.sim.trajectories import simulate_design
from c3.stats import doc_boot_counts

ROOT = Path(__file__).resolve().parents[1]
HI, LO = max(POOL_SIZES), min(POOL_SIZES)


def test_distance_to_unmasked_matches_the_simulator():
    df = simulate_design("calibrated", n_docs=6, doc_len=40)
    d = C.distance_to_unmasked(df.state_id.to_numpy(), df.position.to_numpy(),
                               df.doc_len.to_numpy())
    assert np.array_equal(d, df.truth_dist.to_numpy())


def test_distance_to_unmasked_sees_the_prompt():
    # one state: positions 5..9 masked in a 12-token document whose prompt is 0..4
    d = C.distance_to_unmasked(np.zeros(5, int), np.arange(5, 10), np.full(5, 12))
    assert d.tolist() == [1, 2, 3, 2, 1]
    d = C.distance_to_unmasked(np.zeros(3, int), np.array([0, 1, 2]), np.full(3, 3))
    assert d.tolist() == [3, 3, 3]          # fully masked: the document length


def test_pc_sampler_weight():
    w = C.pc_sampler_weight(np.array([10, 11, 14]), 10)
    assert np.allclose(w, np.exp(-0.25 * np.array([0, 1, 4])))
    lf = np.log(np.array([0.5, 1e-9]))
    w = C.pc_sampler_weight(np.array([0, 0]), 0, log_freq=lf, top1=np.array([0, 1]))
    assert np.allclose(w, [-np.log(0.5), 10.0])           # capped at alpha


@pytest.mark.parametrize("s,cls", [(",", "punct"), ("\n", "punct"), (" ", "punct"),
                                   (" the", "stop"), ("The", "stop"), (" dog", "content"),
                                   ("ing", "content"), ("2024", "content"), ("--", "punct")])
def test_classify_string(s, cls):
    assert C.classify_string(s) == cls


def test_token_class_table_uses_the_decoded_strings():
    class Tok:
        vocab = [",", " the", " cat", "\n"]

        def __len__(self):
            return len(self.vocab)

        def batch_decode(self, ids):
            return [self.vocab[i[0]] for i in ids]

    t = C.token_class_table(Tok())
    assert [C.TOKEN_CLASSES[i] for i in t] == ["punct", "stop", "content", "punct"]


def _frames(scenario, n_docs=80, **kw):
    df = simulate_design(scenario, n_docs=n_docs, ks=(4,), seed=2, **kw)
    sub = df[df.mask_kind == "decoder"].reset_index(drop=True)
    pos, pools = position_and_pool_frames(sub, pool_sizes=POOL_SIZES, seed=0, retain=1,
                                          min_state_size=max(POOL_SIZES))
    docs = np.unique(pos.doc_id)
    fit, rep = docs[docs % 2 == 0], docs[docs % 2 == 1]
    codes = rep
    counts = doc_boot_counts(codes.size, 200, seed=0)
    return sub, pos, pools, np.isin(pos.doc_id, fit), np.isin(pos.doc_id, rep), codes, counts


def test_an_uninformative_control_changes_nothing():
    sub, pos, pools, fm, rm, codes, counts = _frames("noisy")
    X = np.random.default_rng(0).normal(size=(len(pos), 2))
    ce = C.covariate_control(pools, pos, X, fm, rm, codes, counts)
    d0, _ = ce.delta.contrast(HI, LO)
    d1, _ = ce.delta_adj.contrast(HI, LO)
    f0, _ = ce.field0.contrast(HI, LO)
    f1, _ = ce.field_x.contrast(HI, LO)
    assert abs(d1 - d0) < 0.003 and abs(f1 - f0) < 0.003


def test_the_position_control_absorbs_positional_miscalibration():
    sub, pos, pools, fm, rm, codes, counts = _frames("position", n_docs=120, shift=1.0)
    X, names = C.position_covariates(sub)
    assert names == list(C.POSITION_COVARIATES)
    ce = C.covariate_control(pools, pos, X, fm, rm, codes, counts)
    f0, _ = ce.field0.contrast(HI, LO)
    f1, _ = ce.field_x.contrast(HI, LO)
    d0, _ = ce.delta.contrast(HI, LO)
    d1, d1_draws = ce.delta_adj.contrast(HI, LO)
    # E2: the field term was positional, and x in both calibrators absorbs it.
    assert abs(f1) < abs(f0) / 3
    # E1 keeps the own-confidence part (a uniform shift steepens Delta(m) through
    # selection on ell too, T0 row two), so only the control term must be nonzero.
    t, t_draws = ce.control_term.contrast(HI, LO)
    assert np.quantile(t_draws, 0.025) > 0 or np.quantile(t_draws, 0.975) < 0
    # The identity behind the row: adjusted = Delta - control term, draw by draw.
    _, d0_draws = ce.delta.contrast(HI, LO)
    assert np.isclose(d1, d0 - t) and np.allclose(d1_draws, d0_draws - t_draws)


def test_logodds_temperature_is_recovered_per_bin():
    rng = np.random.default_rng(0)
    rate = rng.uniform(0.05, 1.0, 40_000)
    true = rng.normal(0.5, 2.0, rate.size)
    agree = (rng.random(rate.size) < 1 / (1 + np.exp(-true))).astype(float)
    T0 = 1 + rate                       # reported = true x T0: overconfident at high rates
    T = C.fit_temperatures("logodds", rate, np.ones(rate.size), ell=true * T0, agree=agree)
    centres = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert np.all(np.abs(T[1:-1] - (1 + centres[1:-1])) < 0.1)
    conf, ell = C.apply_temperatures("logodds", T, rate, ell=true * T0)
    assert np.allclose(conf, 1 / (1 + np.exp(-ell)))


def test_softmax_temperature_is_exact_at_one_and_recovers_a_known_t():
    rng = np.random.default_rng(1)
    n, k = 20_000, 5
    z = rng.normal(0, 2, (n, k))
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=pp) for pp in p])            # calibrated at T = 1
    zs = z * 2.0                                                # the model is too sharp
    q = np.exp(zs) / np.exp(zs).sum(1, keepdims=True)
    order = np.argsort(-q, axis=1)
    topk = np.take_along_axis(q, order, 1)
    p_true = q[np.arange(n), y]
    conf, pt = C.tempered_softmax(topk, p_true, k, 1.0)
    assert np.allclose(conf, topk[:, 0]) and np.allclose(pt, p_true)
    T = C.fit_temperatures("softmax", np.full(n, 0.5), np.ones(n), topk_p=topk, p_true=p_true,
                           vocab=k)
    assert abs(T[4] - 2.0) < 0.1 and np.all(T[np.arange(9) != 4] == 1.0)


def test_ablation_row_and_standardised_contrast():
    d0, d1 = np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.0, 1.5])
    r = C.ablation_row("x", "E1_contrast", "uniform", (2.0, d0), (1.0, d1), 10)
    assert r["difference"]["value"] == -1.0
    sub, pos, pools, fm, rm, codes, counts = _frames("calibrated", n_docs=20)
    from c3.endpoints.e1 import winner_gap
    g = winner_gap(pools, pos, pos.conf, pos.agree, codes, counts, rm)
    p, d = C.standardised_contrast({"all": g}, {"all": 1.0})
    p0, d0 = g.contrast(HI, LO)
    assert p == p0 and np.array_equal(d, d0)


def _run(*args, check=True):
    return subprocess.run([sys.executable, *map(str, args)], check=check, capture_output=True,
                          text=True, cwd=ROOT)


def test_cli_writes_one_table_with_every_control(tmp_path):
    _run("scripts/run_controls.py", "--traces", "sim-design:position,n_docs=40", "--split", "half",
         "--half", "report", "--n-boot", "50", "--temperature", "logodds", "--out", tmp_path)
    res = json.loads((tmp_path / "ablation.json").read_text())
    controls = {r["control"] for r in res["rows"]}
    assert {"position", "token_type", "token_type_standardised", "temperature_logodds"} <= controls
    for r in res["rows"]:
        assert np.isclose(r["difference"]["value"], r["with"]["value"] - r["without"]["value"])
    assert "\\begin{tabular}" in (tmp_path / "ablation.tex").read_text()
    macros = (tmp_path / "macros.tex").read_text()
    names = [ln.split("{\\")[1].split("}")[0] for ln in macros.splitlines() if "newcommand" in ln]
    assert names and not any(ch.isdigit() for n in names for ch in n)
    assert len(names) == len(set(names))


def test_cli_precision_on_identical_traces_differs_by_nothing(tmp_path):
    _run("scripts/run_controls.py", "--traces", "sim-design:noisy,n_docs=30",
         "--precision-trace", "sim-design:noisy,n_docs=30", "--controls", "precision",
         "--split", "half", "--half", "report", "--n-boot", "50", "--out", tmp_path)
    res = json.loads((tmp_path / "ablation.json").read_text())
    assert res["rows"] and all(r["control"] == "precision" for r in res["rows"])
    for r in res["rows"]:
        assert r["difference"]["value"] == 0.0 and r["difference"]["ci"] == [0.0, 0.0]
    pu = res["precision"]["paired_uniform"]
    assert pu["abs_ell_shift"]["value"] == 0.0 and pu["top1_agreement"]["value"] == 1.0


def test_cli_precision_on_a_merged_dataset_trace(tmp_path):
    """The precision step on a real (parquet-backed) trace, not a simulated one."""
    pytest.importorskip("torch")
    from c3.traces.collect import CollectConfig, Document, collect_shard
    from c3.traces.models import ToyDiffusionLM
    from c3.traces.shards import merge

    rng = np.random.default_rng(0)
    docs = [Document(i, rng.integers(0, 60, 80)) for i in range(12)]
    collect_shard(ToyDiffusionLM(), docs, CollectConfig(topk=5), tmp_path / "run", 0, 1,
                  part_docs=4, progress_every=0)
    man = merge(tmp_path / "run", tmp_path / "run.dataset.json")
    r = subprocess.run([sys.executable, "scripts/run_controls.py", "--traces", str(man),
                        "--precision-trace", str(man), "--controls", "precision",
                        "--split", "half", "--half", "report", "--n-boot", "20", "--knots", "3",
                        "--out", str(tmp_path / "out")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((tmp_path / "out" / "ablation.json").read_text())
    assert res["rows"] and all(x["difference"]["value"] == 0.0 for x in res["rows"])
    assert res["precision"]["paired_uniform"]["top1_agreement"]["value"] == 1.0
