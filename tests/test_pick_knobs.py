"""T12 knob calibration: scripts/pick_knobs.py.

Fit-half calibration runs (run_t12 output directories) go in; one grid per task comes out,
with every adaptive decoder's knob interpolated to 2 / 4 / 8 tokens per step.  The speed of a
configuration is the aggregate 256*N / sum(n_forward), never the mean of per-problem ratios.
"""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pick_knobs as pk  # noqa: E402

CAL = ROOT / "prereg" / "t12_cal.json"


def rec(doc_id, nfe, config="c", gen=256):
    return {"config": config, "doc_id": doc_id, "n_forward": nfe,
            "tokens_per_step": gen / nfe, "seconds": 0.5 * nfe, "error": None}


# ---- the speed metric ----------------------------------------------------------------

def test_speed_is_aggregate_not_mean_of_ratios():
    # two problems: 32 forward passes (8 tok/step) and 128 (2 tok/step)
    rs = [rec(0, 32), rec(1, 128)]
    mean_of_ratios = sum(r["tokens_per_step"] for r in rs) / len(rs)
    assert mean_of_ratios == 5.0
    assert pk.aggregate_speed(rs, 256) == pytest.approx(256 * 2 / 160)  # 3.2
    assert pk.aggregate_speed(rs, 256) != pytest.approx(mean_of_ratios)


def test_speed_of_empty_set_is_an_error():
    with pytest.raises(ValueError):
        pk.aggregate_speed([], 256)


# ---- interpolation to a target ---------------------------------------------------------

def tau_points(*pairs):
    """(tokens/step, config) pairs of a threshold path, slowest first."""
    return [(s, {"policy": "threshold", "tau": t}) for s, t in pairs]


def test_interpolates_log_speed_and_logit_tau():
    pts = tau_points((2.0, 0.9), (8.0, 0.5))
    knobs, info = pk.pick_for_target("threshold", pts, 4.0)
    # 4 is the geometric midpoint of 2 and 8, logit(0.9)/2 + logit(0.5)/2 = 1.0986 -> 0.75
    assert knobs == {"tau": pytest.approx(0.75, abs=1e-3)}
    assert info["unmatched_at_target"] is False


def test_pick_at_a_calibrated_point_returns_that_point():
    pts = tau_points((2.0, 0.9), (4.0, 0.8), (8.0, 0.5))
    knobs, _ = pk.pick_for_target("threshold", pts, 4.0)
    assert knobs == {"tau": pytest.approx(0.8, abs=1e-3)}


def test_picks_are_monotone_in_the_target_even_if_speeds_are_noisy():
    # tau 0.6 measured slower than tau 0.8 (noise): the order along the knob axis rules
    pts = tau_points((2.0, 0.99), (3.0, 0.9), (2.8, 0.8), (6.0, 0.6), (12.0, 0.4))
    taus = [pk.pick_for_target("threshold", pts, t)[0]["tau"] for t in (2.0, 3.0, 4.0, 6.0, 8.0)]
    assert taus == sorted(taus, reverse=True)
    assert all(0.4 <= t <= 0.99 for t in taus)


@pytest.mark.parametrize("policy,key,lo,hi", [
    ("entropy_bounded", "gamma", 0.01, 1.0),
    ("c3a", "alpha", 0.02, 0.6),
])
def test_other_knobs_increase_with_speed(policy, key, lo, hi):
    pts = [(2.0, {"policy": policy, key: lo}), (8.0, {"policy": policy, key: hi})]
    picks = [pk.pick_for_target(policy, pts, t)[0][key] for t in (2.0, 3.0, 4.0, 6.0, 8.0)]
    assert picks == sorted(picks)
    assert picks[0] == pytest.approx(lo, rel=0.05) and picks[-1] == pytest.approx(hi, rel=0.05)
    assert picks[2] == pytest.approx(math.sqrt(lo * hi), rel=0.05) if policy == "entropy_bounded" else True


def test_klass_interpolates_tau_and_eps_along_the_path():
    pts = [(2.0, {"policy": "klass", "tau": 0.9, "eps_kl": 0.001, "history": 2, "fallback": 1}),
           (8.0, {"policy": "klass", "tau": 0.5, "eps_kl": 0.1, "history": 2, "fallback": 1})]
    knobs, _ = pk.pick_for_target("klass", pts, 4.0)
    assert knobs["tau"] == pytest.approx(0.75, abs=1e-3)
    assert knobs["eps_kl"] == pytest.approx(0.01, rel=0.05)  # geometric midpoint
    assert knobs["history"] == 2 and knobs["fallback"] == 1


def test_target_far_outside_the_calibrated_range_is_refused():
    pts = tau_points((2.0, 0.9), (4.0, 0.8))
    with pytest.raises(pk.OutOfRange):
        pk.pick_for_target("threshold", pts, 8.0)       # fastest is 4: half of the target
    with pytest.raises(pk.OutOfRange):
        pk.pick_for_target("threshold", pts, 1.0)


def test_target_within_ten_percent_of_an_end_uses_that_end():
    pts = tau_points((2.0, 0.9), (7.5, 0.5))              # 7.5 is within 10% of 8
    knobs, info = pk.pick_for_target("threshold", pts, 8.0)
    assert knobs == {"tau": 0.5}
    assert info["achieved_tokens_per_step"] == 7.5
    assert info["unmatched_at_target"] is False


# ---- KLASS at speed 8 ----------------------------------------------------------------

def test_klass_that_cannot_reach_the_top_target_is_marked_unmatched():
    pts = [(2.0, {"policy": "klass", "tau": 0.9, "eps_kl": 0.001, "history": 2, "fallback": 1}),
           (5.0, {"policy": "klass", "tau": 0.4, "eps_kl": 0.1, "history": 2, "fallback": 1})]
    knobs, info = pk.pick_for_target("klass", pts, 8.0)
    assert knobs == {"tau": 0.4, "eps_kl": 0.1, "history": 2, "fallback": 1}  # the fastest
    assert info["unmatched_at_target"] is True
    assert info["achieved_tokens_per_step"] == 5.0


# ---- synthetic calibration runs -------------------------------------------------------

def cal_configs():
    return json.loads(CAL.read_text())["configs"]


def path_nfes(klass_top):
    """Forward passes per problem along each adaptive path of the 25-config calibration grid:
    2, 2.67, 4, 5.33, 8, 10.67 tokens/step for six knobs (klass_top = the fastest klass NFE)."""
    nfes = [128, 96, 64, 48, 32, 24]
    return {"threshold": nfes, "entropy_bounded": nfes, "c3a": nfes,
            "klass": nfes[:-1] + [klass_top]}


def make_run(root, name, task, configs, scale=1.0, klass_top=24, half="fit",
             manifest_halves=None, n_problems=3, manifest_name="manifest.json"):
    run = root / name
    run.mkdir()
    manifest = {str(d): (manifest_halves or {}).get(d, 0) for d in range(n_problems)}
    (root / manifest_name).write_text(json.dumps(manifest))
    meta = {"task": task, "half": half, "gen_length": 256, "block_length": 32,
            "precision": "fp16", "prompt": {"format": "chat", "fewshot": 0},
            "manifest": str(root / manifest_name)}
    (run / "t12.shard0of1.meta.json").write_text(json.dumps(meta))
    seen = {}
    lines = []
    for cfg in configs:
        i = seen.get(cfg["policy"], 0)
        seen[cfg["policy"]] = i + 1
        nfe = 64 if cfg["policy"] == "topk_conf" else path_nfes(klass_top)[cfg["policy"]][i]
        key = pk.config_key(cfg)
        for d in range(n_problems):
            lines.append(json.dumps({**cfg, **rec(d, round(nfe * scale), key)}))
    (run / "t12.shard0of1.jsonl").write_text("\n".join(lines) + "\n")
    return run


def test_committed_calibration_grid_is_fit_half_and_has_the_adaptive_paths():
    d = json.loads(CAL.read_text())
    assert d["half"] == "fit"
    by = {}
    for c in d["configs"]:
        by.setdefault(c["policy"], []).append(c)
    assert len(d["configs"]) == 25
    assert {p: len(v) for p, v in by.items()} == {
        "topk_conf": 1, "threshold": 6, "entropy_bounded": 6, "klass": 6, "c3a": 6}
    assert len({pk.config_key(c) for c in d["configs"]}) == 25


def test_per_task_grids_come_out_one_per_task(tmp_path):
    cfgs = cal_configs()
    g = make_run(tmp_path, "cal-gsm8k", "gsm8k", cfgs)
    m = make_run(tmp_path, "cal-mbpp", "mbpp", cfgs, scale=1.25, manifest_name="m2.json")
    grids = pk.pick_runs([g, m])
    assert sorted(grids) == ["gsm8k", "mbpp"]
    for task, grid in grids.items():
        assert grid["task"] == task
        assert grid["status"] == "calibrated"
        cs = grid["configs"]
        assert len(cs) == 21                                  # 7 decoders x 3 speeds
        assert [c["k"] for c in cs if c["policy"] == "topk_conf"] == [2, 4, 8]
        assert [c["k"] for c in cs if c["policy"] == "margin"] == [2, 4, 8]
        # the policy kwargs are exactly what run_t12.build_policy hands to the constructors
        assert all(set(c) <= {"policy", "k", "tau", "gamma", "eps_kl", "history", "fallback",
                              "alpha", "lam"} for c in cs)
        assert len({pk.config_key(c) for c in cs}) == 21
        assert set(grid["speed_of_config"].values()) == {1, 2, 3}
    # different speeds in the two tasks give different knobs for the same decoder
    thr = {t: [c["tau"] for c in grids[t]["configs"] if c["policy"] == "threshold"]
           for t in grids}
    assert thr["gsm8k"] != thr["mbpp"]
    for t in thr:
        assert thr[t] == sorted(thr[t], reverse=True)


def test_picks_are_the_calibrated_knobs_when_the_targets_are_hit_exactly(tmp_path):
    run = make_run(tmp_path, "cal-gsm8k", "gsm8k", cal_configs())
    grid = pk.pick_runs([run])["gsm8k"]
    # NFE 128 / 64 / 32 are exactly 2 / 4 / 8 tok/step: knobs 1, 3 and 5 of each path
    cal = cal_configs()
    for pol, key in (("threshold", "tau"), ("entropy_bounded", "gamma"),
                     ("klass", "tau"), ("c3a", "alpha")):
        path = [c[key] for c in cal if c["policy"] == pol]
        got = [c[key] for c in grid["configs"] if c["policy"] == pol]
        assert got == pytest.approx([path[0], path[2], path[4]], rel=0.02), pol
    assert all(not p["unmatched_at_target"] for p in grid["picks"])
    assert len(grid["picks"]) == 12                           # 4 adaptive decoders x 3 speeds


def test_klass_that_tops_out_at_five_is_unmatched_at_8(tmp_path):
    # make every klass setting slow: scale NFE so the fastest is 5 tok/step
    cfgs = cal_configs()
    run = make_run(tmp_path, "cal-mbpp", "mbpp", cfgs, klass_top=24)
    # rewrite the klass records to NFEs 256, 200, 160, 128, 90, 51 (1.0 .. 5.02 tok/step)
    jl = run / "t12.shard0of1.jsonl"
    new = {pk.config_key(c): n for c, n in
           zip([c for c in cfgs if c["policy"] == "klass"], [256, 200, 160, 128, 90, 51])}
    out = []
    for line in jl.read_text().splitlines():
        r = json.loads(line)
        if r["policy"] == "klass":
            r["n_forward"] = new[r["config"]]
            r["tokens_per_step"] = 256 / r["n_forward"]
        out.append(json.dumps(r))
    jl.write_text("\n".join(out) + "\n")
    grid = pk.pick_runs([run])["mbpp"]
    slot3 = [p for p in grid["picks"] if p["policy"] == "klass" and p["speed"] == 3][0]
    assert slot3["unmatched_at_target"] is True
    assert slot3["achieved_tokens_per_step"] == pytest.approx(256 / 51, rel=1e-6)
    last = [c for c in cfgs if c["policy"] == "klass"][-1]
    klass = [c for c in grid["configs"] if c["policy"] == "klass"]
    assert klass[2]["tau"] == last["tau"] and klass[2]["eps_kl"] == last["eps_kl"]
    # the unmatched flag lives next to the configs, never inside them (it would reach the
    # policy constructor and change the config key)
    assert "unmatched_at_target" not in klass[2]
    others = [p for p in grid["picks"] if p["policy"] != "klass"]
    assert all(not p["unmatched_at_target"] for p in others)


# ---- only the fit half -----------------------------------------------------------------

def test_refuses_a_run_whose_meta_half_is_not_fit(tmp_path):
    run = make_run(tmp_path, "cal-gsm8k", "gsm8k", cal_configs(), half="report")
    with pytest.raises(SystemExit, match="fit"):
        pk.pick_runs([run])


def test_refuses_a_run_with_a_report_half_record(tmp_path):
    run = make_run(tmp_path, "cal-gsm8k", "gsm8k", cal_configs(), manifest_halves={2: 1})
    with pytest.raises(SystemExit, match="report half"):
        pk.pick_runs([run])


def test_refuses_a_run_without_a_manifest_to_check_against(tmp_path):
    run = make_run(tmp_path, "cal-gsm8k", "gsm8k", cal_configs())
    meta_path = run / "t12.shard0of1.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["manifest"] = None
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="manifest"):
        pk.pick_runs([run])


def test_refuses_two_runs_of_the_same_task(tmp_path):
    a = make_run(tmp_path, "a", "gsm8k", cal_configs())
    b = make_run(tmp_path, "b", "gsm8k", cal_configs(), manifest_name="m2.json")
    with pytest.raises(SystemExit, match="gsm8k"):
        pk.pick_runs([a, b])


# ---- command line ------------------------------------------------------------------------

def test_cli_writes_one_file_per_task_and_none_by_default(tmp_path, capsys):
    cfgs = cal_configs()
    g = make_run(tmp_path, "cal-gsm8k", "gsm8k", cfgs)
    m = make_run(tmp_path, "cal-mbpp", "mbpp", cfgs, manifest_name="m2.json")
    assert pk.main([str(g), str(m)]) == 0
    assert "tok/step" in capsys.readouterr().out
    assert not list(tmp_path.glob("t12_grid*"))
    out = tmp_path / "grids"
    assert pk.main([str(g), str(m), "--out-dir", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["t12_grid_gsm8k.json", "t12_grid_mbpp.json"]
    d = json.loads((out / "t12_grid_gsm8k.json").read_text())
    assert len(d["configs"]) == 21 and d["task"] == "gsm8k"


# ---- the grid is what run_t12 consumes -----------------------------------------------------

def test_grid_loads_through_run_t12_and_keys_agree(tmp_path):
    run_t12 = pytest.importorskip("run_t12")
    cfgs = cal_configs()
    run = make_run(tmp_path, "cal-gsm8k", "gsm8k", cfgs)
    assert all(pk.config_key(c) == run_t12.config_key(c) for c in cfgs)
    out = tmp_path / "grids"
    assert pk.main([str(run), "--out-dir", str(out)]) == 0
    path = out / "t12_grid_gsm8k.json"
    loaded = run_t12.load_configs(path)
    assert len(loaded) == 21
    grid = json.loads(path.read_text())
    assert set(grid["speed_of_config"]) == {run_t12.config_key(c) for c in loaded}
    # the calibration grid itself loads too, and is what --half fit runs
    assert len(run_t12.load_configs(CAL)) == 25
