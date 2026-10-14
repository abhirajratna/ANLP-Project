"""End to end on the CPU: traces -> calibrators -> toy infilling -> matched-speed analysis."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from c3.frame import hash_halves
from c3.sim.trajectories import simulate_design

ROOT = Path(__file__).resolve().parents[1]


def _run(*args, env=None, check=True):
    p = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    if check and p.returncode != 0:
        raise AssertionError(f"{args[0]} failed:\n{p.stdout}\n{p.stderr}")
    return p


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    d = tmp_path_factory.mktemp("setcal")
    df = simulate_design("calibrated", n_docs=120, doc_len=64, ks=(2, 4), matched=False, seed=3,
                         err_corr=0.9, mu=3.0)
    c = df["conf"].to_numpy()
    df["entropy"] = -(c * np.log(c) + (1 - c) * np.log1p(-c))
    trace = d / "sim.parquet"
    df.to_parquet(trace, index=False)
    docs = np.unique(df["doc_id"])
    man = d / "dev.manifest.json"
    man.write_text(json.dumps({str(int(x)): int(h) for x, h in zip(docs, hash_halves(docs, 1))}))
    out = d / "setcal.json"
    _run("scripts/analyze_setcal.py", "--traces", trace, "--manifest", man,
         "--arms", "decoder:topk_conf:k2", "decoder:topk_conf:k4", "--n-boot", 200,
         "--save-calibrators", d / "cal", "--out", out)
    return d, json.loads(out.read_text())


def test_setcal_script_scores_every_arm_and_saves_the_calibrators(calibrated):
    d, res = calibrated
    assert set(res["arms"]) == {"decoder:topk_conf:k2", "decoder:topk_conf:k4", "pooled"}
    pooled = res["arms"]["pooled"]
    assert set(pooled["compare"]) == {"set_vs_indep", "set_vs_raw", "indep_vs_raw"}
    assert pooled["compare"]["set_vs_indep"]["log_loss"]["gain"] > 0      # correlated errors
    assert [r["j"] for r in pooled["gap_by_j"]["set"]][:2] == ["1-1", "2-2"]
    assert (d / "cal" / "tok.json").exists() and (d / "cal" / "set.json").exists()


def test_toy_infill_end_to_end_resumes_and_is_deterministic(calibrated, tmp_path):
    d, _ = calibrated
    man = tmp_path / "toy.manifest.json"
    man.write_text(json.dumps({str(i): int(i % 4 == 0) for i in range(24)}))
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"configs": [
        {"policy": "topk_conf", "k": k} for k in (1, 2, 4, 8)] + [
        {"policy": "threshold", "tau": t} for t in (0.3, 0.6, 0.9)] + [
        {"policy": "c3set", "alpha": al} for al in (0.05, 0.3, 0.6, 0.9)] + [
        {"policy": "c3indep", "alpha": 0.3}, {"policy": "c3a", "alpha": 0.3}]}))
    common = ["scripts/eval_infill.py", "--model", "toy", "--device", "cpu", "--manifest", man,
              "--half", "fit", "--configs", grid, "--calibrator", d / "cal" / "tok.json",
              "--set-calibrator", d / "cal" / "set.json", "--max-tokens", 64, "--span", 16, 48]
    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    _run(*common, "--record-steps", "--out-dir", out1)
    again = _run(*common, "--record-steps", "--out-dir", out1)
    assert "documents in" not in again.stdout                     # everything already done
    _run(*common, "--out-dir", out2)
    lines = (out1 / "infill.shard0of1.jsonl").read_text().splitlines()
    assert len(lines) == 18 * 13                                    # 18 fit documents x 13 configs
    assert list((out1 / "steps").glob("*.parquet"))

    res_path = tmp_path / "analysis.json"
    _run("scripts/analyze_infill.py", "--runs", out1, "--compare-to", out2, "--speeds", 2, 4,
         "--n-boot", 200, "--baselines", "topk_conf", "threshold", "--out", res_path)
    res = json.loads(res_path.read_text())
    assert res["repeat_check"]["identical_share"] == 1.0
    assert set(res["primary"]) == {"2.0", "4.0"}
    tk = {c["k"]: c["tokens_per_step"] for c in res["curves"] if c["policy"] == "topk_conf"}
    assert tk == {1.0: 1.0, 2.0: 2.0, 4.0: 4.0, 8.0: 8.0}
    assert {c["policy"] for c in res["coverage"]} == {"c3set", "c3indep", "c3a"}

    # Refit on the on-policy states: dev re-split of the fit documents, as make_dev_subset.
    fit = np.array([i for i in range(24) if i % 4 != 0])
    dev = tmp_path / "toy_dev.json"
    dev.write_text(json.dumps({str(int(i)): int(h) for i, h in zip(fit, hash_halves(fit, 1))}))
    onp = tmp_path / "onpolicy.json"
    _run("scripts/analyze_setcal.py", "--onpolicy", out1 / "steps",
         "--manifest", dev, "--n-boot", 100, "--out", onp)
    assert "pooled" in json.loads(onp.read_text())["arms"]
    # The trace calibrators scored, not refitted, on the same free-running states.
    given = tmp_path / "onpolicy_tracecal.json"
    _run("scripts/analyze_setcal.py", "--onpolicy", out1 / "steps", "--manifest", dev,
         "--use-calibrators", d / "cal", "--n-boot", 100, "--out", given)
    g = json.loads(given.read_text())
    assert g["set_calibrator"] == json.loads((d / "cal" / "set.json").read_text())
    assert g["arms"]["pooled"]["n_rows"] == json.loads(onp.read_text())["arms"]["pooled"]["n_rows"]


def test_the_report_half_of_the_frozen_split_is_refused(tmp_path):
    man = tmp_path / "split_manifest.json"
    man.write_text(json.dumps({str(i): i % 2 for i in range(8)}))
    grid = tmp_path / "g.json"
    grid.write_text(json.dumps([{"policy": "topk_conf", "k": 4}]))
    args = ["scripts/eval_infill.py", "--model", "toy", "--device", "cpu", "--manifest", man,
            "--half", "report", "--configs", grid, "--max-tokens", 64, "--span", 16, 48,
            "--out-dir", tmp_path / "o"]
    p = _run(*args, check=False, env={"C3_REPORT_HALF_OK": ""})
    assert p.returncode != 0 and "refusing" in (p.stdout + p.stderr)
    _run(*args, env={"C3_REPORT_HALF_OK": "1"})
