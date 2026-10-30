"""T11: fitting and freezing the C3a calibrator on the fit half of a trace."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from c3.calibrators import DECODE_FEATURES, ContextLogistic
from c3.sim.trajectories import simulate_design

ROOT = Path(__file__).resolve().parents[1]


def _trace(tmp_path):
    df = simulate_design("noisy", n_docs=60, doc_len=64, ks=(2, 4), matched=False, seed=5)
    df["policy"] = np.where(df["mask_kind"] == "decoder", "topk_conf", "")
    path = tmp_path / "trace.parquet"
    df.to_parquet(path, index=False)
    docs = sorted(df["doc_id"].unique())
    man = tmp_path / "split_manifest_task.json"
    man.write_text(json.dumps({str(d): int(d % 2) for d in docs}))
    return df, path, man


def test_fit_reads_only_the_fit_half_and_freezes_a_loadable_calibrator(tmp_path):
    df, path, man = _trace(tmp_path)
    out = tmp_path / "c3" / "toy_fp32_c3a-task.json"
    p = subprocess.run([sys.executable, "scripts/fit_c3a_calibrator.py", "--traces", str(path),
                        "--manifest", str(man), "--arms", "decoder:topk_conf:k2",
                        "decoder:topk_conf:k4", "--n-boot", "100", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    rec = json.loads(out.read_text())
    fit_docs = [d for d in df["doc_id"].unique() if d % 2 == 0]
    dec = df[(df["mask_kind"] == "decoder") & df["doc_id"].isin(fit_docs)]
    assert rec["n_fit_docs"] == len(fit_docs) and rec["n_fit_positions"] == len(dec)
    assert rec["features_requested"] == list(DECODE_FEATURES)
    assert rec["split"]["fit_half"] == 0 and "git_commit" in rec["provenance"]
    chk = rec["dev_check"]
    assert chk["n_dev_fit_docs"] + chk["n_dev_eval_docs"] == len(fit_docs)
    assert rec["n_fit_states"] > 0 and 0.0 <= rec["monotone_share"] <= 1.0
    cal = ContextLogistic.load(out)
    assert set(cal.feature_ranges) == set(cal.used) == set(rec["feature_quantiles"])
    for name, r in cal.feature_ranges.items():
        q = rec["feature_quantiles"][name]
        assert r["min"] <= q["p05"] <= q["p50"] <= q["p95"] <= r["max"]
    st = {"median_ell": 0.5, "mask_rate": 0.4, "log_n_masked": np.log(20)}
    q = cal.predict_arrays(np.array([-2.0, 0.0, 3.0]), st)
    assert np.all((q > 0) & (q < 1)) and q[0] < q[2]


def test_fit_refuses_when_no_fit_half_rows_match(tmp_path):
    _, path, man = _trace(tmp_path)
    p = subprocess.run([sys.executable, "scripts/fit_c3a_calibrator.py", "--traces", str(path),
                        "--manifest", str(man), "--arms", "decoder:topk_conf:k8",
                        "--out", str(tmp_path / "x.json")], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0 and "no fit-half rows" in (p.stdout + p.stderr)
