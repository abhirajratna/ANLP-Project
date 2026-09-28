"""T0: exact simulation truth, the scenario runner, and the one-command script."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from c3 import POOL_SIZES
from c3.sim.positions import simulate_positions
from c3.sim.t0 import EXPECTED, T0Config, field_growth, matches, run_scenario, sign_class
from c3.sim.truth import pi_given_state, pi_marginal, true_calibrators

ROOT = Path(__file__).resolve().parents[1]


def test_calibrated_truth_is_the_sigmoid():
    ell = np.linspace(-6, 6, 50)
    sig = 1 / (1 + np.exp(-ell))
    assert np.allclose(pi_given_state(ell, 0.5, 0.0), sig)
    assert np.allclose(pi_marginal(ell, "calibrated"), sig, atol=1e-6)


@pytest.mark.parametrize("scenario", ["uniform", "context"])
def test_truth_matches_the_simulated_probabilities(scenario):
    df = simulate_positions(scenario, n_docs=150, seed=3)
    own, ctx = true_calibrators(df, scenario)
    for pi in (own, ctx):
        b = pd.qcut(pi, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"b": b, "pi": pi, "pt": df.p_true}).groupby("b").mean()
        assert (g.pt - g.pi).abs().max() < 0.02          # E[p* | pi] = pi
    if scenario == "uniform":
        assert np.allclose(own, ctx, atol=1e-6)         # identical states: no field


def test_sign_classes_and_pattern_matching():
    assert sign_class((0.1, 0.2)) == ">0" and sign_class((-0.2, -0.1)) == "<0"
    assert sign_class((-0.1, 0.1)) == "0"
    assert matches("?", "<0") and matches("!=0", ">0") and not matches("!=0", "0")
    assert set(EXPECTED) == {"calibrated", "uniform", "context", "misreading"}


def test_true_field_term_is_zero_at_m1_and_positive_early():
    cfg = T0Config(check_docs=120, check_boot=50, noise_levels=(1.0,))
    g = field_growth(cfg)["1.0"]
    assert g["m"] == list(POOL_SIZES)
    assert abs(g["field"][0]) < 0.01                    # law of total expectation at m=1
    assert g["field"][2] > g["field"][0]                # grows over the first steps


def test_run_scenario_row_is_complete():
    cfg = T0Config(n_docs=30, n_boot=50, eq_boot=5, noise_levels=())
    r = run_scenario("context", cfg)
    row = r["row"]
    assert {"contrast", "grows", "field_m32", "field_m32_true", "beta_nofe", "beta_fe",
            "expected", "observed", "matches"} <= set(row)
    assert row["expected"] == EXPECTED["context"]
    assert r["n_docs_fit"] + r["n_docs_report"] == 30
    s = r["series"]
    # Prop. 2 holds term by term on the reported winners.
    total = np.add(s["delta_own"]["point"], s["field"]["point"]) + np.array(s["residual"]["point"])
    assert np.allclose(total, s["delta"]["point"])


def test_script_writes_every_output(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "t0_scenarios.py"), "--n-docs", "24",
                    "--n-boot", "40", "--eq-boot", "4", "--noise-levels", "1.0",
                    "--check-docs", "40", "--check-boot", "20", "--out", str(tmp_path)],
                   check=True, capture_output=True, text=True, cwd=ROOT)
    for name in ("table.json", "table.tex", "figure1.pdf", "figure1.png", "macros.tex",
                 "field_growth.png"):
        assert (tmp_path / name).exists(), name
    res = json.loads((tmp_path / "table.json").read_text())
    assert set(res["scenarios"]) == set(EXPECTED) and res["provenance"]["git_commit"]
    macros = (tmp_path / "macros.tex").read_text()
    names = [ln.split("{\\")[1].split("}")[0] for ln in macros.splitlines()
             if ln.startswith("\\newcommand")]
    assert "TZeroGapThirtyTwo" in names and not any(ch.isdigit() for n in names for ch in n)
