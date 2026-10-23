"""T9 for the paper: scripts/figures_t9.py on the committed probe results."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "artifacts" / "probe"
NEWCOMMAND = re.compile(r"^\\newcommand\{\\([^}]*)\}\{(.*)\}$", re.M)
RHO = {0.0: "Zero", 0.25: "Quarter", 0.5: "Half", 0.75: "ThreeQuarters", 1.0: "One"}
K = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}


def run(out: Path, *extra: str) -> tuple[dict, list]:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "figures_t9.py"), "--out-dir", str(out),
                    *extra], check=True, capture_output=True, text=True, cwd=ROOT)
    return (json.loads((out / "main_variant.json").read_text()),
            NEWCOMMAND.findall((out / "macros_t9.tex").read_text()))


def test_rule_figures_and_macros_from_the_committed_results(tmp_path):
    pytest.importorskip("matplotlib")
    dec, pairs = run(tmp_path)
    # Neither marker effect reaches 0.5, so the lower validation NELBO decides.
    assert dec["choice"] == "ft110m-v3-visible" and dec["decided_by"].startswith("fallback")
    assert not any(c["reaches_bar"] for c in dec["candidates"].values())
    assert dec["provenance"]["git_commit"] and dec["sources"]
    for stem in ("c2_crossover_ft110m-v3-visible", "c2_crossover_ft110m-v3", "c2_untuned_vs_tuned"):
        for ext in ("pdf", "png"):
            assert (tmp_path / "figures" / f"{stem}.{ext}").stat().st_size > 10_000, stem

    names = [n for n, _ in pairs]
    assert names and all(n.isascii() and n.isalpha() for n in names)
    assert len(names) == len(set(names))
    mac = dict(pairs)
    assert mac["CTwoValNelbo"] == "0.1751" and mac["CTwoAltValNelbo"] == "0.1892"
    assert mac["CTwoMarkerEffect"] == "0.251" and mac["CTwoAltMarkerEffect"] == "0.014"
    # Every cell macro of the main variant is its grid value, under the documented name.
    grid = json.loads((PROBE / "results" / "grid_ft110m-v3-visible.json").read_text())
    for c in grid["cells"]:
        at = f"{'Conf' if c['rule'] == 'confidence' else 'Rand'}{{q}}atRho{RHO[c['rho']]}K{K[c['k']]}"
        for q in ("A", "B"):
            assert float(mac["CTwo" + at.format(q=q)]) == pytest.approx(
                c[f"{q}_per_field"]["mean"], abs=5e-4)
        assert float(mac["CTwo" + at.format(q="ShareA")]) == pytest.approx(
            100 * c["share_A"], abs=0.5)
    for key, x in grid["crossover"].items():
        rule, rho = key.split("/rho=")
        name = f"CTwo{'Conf' if rule == 'confidence' else 'Rand'}CrossKatRho{RHO[float(rho)]}"
        assert mac[name] == str(x["first_k_A_exceeds_B"] or "--")


def test_committed_outputs_are_current(tmp_path):
    dec, pairs = run(tmp_path, "--no-figures")
    assert not (tmp_path / "figures").exists()
    committed = json.loads((PROBE / "main_variant.json").read_text())
    for key in ("choice", "decided_by", "candidates", "sources"):
        assert committed[key] == dec[key], key
    assert NEWCOMMAND.findall((PROBE / "macros_t9.tex").read_text()) == pairs
