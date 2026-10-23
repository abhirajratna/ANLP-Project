"""T10 for the paper: scripts/macros_t10.py on the committed T10 outputs."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
C2 = ROOT / "artifacts" / "c2"
NEWCOMMAND = re.compile(r"^\\newcommand\{\\([^}]*)\}\{(.*)\}$", re.M)
K = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
RHO = {0.0: "Zero", 0.25: "Quarter", 0.5: "Half", 0.75: "ThreeQuarters", 1.0: "One"}
RATE = {0.1: "PtOne", 0.2: "PtTwo", 0.3: "PtThree", 0.4: "PtFour", 0.5: "PtFive", 0.6: "PtSix",
        0.7: "PtSeven", 0.8: "PtEight", 0.9: "PtNine", 1.0: "One"}
RUNS = {"": "t10_local-110m_fp32_train110m-v3.json", "Llada": "t10_llada-8b-base_fp16_main.json"}


def run(out: Path) -> list:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "macros_t10.py"), "--out", str(out)],
                   check=True, capture_output=True, text=True, cwd=ROOT)
    return NEWCOMMAND.findall(out.read_text())


def value(body: str) -> float:
    """The number a macro prints: \\ensuremath{-0.003} -> -0.003, 1{,}000 -> 1000."""
    return float(re.sub(r"[^0-9.+-]", "", body.replace("{,}", "")))


def test_every_macro_is_its_json_value(tmp_path):
    pairs = run(tmp_path / "macros_t10.tex")
    names = [n for n, _ in pairs]
    assert names and all(n.isascii() and n.isalpha() and n.startswith("CTwoTTen") for n in names)
    assert len(names) == len(set(names))
    mac = dict(pairs)
    for word, file in RUNS.items():
        rt = json.loads((C2 / file).read_text())["real_text"]
        assert mac[f"CTwoTTen{word}NDocs"] == str(rt["n_docs"])
        for key, tag in (("per_k_rates_below_1", ""), ("per_k", "AllRates")):
            for e in rt[key]:
                for q, short, tol in (("disagreement", "Dis", 5e-5), ("delta", "Delta", 5e-4),
                                      ("ce", "CE", 5e-3)):
                    name = f"CTwoTTen{word}{short}{tag}K{K[e['k']]}"
                    assert value(mac[name]) == pytest.approx(e[q]["mean"], abs=tol), name
                    for side, b in (("Lo", 0), ("Hi", 1)):
                        assert value(mac[name + side]) == pytest.approx(e[q]["ci95"][b], abs=tol)
        for e in rt["per_k_rate"]:
            if e["k"] > 1:
                name = f"CTwoTTen{word}DisK{K[e['k']]}Rate{RATE[e['rate']]}"
                assert value(mac[name]) == pytest.approx(e["disagreement"]["mean"], abs=5e-5)
        low = max(e["disagreement"]["mean"] for e in rt["per_k_rate"]
                  if e["k"] > 1 and e["rate"] <= 0.7)
        assert value(mac[f"CTwoTTen{word}DisMaxUpToRatePtSeven"]) == pytest.approx(low, abs=5e-5)
    # A negative number prints a minus, not a hyphen; a mean that can take either sign is signed.
    assert mac["CTwoTTenDeltaKOneLo"].startswith("\\ensuremath{-")
    assert mac["CTwoTTenDeltaKEight"].startswith("\\ensuremath{+")

    probe = json.loads((C2 / RUNS[""]).read_text())["probe_calibration"]
    for tag, word in (("ft110m-v3-visible", ""), ("ft110m-v3", "Alt")):
        sp = probe[tag]["spearman"]
        assert mac[f"CTwoTTen{word}ProbeSpearman"] == f"{sp['rho_s']:.2f}"
        assert (value(mac[f"CTwoTTen{word}ProbeSpearmanLo"]),
                value(mac[f"CTwoTTen{word}ProbeSpearmanHi"])) == pytest.approx(sp["ci95"], abs=5e-3)
        for c in probe[tag]["cells"]:
            name = f"CTwoTTen{word}ProbeDisatRho{RHO[c['rho']]}K{K[c['k']]}"
            assert value(mac[name]) == pytest.approx(c["disagreement"]["mean"], abs=5e-4)
    assert mac["CTwoTTenProbeNCells"] == str(probe["ft110m-v3-visible"]["spearman"]["n_cells"])


def test_committed_macros_are_current(tmp_path):
    assert NEWCOMMAND.findall((C2 / "macros_t10.tex").read_text()) == run(tmp_path / "m.tex")
