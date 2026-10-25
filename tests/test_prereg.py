"""Phase C: the split manifest, gate G1, and the frozen pre-registration config."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import c3
from c3.data.fineweb import CorpusSpec, content_hash, document_split
from c3.gates import g1_arm, g1_decision
from c3.sim.positions import simulate_positions

ROOT = Path(__file__).resolve().parents[1]


def _run(*args, check=True):
    return subprocess.run([sys.executable, *map(str, args)], check=check, capture_output=True,
                          text=True, cwd=ROOT)


def _corpus(tmp_path, n=40):
    p = tmp_path / "docs.jsonl"
    with p.open("w") as fh:
        for i in range(n):
            text = f"document number {i} " * 30
            fh.write(json.dumps({"doc_id": i, "id": f"<urn:{i}>", "text": text,
                                 "hash": content_hash(text)}) + "\n")
    return p


def test_split_manifest_follows_document_split_and_drops_the_pilot(tmp_path):
    docs = _corpus(tmp_path)
    out = tmp_path / "split.json"
    _run("scripts/make_split_manifest.py", "--docs", docs, "--pilot-ids", "0-4",
         "--exclusion-manifest", "", "--out", out)
    m = json.loads(out.read_text())
    fit, report = document_split(40, seed=CorpusSpec().seed)
    assert set(m) == {str(i) for i in range(5, 40)}
    assert all(m[str(i)] == 0 for i in fit if i >= 5)
    assert all(m[str(i)] == 1 for i in report if i >= 5)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["excluded"] == {str(i): "pilot" for i in range(5)}
    # Frozen once written.
    r = _run("scripts/make_split_manifest.py", "--docs", docs, "--pilot-ids", "0-4",
             "--exclusion-manifest", "", "--out", out, check=False)
    assert r.returncode != 0 and "frozen" in (r.stderr + r.stdout)


def test_split_manifest_rejects_a_pilot_file_that_overlaps_the_corpus(tmp_path):
    docs = _corpus(tmp_path)
    pilot = tmp_path / "pilot.jsonl"
    pilot.write_text(docs.read_text().splitlines()[3] + "\n")
    r = _run("scripts/make_split_manifest.py", "--docs", docs, "--pilot-docs", pilot,
             "--exclusion-manifest", "", "--out", tmp_path / "s.json", check=False)
    assert r.returncode != 0 and "also in the main corpus" in r.stderr


def test_half_split_warns_without_a_manifest_and_refuses_unlisted_docs(tmp_path):
    common = ["scripts/analyze_traces.py", "--traces", "sim:uniform,n_docs=12", "--split", "half",
              "--n-boot", "20", "--skip-regression", "--no-figures", "--no-smooth-ece"]
    r = _run(*common, "--out", tmp_path / "a")
    assert "WARNING: --split half without --manifest" in r.stderr
    man = tmp_path / "m.json"
    man.write_text(json.dumps({str(i): i % 2 for i in range(2, 12)}))   # docs 0, 1 left out
    r = _run(*common, "--manifest", man, "--out", tmp_path / "b", check=False)
    assert r.returncode != 0 and "not in" in r.stderr


def test_g1_rule_takes_the_worst_arm_and_endpoint():
    def arm(nd, nf):
        return {"delta_contrast": {"n_required": nd}, "field_contrast": {"n_required": nf}}

    d = g1_decision({"uniform": arm(100, 50), "decoder": arm(300, 900)}, floor=0.0)
    assert d["decision"] == "full_plan" and d["binding"] == {"arm": "decoder",
                                                             "endpoint": "field_contrast"}
    assert g1_decision({"u": arm(1500, 10)}, 0.0)["decision"] == "fewer_8b_settings"
    assert g1_decision({"u": arm(10, 2500)}, 0.0)["decision"] == "shift_to_110m_and_c2"
    assert not g1_decision({"u": arm(10, 10)}, floor=0.02)["delta_min_above_2x_floor"]
    assert g1_decision({"u": arm(None, 10)}, 0.0)["decision"] == "undetermined"


def test_g1_arm_on_simulation():
    df = simulate_positions("context", n_docs=20, seed=2)
    r = g1_arm(df, n_boot=100)
    for key in ("delta_contrast", "field_contrast"):
        assert r[key]["n_docs"] == 20 and r[key]["sigma_d"] > 0
        assert r[key]["n_required"] == int(np.ceil(7.848879 * r[key]["sigma_d"] ** 2 / 0.02 ** 2))
    assert 0 < r["ece"]["value"] < 1


def test_frozen_config_reads_the_code(tmp_path):
    out = tmp_path / "frozen.json"
    _run("scripts/freeze_prereg.py", "--out", out)
    cfg = json.loads(out.read_text())
    assert cfg["constants"]["POOL_SIZES"] == list(c3.POOL_SIZES)
    assert cfg["constants"]["DELTA_MIN"] == c3.DELTA_MIN
    assert cfg["knots"] == {"primary": c3.DEFAULT_KNOTS, "sensitivity": [5, 13]}
    assert cfg["gates"]["G2"]["max_relative_change"] == 0.005
    assert "def declare" in cfg["declaration_rule"]["source"]
    assert cfg["provenance"]["git_commit"]


@pytest.mark.parametrize("path", ["prereg/analysis_commands.sh", "slurm/prereg_analysis.sbatch",
                                  "tests/prereg_stub_py.sh"])
def test_prereg_script_parses(path):
    subprocess.run(["bash", "-n", str(ROOT / path)], check=True)


def test_summarize_c1_applies_holm(tmp_path):
    for name, sc in (("Ctx", "context"), ("Cal", "calibrated")):
        _run("scripts/analyze_traces.py", "--traces", f"sim:{sc},n_docs=16", "--split", "kfold-doc",
             "--n-boot", "100", "--eq-boot", "10", "--no-figures", "--no-smooth-ece",
             "--out", tmp_path / name)
    _run("scripts/summarize_c1.py", "--results", f"Ctx={tmp_path / 'Ctx'}",
         f"Cal={tmp_path / 'Cal'}", "--out", tmp_path / "sum")
    s = json.loads((tmp_path / "sum" / "c1_summary.json").read_text())
    assert len(s["rows"]) == 2
    for r in s["rows"]:
        assert r["contrast"]["holm_family"] == "E1" and 0 <= r["contrast"]["p_holm"] <= 1
        assert r["contrast"]["p_holm"] >= r["contrast"]["p_boot"]
        assert r["gamma"]["gamma_2"] is None or "p_holm" in r["gamma"]["gamma_2"]
    assert (tmp_path / "sum" / "table.tex").exists()


def test_summarize_c1_macro_tags_are_letters_only():
    spec = importlib.util.spec_from_file_location("summarize_c1",
                                                  ROOT / "scripts" / "summarize_c1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.arm_tag("uniform") == "Uni"
    assert mod.arm_tag("decoder:topk_conf:k4") == "DecTopkConfKFour"   # the primary decoder arm
    assert mod.arm_tag("decoder:topk_conf:k2") != mod.arm_tag("decoder:topk_conf:k8")
    assert all(mod.arm_tag(a).isalpha() for a in ("uniform_matched", "decoder:topk_conf:k8"))


def test_frozen_config_records_the_new_choices(tmp_path):
    out = tmp_path / "frozen.json"
    _run("scripts/freeze_prereg.py", "--out", out)
    cfg = json.loads(out.read_text())
    assert cfg["contamination"]["threshold"] == c3.CONTAM_THRESHOLD == 32
    assert cfg["holm_families"]["primary_arms"] == ["uniform", "decoder:topk_conf:k4"]
    assert all(v for v in cfg["split"]["manifests"].values())          # every manifest hashed
    assert cfg["floors"]["llada"]["floor_delta"] == 0.0
    assert cfg["analysis_commands"]["sha256"] and cfg["deviations"]["sha256"]
    assert cfg["e3"]["rate_bin_centres"][0] == 0.1 and cfg["t8"]["position"]["covariates"]


def test_the_frozen_commands_refuse_to_read_the_report_half_by_default():
    r = subprocess.run(["bash", str(ROOT / "prereg/analysis_commands.sh")], capture_output=True,
                       text=True, cwd=ROOT, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode != 0 and "refusing" in r.stderr


def test_split_manifest_for_a_corpus_whose_ids_are_not_positions(tmp_path):
    p = tmp_path / "ts.jsonl"
    with p.open("w") as fh:
        for i in range(30):
            text = f"later document {i} " * 30
            fh.write(json.dumps({"doc_id": 200000 + i, "id": f"<ts:{i}>", "text": text,
                                 "hash": content_hash(text)}) + "\n")
    out = tmp_path / "s.json"
    r = _run("scripts/make_split_manifest.py", "--docs", p, "--pilot-docs", _corpus(tmp_path),
             "--exclusion-manifest", "", "--out", out, check=False)
    assert r.returncode != 0 and "--any-ids" in r.stderr
    _run("scripts/make_split_manifest.py", "--docs", p, "--pilot-docs", _corpus(tmp_path),
         "--exclusion-manifest", "", "--any-ids", "--out", out)
    m = json.loads(out.read_text())
    fit, report = document_split(30, seed=CorpusSpec().seed)
    assert set(m) == {str(200000 + i) for i in range(30)}
    assert all(m[str(200000 + i)] == 0 for i in fit) and all(m[str(200000 + i)] == 1 for i in report)
