"""The contamination screen and --exclude-docs."""
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from c3.data.contamination import excluded_doc_ids, flag, longest_run

ROOT = Path(__file__).resolve().parents[1]


def test_longest_run_is_position_aligned():
    assert longest_run([1, 2, 3, 4, 5], [1, 2, 0, 4, 5]) == 2
    assert longest_run([1, 2, 3], [9, 1, 2]) == 0          # shifted copies do not count
    assert longest_run([7] * 50, [7] * 50) == 50


def test_flagging_and_exclusion(tmp_path):
    res = flag({"0": {"run_length": 40}, "1": {"run_length": 3}, "2": {"run_length": 32}}, 32)
    assert [d for d, r in res.items() if r["flagged"]] == ["0", "2"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"docs": res, "threshold": 32}))
    assert excluded_doc_ids(p) == {0, 2}
    (tmp_path / "l.json").write_text("[4, 5]")
    assert excluded_doc_ids(tmp_path / "l.json") == {4, 5}
    (tmp_path / "bad.json").write_text(json.dumps({"docs": {"0": {"run_length": 1}}}))
    with pytest.raises(ValueError, match="threshold"):
        excluded_doc_ids(tmp_path / "bad.json")


def test_screen_separates_a_memoriser_from_a_model_that_knows_nothing():
    pytest.importorskip("torch")
    from c3.data.contamination import screen_document
    from c3.traces.models import ToyDiffusionLM

    @dataclass
    class Memoriser(ToyDiffusionLM):
        def __post_init__(self):
            super().__post_init__()
            self.token_skill[:] = 12.0          # always confident, always right

    ids = np.random.default_rng(0).integers(0, 60, 120)
    m = Memoriser()
    m.set_reference(ids[:100])
    r = screen_document(m, ids)
    assert r["run_length"] == 50 and r["prefix_match"] == 50 and r["forward_passes"] == 50

    blind = Memoriser()
    blind.set_reference((ids[:100] + 1) % 60)   # "knows" a different continuation
    assert screen_document(blind, ids)["run_length"] == 0


def test_analyze_traces_drops_flagged_documents(tmp_path):
    flags = tmp_path / "c.json"
    flags.write_text(json.dumps({"docs": {str(d): {"run_length": 50, "flagged": d < 3}
                                          for d in range(12)}}))
    out = tmp_path / "res"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_traces.py"), "--traces",
                    "sim:uniform,n_docs=12", "--split", "kfold-doc", "--n-boot", "20",
                    "--skip-regression", "--no-figures", "--no-smooth-ece",
                    "--exclude-docs", str(flags), "--out", str(out)],
                   check=True, capture_output=True, text=True, cwd=ROOT)
    res = json.loads((out / "results.json").read_text())
    assert res["input"]["excluded_docs"]["doc_ids"] == [0, 1, 2]
    assert res["by_mask_kind"]["uniform"]["n_docs"] == 9
