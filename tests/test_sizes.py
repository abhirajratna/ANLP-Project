"""T13 ablation 3: nested fitting subsets for C3's calibrators."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from c3.frame import nested_doc_subsets

ROOT = Path(__file__).resolve().parents[1]


def test_subsets_are_nested_and_do_not_depend_on_other_documents():
    ids = np.arange(1000, 1500)
    s = nested_doc_subsets(ids, [100, 300, 500], seed=0)
    assert [v.size for v in s.values()] == [100, 300, 500]
    assert set(s[100]) <= set(s[300]) <= set(s[500])
    # dropping documents outside a subset leaves the subset unchanged
    t = nested_doc_subsets(s[300], [100], seed=0)
    assert np.array_equal(t[100], s[100])
    assert not np.array_equal(nested_doc_subsets(ids, [100], seed=1)[100], s[100])
    with pytest.raises(ValueError):
        nested_doc_subsets(ids, [501])


def test_cli_writes_one_manifest_per_size_with_the_same_scored_documents(tmp_path):
    man = tmp_path / "dev.manifest.json"
    man.write_text(json.dumps({str(d): int(d % 2) for d in range(400)}))
    subprocess.run([sys.executable, "scripts/make_size_manifests.py", "--manifest", str(man),
                    "--sizes", "50", "all", "--out-dir", str(tmp_path / "sizes")],
                   check=True, cwd=ROOT, capture_output=True)
    small = json.loads((tmp_path / "sizes" / "n50.manifest.json").read_text())
    full = json.loads((tmp_path / "sizes" / "n200.manifest.json").read_text())
    assert sum(v == 0 for v in small.values()) == 50 and sum(v == 0 for v in full.values()) == 200
    scored = {k for k, v in full.items() if v == 1}
    assert scored == {k for k, v in small.items() if v == 1} and len(scored) == 200
    assert {k for k, v in small.items() if v == 0} <= {k for k, v in full.items() if v == 0}
