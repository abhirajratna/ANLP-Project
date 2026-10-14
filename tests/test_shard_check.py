"""scripts/check_shard_exclusion.py: a manifest checked against shards built without it."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from c3.training.data import build_token_shards
from c3.training.exclusion import Exclusion, ExclusionEntry, token_hash

ROOT = Path(__file__).resolve().parents[1]


def _check_module():
    spec = importlib.util.spec_from_file_location("check_shard_exclusion",
                                                  ROOT / "scripts" / "check_shard_exclusion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tok(texts, add_special_tokens=False):
    if isinstance(texts, str):
        return {"input_ids": [ord(c) % 50000 for c in texts]}
    return {"input_ids": [[ord(c) % 50000 for c in t] for t in texts]}


def test_a_document_built_into_the_shards_is_found_and_others_are_not(tmp_path):
    rng = np.random.default_rng(0)
    recs = [{"id": f"doc{i}", "text": "".join(chr(97 + int(c)) for c in rng.integers(0, 26, 400))}
            for i in range(300)]
    built = tmp_path / "built.json"
    Exclusion([ExclusionEntry(role="main", id="doc0", from_stream=True)]).save(built)
    build_token_shards(tmp_path / "shards", built, _tok, target_tokens=10 ** 9,
                       shard_tokens=20_000, val_fraction=0.05, records=recs,
                       content_hash_fn=lambda t: "h" + t[:16])
    index = tmp_path / "shards" / "index.json"
    assert json.loads(index.read_text())["val_documents"] > 0

    mod = _check_module()
    # doc0 was excluded at build time; doc7 and doc200 were not, so they are in the shards.
    late = tmp_path / "late.json"
    Exclusion([ExclusionEntry(role="timesplit", token_hash=token_hash(_tok(recs[i]["text"])["input_ids"]),
                              from_stream=False) for i in (0, 7, 200)]).save(late)
    res = mod.scan(index, Exclusion.load(late))
    assert res["documents"] == 299
    assert len(res["matches"]) == 2

    clean = tmp_path / "clean.json"
    Exclusion([ExclusionEntry(role="timesplit", token_hash=token_hash([1, 2, 3]),
                              from_stream=False)]).save(clean)
    assert mod.scan(index, Exclusion.load(clean))["matches"] == []
