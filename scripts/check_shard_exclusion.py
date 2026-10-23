#!/usr/bin/env python3
"""Check built training shards against exclusion manifests, after the fact.

    python scripts/check_shard_exclusion.py --index <shards>/index.json \\
        --manifest artifacts/exclusion_timesplit.json --out artifacts/shard_check_timesplit.json

For a manifest that was not available when the shards were built (the v2 shards
were rebuilt on 2026-10-12 before the time-split set existed).  Every document in
the train and val shards is hashed over its first 256 tokens -- the same
``token_hash`` the build applies -- and looked up in the manifests.  Raw text is
not stored in the shards, so the text-hash key cannot be rechecked; an exact
text duplicate has identical tokens, so the token key covers it.  FineWeb ids
cannot match an external set's ids.  Exit status 1 if any document matches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.provenance import collect as provenance, sha256_file
from c3.training.exclusion import TOKEN_WINDOW, Exclusion, token_hash


def scan(index_path: str | Path, excl: Exclusion) -> dict:
    index_path = Path(index_path)
    idx = json.loads(index_path.read_text())
    out = {"documents": 0, "matches": []}
    for split in ("shards", "val_shards"):
        for s in idx.get(split, []):
            path = index_path.parent / Path(s["path"]).name
            arr = np.memmap(path, dtype=np.uint16, mode="r")
            off = np.load(index_path.parent / Path(s["offsets"]).name)
            for d in range(off.size - 1):
                a, b = int(off[d]), int(off[d + 1])
                h = token_hash(arr[a:min(b, a + TOKEN_WINDOW)])
                if excl.check(tokens=h):
                    out["matches"].append({"split": split, "shard": path.name, "doc": d,
                                           "token_hash": h})
            out["documents"] += off.size - 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True, nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    excl = Exclusion.load(*a.manifest)
    res = scan(a.index, excl)
    summary = excl.summary()
    print(f"[check] {res['documents']:,} shard documents scanned; "
          f"{len(res['matches'])} match the manifests by first-{TOKEN_WINDOW}-token hash; "
          f"by role {summary['by_role']}")
    if a.out:
        Path(a.out).write_text(json.dumps({
            "index": str(a.index), "index_sha256": sha256_file(a.index),
            "manifests": {m: sha256_file(m) for m in a.manifest},
            "documents_scanned": res["documents"], "matches": res["matches"],
            "summary": summary, "provenance": provenance()}, indent=1))
    return 1 if res["matches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
