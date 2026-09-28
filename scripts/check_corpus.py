#!/usr/bin/env python3
"""Check a calibration corpus file against the committed exclusion record.

``artifacts/exclusion_manifest.json`` is the authoritative record of the 2,000
held-out documents.  A corpus regenerated on another account (same seed, same
stream) or copied from a teammate must match it exactly -- every text hash and
every FineWeb id -- before anything is built from it.  Exits non-zero on any
difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.data.fineweb import content_hash


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="calibration_docs.jsonl")
    ap.add_argument("--manifest", default="artifacts/exclusion_manifest.json")
    a = ap.parse_args()

    man = json.loads(Path(a.manifest).read_text())
    want_h, want_i = set(man["excluded_hashes"]), {str(i) for i in man["excluded_ids"]}
    got_h, got_i, n = set(), set(), 0
    for line in open(a.docs, encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            got_h.add(content_hash(rec["text"]))
            got_i.add(str(rec["id"]))
            n += 1
    ok = got_h == want_h and got_i == want_i and n == len(want_h)
    print(f"documents: {n} (manifest {len(want_h)})")
    print(f"text hashes: {len(got_h & want_h)} match, {len(got_h - want_h)} extra, {len(want_h - got_h)} missing")
    print(f"ids:         {len(got_i & want_i)} match, {len(got_i - want_i)} extra, {len(want_i - got_i)} missing")
    print("IDENTICAL to the committed held-out record" if ok else "MISMATCH -- do not build from this corpus")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
