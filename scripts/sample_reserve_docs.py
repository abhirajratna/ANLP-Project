#!/usr/bin/env python3
"""Sample spare held-out documents (pilot, replacements) before the shards are built.

Writes ``reserve_docs.jsonl`` in the calibration corpus's format.  Run it once,
before ``build_exclusion.py``, so the reserve is excluded from the 110M run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.util import hard_exit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True, help="calibration_docs.jsonl (ids to skip)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-documents", type=int, default=500)
    ap.add_argument("--pool", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args()

    from c3.data.fineweb import content_hash, fineweb_stream
    from c3.training.exclusion import sample_reserve

    main_ids = [json.loads(line)["id"] for line in open(a.calibration, encoding="utf-8") if line.strip()]
    docs = sample_reserve(fineweb_stream(), main_ids, a.n_documents, a.pool, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i, d in enumerate(docs):
            fh.write(json.dumps({"doc_id": 100_000 + i, "id": d["id"], "text": d["text"],
                                 "hash": content_hash(d["text"])}) + "\n")
    print(f"[reserve] {len(docs)} documents -> {out} (doc_id from 100000, disjoint from the main 2,000)")
    return 0


if __name__ == "__main__":
    hard_exit(main())
