#!/usr/bin/env python3
"""Build the v2 exclusion manifest: every document the study evaluates on.

Each input is ROLE=PATH to a JSONL corpus file (``text`` plus the source ``id``
where available), e.g.

    python scripts/build_exclusion.py --out runs/corpus/exclusion_v2.json \\
        main=runs/corpus/calibration_docs.jsonl \\
        pilot=runs/corpus/pilot_docs.jsonl \\
        timesplit=runs/corpus/timesplit_docs.jsonl:external

``:external`` marks a set that was not sampled from the FineWeb stream (the
time-split set), so the shard build does not demand that it be matched.  Keys
per document: FineWeb id, raw-text hash (the same function the shard builder
uses) and the hash of the first 256 GPT-2 tokens.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.training.exclusion import Exclusion, entries_from_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="ROLE=PATH[:external]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="gpt2")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    entries = []
    for spec in a.inputs:
        role, _, path = spec.partition("=")
        external = path.endswith(":external")
        path = path[: -len(":external")] if external else path
        got = entries_from_jsonl(path, role, tok, from_stream=not external)
        print(f"[exclusion] {role}: {len(got)} documents from {path}"
              f"{' (external)' if external else ''}; "
              f"{sum(e.id is not None for e in got)} with ids")
        entries += got
    ex = Exclusion(entries)
    ex.save(a.out)
    print(f"[exclusion] wrote {len(entries)} entries -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
