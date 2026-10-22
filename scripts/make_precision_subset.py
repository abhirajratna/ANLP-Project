#!/usr/bin/env python3
"""The T8 precision documents: 200 of the 2,000, drawn with a fixed seed.

    python scripts/make_precision_subset.py --docs runs/corpus/calibration_docs.jsonl \\
        --manifest artifacts/split_manifest.json --out-docs runs/corpus/precision_docs.jsonl \\
        --out artifacts/t8_precision_docs.json

Half from the fit half and half from the report half of the committed split:
the 4-bit run needs fit-half documents for its own calibrators (E2) and
report-half documents for the numbers, exactly like the main runs.  The subset
JSONL keeps each record's ``doc_id``, so the 4-bit trace carries the same ids as
the fp16 trace and the two pair up document by document.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3.provenance import collect as provenance, sha256_file
from c3.rng import generator


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out-docs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    split = {int(k): int(v) for k, v in json.loads(Path(a.manifest).read_text()).items()}
    recs = {json.loads(line)["doc_id"]: line for line in
            Path(a.docs).read_text(encoding="utf-8").splitlines() if line.strip()}
    rng = generator(a.seed, "t8-precision-docs")
    chosen = []
    for half in (0, 1):
        ids = np.array(sorted(d for d, h in split.items() if h == half and d in recs))
        chosen += sorted(rng.choice(ids, size=a.n // 2, replace=False).tolist())
    chosen = sorted(int(d) for d in chosen)
    Path(a.out_docs).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_docs, "w", encoding="utf-8") as fh:
        for d in chosen:
            fh.write(recs[d] + "\n")
    out = {"doc_ids": chosen, "n": len(chosen), "seed": a.seed,
           "n_fit_half": sum(split[d] == 0 for d in chosen),
           "n_report_half": sum(split[d] == 1 for d in chosen),
           "source": a.docs, "source_sha256": sha256_file(a.docs),
           "manifest_sha256": sha256_file(a.manifest),
           "subset_file": a.out_docs, "subset_sha256": sha256_file(a.out_docs),
           "provenance": provenance()}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[precision] {len(chosen)} documents ({out['n_fit_half']} fit / "
          f"{out['n_report_half']} report) -> {a.out_docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
