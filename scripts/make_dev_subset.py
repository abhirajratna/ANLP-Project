#!/usr/bin/env python3
"""A development copy of a merged trace that holds only fit-half documents.

    python scripts/make_dev_subset.py --dataset <run>.dataset.json \\
        --manifest artifacts/split_manifest.json --n-docs 150 \\
        --out-trace runs/dev/<run>.fit150.parquet --out-manifest runs/dev/<run>.fit150.manifest.json

The plan's rule (Sec. 5) is that nothing is computed on the report half before the
pre-registration is frozen.  Development on real traces therefore reads only
fit-half documents (manifest value 0): they are selected while reading, so a
report-half row never reaches the analysis.  ``--out-manifest`` splits those fit
documents again by ``c3.frame.hash_halves(doc_ids, 1)`` -- the same partition
``analyze_e3.py`` / ``run_controls.py --half fit`` make -- so ``analyze_traces.py
--split half --manifest <out-manifest>`` fits on one part of the fit half and
"reports" on the other.  None of these numbers is a result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3.assemble import SCALAR_COLUMNS, read_trace_table, trace_meta
from c3.frame import hash_halves
from c3.provenance import collect as provenance, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="*.dataset.json manifest or merged parquet")
    ap.add_argument("--manifest", required=True, help="the committed split manifest")
    ap.add_argument("--n-docs", type=int, default=0, help="fit-half documents to keep (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-trace", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--columns", choices=["all", "scalar"], default="all",
                    help="scalar: drop the top-k lists (c3.assemble.SCALAR_COLUMNS), most of a "
                         "trace's size, which the calibration analyses do not read")
    a = ap.parse_args()

    split = {int(k): int(v) for k, v in json.loads(Path(a.manifest).read_text()).items()}
    present = set(int(d) for d in trace_meta(a.dataset).get("doc_ids", []))
    fit = np.array(sorted(d for d, h in split.items() if h == 0 and (not present or d in present)))
    if a.n_docs and a.n_docs < fit.size:
        rng = np.random.default_rng(a.seed)
        fit = np.sort(rng.choice(fit, size=a.n_docs, replace=False))
    if any(split[int(d)] != 0 for d in fit):
        raise SystemExit("a report-half document was selected; refusing")
    columns = list(SCALAR_COLUMNS) if a.columns == "scalar" else None
    table = read_trace_table(a.dataset, columns=columns,
                             filters=[("doc_id", "in", fit.tolist())])
    got = np.unique(table.column("doc_id").to_numpy())
    if not set(got.tolist()) <= set(fit.tolist()):
        raise SystemExit("the filtered trace holds documents outside the selection; refusing")
    import pyarrow.parquet as pq

    out = Path(a.out_trace)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")
    meta = {k: v for k, v in trace_meta(a.dataset).items() if k not in ("files", "doc_ids", "parts")}
    meta.update({"note": "DEVELOPMENT subset: fit-half documents only, not a result",
                 "source": str(a.dataset), "source_sha256": sha256_file(a.dataset),
                 "manifest": a.manifest, "columns": a.columns, "doc_ids": got.tolist(), "n_docs": int(got.size),
                 "n_rows": int(table.num_rows), "provenance": provenance()})
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    h = hash_halves(got, 1)
    Path(a.out_manifest).write_text(json.dumps({str(int(d)): int(x) for d, x in zip(got, h)}))
    print(f"[dev] {got.size} fit-half documents, {table.num_rows} rows -> {out} "
          f"(dev split {int((h == 0).sum())} / {int((h == 1).sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
