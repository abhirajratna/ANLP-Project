#!/usr/bin/env python3
"""Check a merged schema-v2 trace against the design: rows per document by mask kind.

    python scripts/validate_traces.py --traces main-110m-v3.parquet \\
        --estimate artifacts/trace_size_estimate.json --out artifacts/traces_110m/validate_main.json

Decoder and matched-uniform rows are deterministic given a document's maskable
length n (k tokens committed per step: n + (n - k) + ... per policy and k), so they
must equal the closed form of scripts/estimate_trace_size.py exactly.  Uniform
masks are Bernoulli, so their rows are checked against the expectation
masks_per_rate * sum(rates) * n with a z-score.  The mean rows per document of
the 256-token documents is set beside artifacts/trace_size_estimate.json.

Only counts are read (doc_id, mask kind, policy, k, span length): no calibration
number is computed, so this is safe to run before the pre-registration is frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.provenance import collect as provenance


def decoder_rows(n: int, k: int) -> int:
    return sum(n - j * k for j in range(-(-n // k)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="merged parquet (with .meta.json beside it)")
    ap.add_argument("--estimate", default="artifacts/trace_size_estimate.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-z", type=float, default=5.0)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    path = Path(a.traces)
    meta = json.loads(path.with_suffix(".meta.json").read_text()) \
        if path.with_suffix(".meta.json").exists() else {}
    cfg = meta.get("config", {})
    rates = cfg.get("mask_rates", [0.1 * i for i in range(1, 10)])
    per_rate = cfg.get("masks_per_rate", 2)
    ks = cfg.get("decoder_k", [2, 4, 8])
    n_pol = len(cfg.get("decoder_policies", ["topk_conf"]))

    # mask_kind as a dictionary column: 1e8 rows of Python strings would not fit in memory
    t = pq.read_table(path, columns=["doc_id", "span_len", "mask_kind", "k"],
                      read_dictionary=["mask_kind"]).to_pandas()
    counts = t.groupby(["doc_id", "mask_kind", "k"], observed=True).size() \
        .unstack(["mask_kind", "k"], fill_value=0)
    span = t.groupby("doc_id")["span_len"].first()
    docs = counts.index.to_numpy()
    bad_exact, z_uniform, totals = [], [], []
    for d in docs:
        n = int(span.loc[d])
        row = counts.loc[d]
        for kind in ("decoder", "uniform_matched"):
            for k in ks:
                got = int(row.get((kind, k), 0))
                want = n_pol * decoder_rows(n, k)
                if got != want:
                    bad_exact.append({"doc_id": int(d), "kind": kind, "k": k, "rows": got,
                                      "expected": want})
        uni = int(sum(v for (kind, _), v in row.items() if kind == "uniform"))
        mean = per_rate * sum(rates) * n
        sd = np.sqrt(per_rate * sum(r * (1 - r) for r in rates) * n)
        z_uniform.append((uni - mean) / sd)
        totals.append(int(row.sum()))
    z = np.array(z_uniform)
    totals = np.array(totals)
    full = span.loc[docs].to_numpy() == 256
    est = json.loads(Path(a.estimate).read_text())["per_document"]["full (plan v2 defaults)"]
    res = {
        "traces": str(path), "bytes": path.stat().st_size, "rows": int(len(t)),
        "documents": int(docs.size), "doc_id_range": [int(docs.min()), int(docs.max())],
        "span_len": {"min": int(span.min()), "max": int(span.max()), "n_256": int(full.sum())},
        "decoder_and_matched_rows_exact": not bad_exact, "mismatches": bad_exact[:20],
        "uniform_rows_z": {"min": float(z.min()), "max": float(z.max()),
                           "n_beyond": int((np.abs(z) > a.max_z).sum()), "limit": a.max_z},
        "rows_per_document": {"mean": float(totals.mean()), "min": int(totals.min()),
                              "max": int(totals.max()),
                              "mean_256_token_docs": float(totals[full].mean()) if full.any() else None},
        "estimate_per_256_token_doc": est,
        "pass": bool(not bad_exact and (np.abs(z) <= a.max_z).all()),
        "provenance": provenance()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"[validate] {path.name}: {res['documents']} docs, {res['rows']:,} rows, "
          f"{res['bytes'] / 1e9:.1f} GB; decoder/matched exact: {res['decoder_and_matched_rows_exact']}; "
          f"uniform z in [{z.min():.2f}, {z.max():.2f}]; rows/doc (256-token) "
          f"{res['rows_per_document']['mean_256_token_docs']} vs estimate {est['rows']} "
          f"(closed form {est['closed_form_rows']}) -> {'PASS' if res['pass'] else 'FAIL'}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
