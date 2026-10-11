#!/usr/bin/env python3
"""Validate one merged collection run and write its summary (step 6 of the C1 sweep).

    python scripts/validate_run.py --dataset runs/traces/c1-llada-base/c1-llada-base.dataset.json \\
        --run c1-llada-base --out artifacts/c1/c1-llada-base.json

``--dataset`` is a zero-copy manifest or a merged parquet (scripts/merge_traces.py),
whose merge already refused mismatched shard identities, missing or repeated
documents and repeated rows.  This adds the checks the merge cannot make:

* rows per document as the design implies: decoder and matched-uniform states
  are deterministic in the answer span length n (k = 2, 4, 8 commit k per step
  down to ``min_masked``), so their counts must match exactly; uniform masks are
  Bernoulli, so their count must lie within 6 sd of its mean;
* forward passes per document against the same closed form;
* no non-finite ell or conf anywhere (the harness refuses inf/NaN logits, so a
  complete run should have none; this proves it);
* one GPU model, one set of library versions and one git commit across shards.

The summary (documents, rows, passes, wall time, GPU-hours, throughput) is a
run diagnostic, not a result: nothing in it is an endpoint.  Exit status 1 if
any check fails.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3.assemble import trace_meta
from c3.provenance import collect as provenance, sha256_file
from c3.traces.shards import dataset_files, is_dataset


def expected_rows(n: int, cfg: dict) -> dict:
    """Closed-form rows and passes for one document with an n-token maskable span."""
    m = int(cfg.get("min_masked", 1))
    dec_rows = dec_states = 0
    if cfg.get("include_decoder", True):
        for k in cfg["decoder_k"]:
            for _ in cfg.get("decoder_policies", ["topk_conf"]):
                left, states = n, 0
                while left >= max(m, 1):
                    dec_rows += left
                    states += 1
                    left -= min(int(k), left - m + 1) if left - m + 1 > 0 else left
                dec_states += states
    if cfg.get("decoder_states") is not None:
        raise ValueError("subsampled decoder states (v1) have no closed form here")
    rates = np.asarray(cfg["mask_rates"], dtype=np.float64)
    reps = int(cfg["masks_per_rate"])
    uni = bool(cfg.get("include_uniform", True))
    matched = bool(cfg.get("include_uniform_matched", True)) and cfg.get("include_decoder", True)
    return {"decoder": dec_rows, "uniform_matched": dec_rows if matched else 0,
            "uniform_mean": float(reps * np.sum(rates * n)) if uni else 0.0,
            "uniform_sd": float(np.sqrt(reps * np.sum(n * rates * (1 - rates)))) if uni else 0.0,
            "passes": (reps * rates.size if uni else 0) + dec_states * (2 if matched else 1)}


def scan(files) -> tuple[dict, dict]:
    """Per-document row counts by mask kind, span length, and non-finite counts."""
    import pyarrow.parquet as pq

    counts: dict = defaultdict(lambda: defaultdict(int))
    span: dict = {}
    bad = {"nonfinite_ell": 0, "nonfinite_conf": 0, "conf_outside_unit": 0, "rows": 0}
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in ("doc_id", "mask_kind", "span_len", "doc_len", "ell", "conf")
                if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(columns=cols, batch_size=1 << 20):
            d = batch.column("doc_id").to_numpy()
            kind = np.asarray(batch.column("mask_kind").to_pylist(), dtype=object)
            n = (batch.column("span_len") if "span_len" in cols else batch.column("doc_len")).to_numpy()
            ell = batch.column("ell").to_numpy(zero_copy_only=False)
            conf = batch.column("conf").to_numpy(zero_copy_only=False)
            bad["nonfinite_ell"] += int((~np.isfinite(ell)).sum())
            bad["nonfinite_conf"] += int((~np.isfinite(conf)).sum())
            bad["conf_outside_unit"] += int(((conf < 0) | (conf > 1)).sum())
            bad["rows"] += int(d.size)
            for kd in np.unique(kind):
                sel = kind == kd
                ids, c = np.unique(d[sel], return_counts=True)
                for i, cc in zip(ids.tolist(), c.tolist()):
                    counts[i][kd] += cc
            ids, first = np.unique(d, return_index=True)
            for i, j in zip(ids.tolist(), first.tolist()):
                span[i] = int(n[j])
    return counts, {"span": span, **bad}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="*.dataset.json manifest or merged parquet")
    ap.add_argument("--run", required=True, help="run name (the summary's file name)")
    ap.add_argument("--gpus-per-task", type=int, default=2)
    ap.add_argument("--corpus", default=None,
                    help="the input JSONL, to report how many documents the length filter dropped")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta = trace_meta(a.dataset)
    if not meta:
        raise SystemExit(f"{a.dataset}: no merged meta; run scripts/merge_traces.py first")
    files = dataset_files(a.dataset, verify=True) if is_dataset(a.dataset) else [Path(a.dataset)]
    cfg = meta["config"]
    counts, scanned = scan(files)
    checks: dict = {}

    # documents
    docs = sorted(counts)
    want = sorted(meta.get("doc_ids", []))
    checks["every_document_present"] = {"ok": docs == want and len(docs) > 0,
                                        "n_docs": len(docs), "n_expected": len(want)}
    if a.corpus:
        n_corpus = sum(1 for line in open(a.corpus, encoding="utf-8") if line.strip())
        checks["every_document_present"]["n_corpus"] = n_corpus
        checks["every_document_present"]["dropped_by_length_filter"] = n_corpus - len(want)

    # rows per document
    bad_docs = []
    per_doc_rows = []
    z_uni = []
    for d in docs:
        e = expected_rows(scanned["span"][d], cfg)
        c = counts[d]
        per_doc_rows.append(sum(c.values()))
        problems = []
        for kd in ("decoder", "uniform_matched"):
            if c.get(kd, 0) != e[kd]:
                problems.append(f"{kd} {c.get(kd, 0)} != {e[kd]}")
        if e["uniform_sd"] > 0:
            z = (c.get("uniform", 0) - e["uniform_mean"]) / e["uniform_sd"]
            z_uni.append(z)
            if abs(z) > 6:
                problems.append(f"uniform {c.get('uniform', 0)} is {z:+.1f} sd from {e['uniform_mean']:.0f}")
        if problems:
            bad_docs.append({"doc_id": d, "problems": problems})
    checks["rows_per_document"] = {
        "ok": not bad_docs, "n_bad": len(bad_docs), "first_bad": bad_docs[:10],
        "mean_rows_per_doc": float(np.mean(per_doc_rows)) if per_doc_rows else None,
        "uniform_count_z": {"mean": float(np.mean(z_uni)) if z_uni else None,
                            "sd": float(np.std(z_uni)) if z_uni else None}}

    # forward passes (those spent on documents later skipped for overflow are not
    # in the closed form of the documents present)
    passes = int(meta.get("forward_passes", 0)) - int(meta.get("forward_passes_failed_docs", 0))
    exp_passes = int(sum(expected_rows(scanned["span"][d], cfg)["passes"] for d in docs))
    bs = int(cfg.get("batch_size", 1))
    checks["forward_passes"] = {"ok": passes == exp_passes if bs == 1 else True,
                                "recorded": passes, "expected": exp_passes if bs == 1 else None,
                                "per_doc": passes / max(len(docs), 1),
                                **({} if bs == 1 else {"note": f"closed form needs batch_size 1, "
                                                               f"this run used {bs}"})}

    # non-finite values
    nf = scanned["nonfinite_ell"] + scanned["nonfinite_conf"] + scanned["conf_outside_unit"]
    failed = meta.get("doc_ids_failed", [])
    checks["no_nonfinite_values"] = {"ok": nf == 0, **{k: scanned[k] for k in
                                     ("nonfinite_ell", "nonfinite_conf", "conf_outside_unit")},
                                     "n_saturated": meta.get("n_saturated")}
    checks["no_flagged_nonfinite_passes"] = {
        "ok": not failed, "n_documents_skipped": len(failed), "doc_ids": failed,
        "errors": meta.get("nonfinite_docs", [])[:10],
        "note": "documents whose forward passes overflowed (inf/NaN logits) and were left out"}

    # identity and provenance across shards
    prov = [p or {} for p in meta.get("shard_provenance", [])]
    gpus = sorted({str((p.get("gpu") or {}).get("gpu", p.get("gpu"))) for p in prov})
    pkgs = sorted({json.dumps(p.get("packages", {}), sort_keys=True) for p in prov})
    commits = meta["git_commit"] if isinstance(meta["git_commit"], list) else [meta["git_commit"]]
    checks["one_identity_across_shards"] = {
        "ok": len(gpus) <= 1 and len(pkgs) <= 1 and len(commits) == 1,
        "gpus": gpus, "n_package_sets": len(pkgs), "commits": commits,
        "identity_keys": "config, model, variant, revision, precision, dataset, schema "
                         "(refused at merge if they differ)"}

    stats = meta.get("shard_stats", [])
    secs = [float(s.get("seconds", 0.0)) for s in stats]
    summary = {
        "run": a.run, "model": meta.get("model"), "variant": meta.get("variant"),
        "precision": meta.get("precision"), "dataset": meta.get("dataset"),
        "revision": meta.get("revision"), "device_map": meta.get("device_map"),
        "documents": len(docs), "rows": int(meta.get("n_rows", scanned["rows"])),
        "forward_passes": passes, "num_shards": meta.get("num_shards"),
        "collection_seconds_total": float(sum(secs)),
        "collection_seconds_longest_shard": float(max(secs)) if secs else None,
        "gpu_hours": float(sum(secs)) * a.gpus_per_task / 3600.0,
        "passes_per_second": passes / sum(secs) if sum(secs) > 0 else None,
        "docs_per_hour_per_task": len(docs) / (sum(secs) / 3600.0) if sum(secs) > 0 else None,
        "design": {k: cfg.get(k) for k in ("mask_rates", "masks_per_rate", "decoder_k",
                                           "include_uniform_matched", "topk", "batch_size")},
        "gpus_per_task": a.gpus_per_task,
    }
    out = {"note": "run diagnostics for the C1 sweep, not results: no endpoint is computed here",
           "dataset": str(a.dataset), "dataset_sha256": sha256_file(a.dataset),
           "summary": summary, "checks": checks, "all_ok": all(c["ok"] for c in checks.values()),
           "provenance": provenance(),
           "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    for name, c in checks.items():
        print(f"[validate] {'ok  ' if c['ok'] else 'FAIL'} {name}")
    print(f"[validate] {summary['documents']} docs, {summary['rows']} rows, {passes} passes, "
          f"{summary['gpu_hours']:.1f} GPU-h -> {a.out}")
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
