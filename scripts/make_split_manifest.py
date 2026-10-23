#!/usr/bin/env python3
"""Freeze the fit/report document split (plan v2 Sec. 3.4, Sec. 5; T2).

    python scripts/make_split_manifest.py --docs runs/corpus/calibration_docs.jsonl \\
        --pilot-ids 0-19 --out artifacts/split_manifest.json

Writes ``{doc_id: 0|1}`` (0 = fit half, 1 = report half) -- the format
``analyze_traces.py --split half --manifest`` reads -- plus a ``.meta.json``
with the seed, the corpus hash and provenance.  The split is
``c3.data.fineweb.document_split(n, seed=20260917)`` over the corpus in file
order; the corpus's ``doc_id`` is its position in that file, which is also the
id every trace carries.

Pilot documents are never reused (Sec. 5): with ``--pilot-ids`` they are drawn
from the main corpus and left out of the manifest, so a trace that contains one
fails loudly in analysis; with ``--pilot-docs`` the pilot is a separate file and
is checked to share no document with the main corpus.

The manifest must exist, and be committed, before any report-half trace is
analysed; the script refuses to overwrite one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.data.fineweb import CorpusSpec, content_hash, document_split
from c3.provenance import collect as provenance, sha256_file


def parse_ids(spec: str) -> list[int]:
    """'0-19' or '0,3,5-7' -> ids."""
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return sorted(set(out))


def build(docs_path: Path, seed: int, pilot_ids=None, pilot_docs=None,
          exclusion_manifest=None, any_ids: bool = False) -> tuple[dict, dict]:
    recs = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = [int(r["doc_id"]) for r in recs]
    if not any_ids and ids != list(range(len(ids))):
        raise ValueError("doc_id is not the file position 0..n-1; the split would not "
                         "line up with the ids the traces carry (--any-ids for a corpus "
                         "whose ids are not positions, e.g. the time-split set)")
    if len(set(ids)) != len(ids):
        raise ValueError("doc_id values repeat")
    if exclusion_manifest is not None:
        held_out = set(json.loads(Path(exclusion_manifest).read_text())["excluded_hashes"])
        if {r["hash"] for r in recs} != held_out:
            raise ValueError("the corpus's content hashes differ from the exclusion manifest")
    # The split is over file positions; with --any-ids each position maps to its doc_id.
    fit_idx, report_idx = document_split(len(ids), seed=seed)
    half = {ids[int(i)]: 0 for i in fit_idx} | {ids[int(i)]: 1 for i in report_idx}

    excluded: dict[int, str] = {}
    pilot_meta: dict = {}
    if pilot_ids:
        for d in pilot_ids:
            if d not in half:
                raise ValueError(f"pilot doc_id {d} is not in the corpus")
            excluded[d] = "pilot"
        pilot_meta = {"mode": "main-corpus ids", "ids": pilot_ids}
    elif pilot_docs:
        pilot = [json.loads(line) for line in Path(pilot_docs).read_text(
            encoding="utf-8").splitlines() if line.strip()]
        main_hashes = {r["hash"] for r in recs}
        main_ids = {str(r.get("id")) for r in recs}
        shared = [p for p in pilot if content_hash(p["text"]) in main_hashes
                  or str(p.get("id")) in main_ids]
        if shared:
            raise ValueError(f"{len(shared)} pilot documents are also in the main corpus")
        pilot_meta = {"mode": "separate file", "path": str(pilot_docs), "n": len(pilot),
                      "sha256": sha256_file(pilot_docs)}
    manifest = {str(d): h for d, h in sorted(half.items()) if d not in excluded}
    meta = {"seed": seed, "n_corpus": len(ids), "n_fit": sum(v == 0 for v in manifest.values()),
            "n_report": sum(v == 1 for v in manifest.values()),
            "excluded": {str(d): r for d, r in excluded.items()}, "pilot": pilot_meta,
            "corpus": str(docs_path), "corpus_sha256": sha256_file(docs_path),
            "split_function": "c3.data.fineweb.document_split", "halves": "0 = fit, 1 = report"}
    return manifest, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True, help="calibration_docs.jsonl")
    pilot = ap.add_mutually_exclusive_group(required=True)
    pilot.add_argument("--pilot-ids", help="main-corpus doc_ids used as the pilot, e.g. 0-19")
    pilot.add_argument("--pilot-docs", help="a separate pilot JSONL (checked to be disjoint)")
    ap.add_argument("--seed", type=int, default=CorpusSpec().seed)
    ap.add_argument("--exclusion-manifest", default="artifacts/exclusion_manifest.json",
                    help="check the corpus against it ('' to skip)")
    ap.add_argument("--any-ids", action="store_true",
                    help="allow doc_ids that are not the file positions (split over file order, "
                         "keyed by doc_id): the time-split set uses 200000-200499")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; the split is frozen once written (hard rule 5)")
    manifest, meta = build(Path(a.docs), a.seed,
                           parse_ids(a.pilot_ids) if a.pilot_ids else None, a.pilot_docs,
                           a.exclusion_manifest or None, any_ids=a.any_ids)
    meta["provenance"] = provenance()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=0))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[split] {meta['n_fit']} fit / {meta['n_report']} report documents, "
          f"{len(meta['excluded'])} excluded -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
