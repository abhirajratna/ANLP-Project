#!/usr/bin/env python3
"""Relabel the committed time-split set's doc_ids to TIMESPLIT_ID_OFFSET + i (one-off, 2026-10-12).

The set was built (job-free, scripts/build_timesplit.py) with doc_ids 100000 + i,
which collide with the reserve documents' 100000..100499.  Rebuilding would
re-read Wikipedia's *current* revisions and could change the text, so the labels
are rewritten in place instead: text, order, ids, revids and hashes are untouched.
The manifest's sha256 and doc_id_range are updated and the relabelling is recorded.

    python scripts/relabel_timesplit.py --docs artifacts/timesplit_docs.jsonl \\
        --manifest artifacts/timesplit_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.data.timesplit import TIMESPLIT_ID_OFFSET
from c3.provenance import collect as provenance, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    docs_path, man_path = Path(a.docs), Path(a.manifest)
    manifest = json.loads(man_path.read_text())
    if sha256_file(docs_path) != manifest["sha256"]:
        raise SystemExit("docs file does not match the manifest's sha256; not relabelling")
    docs = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines() if line]
    old = [d["doc_id"] for d in docs]
    for i, d in enumerate(docs):
        d["doc_id"] = TIMESPLIT_ID_OFFSET + i
    with docs_path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    manifest["relabelled"] = {"from_doc_id_range": [old[0], old[-1]],
                              "from_sha256": manifest["sha256"],
                              "reason": "doc_ids 100000+i collided with the reserve set's 100000..100499",
                              "provenance": provenance()}
    manifest["sha256"] = sha256_file(docs_path)
    manifest["doc_id_range"] = [docs[0]["doc_id"], docs[-1]["doc_id"]]
    man_path.write_text(json.dumps(manifest, indent=1))
    print(f"[relabel] {len(docs)} doc_ids {old[0]}..{old[-1]} -> "
          f"{docs[0]['doc_id']}..{docs[-1]['doc_id']}; sha256 {manifest['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
