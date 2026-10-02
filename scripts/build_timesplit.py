#!/usr/bin/env python3
"""Build the 500-document time-split set (plan v2 T4 step 2; see c3.data.timesplit).

    python scripts/build_timesplit.py --out runs/corpus/timesplit_docs.jsonl \\
        --manifest artifacts/timesplit_manifest.json

Needs network access to en.wikipedia.org and the three audited tokenizers
(GPT-2 for the 110M model, LLaDA-8B, Dream-7B) to check every document reaches
256 tokens under each.  Then record the set in a v2 exclusion manifest:

    python scripts/build_exclusion.py --out artifacts/exclusion_timesplit.json \\
        timesplit=runs/corpus/timesplit_docs.jsonl:external
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.data.timesplit import TimeSplitSpec, build
from c3.provenance import collect as provenance, sha256_file
from c3.traces.models import KNOWN_MODELS
from c3.util import hard_exit


def token_counters() -> dict:
    from transformers import AutoTokenizer

    out = {"gpt2": AutoTokenizer.from_pretrained("gpt2")}
    for key in ("llada-8b-base", "dream-7b-base"):
        spec = KNOWN_MODELS[key]
        out[key] = AutoTokenizer.from_pretrained(spec["path"], revision=spec["revision"],
                                                 trust_remote_code=True)
    return {name: (lambda text, t=tok: len(t(text, add_special_tokens=False)["input_ids"]))
            for name, tok in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=TimeSplitSpec.n_documents)
    ap.add_argument("--seed", type=int, default=TimeSplitSpec.seed)
    ap.add_argument("--as-of", default=TimeSplitSpec.as_of)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; the time-split set is fixed once built")
    spec = TimeSplitSpec(n_documents=a.n, seed=a.seed, as_of=a.as_of)
    docs, info = build(spec, token_counters())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    created = sorted(d["created"] for d in docs)
    manifest = {**info, "n_documents": len(docs), "file": out.name, "sha256": sha256_file(out),
                "created_range": [created[0], created[-1]],
                "doc_id_range": [docs[0]["doc_id"], docs[-1]["doc_id"]],
                "ids": [d["id"] for d in docs], "hashes": [d["hash"] for d in docs],
                "provenance": provenance()}
    Path(a.manifest).write_text(json.dumps(manifest, indent=1))
    print(f"[timesplit] {len(docs)} documents created {created[0]} .. {created[-1]} -> {out}")
    print(f"[timesplit] rejected: {info['rejected']}")
    return 0


if __name__ == "__main__":
    hard_exit(main())
