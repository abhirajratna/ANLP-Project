#!/usr/bin/env python3
"""T13 ablation 4: fit-half documents long enough to trace at 512 and 1,024 tokens.

    python scripts/make_long_subset.py --docs runs/corpus/calibration_docs.jsonl \\
        --manifest artifacts/split_manifest.json --model llada-8b-base --min-tokens 1024 \\
        --n 100 --out-docs runs/corpus/long_docs.jsonl --out artifacts/t13_long_docs.json \\
        --out-manifest artifacts/t13_long_docs.dev.manifest.json

Keeps fit-half documents only (manifest value 0: development, the report half is never
traced here) whose text reaches ``--min-tokens`` under the model's tokenizer, and takes
``--n`` of them in a hashed order (``c3.frame.nested_doc_subsets``).  Tracing the same
documents at 256, 512 and 1,024 tokens (``collect_traces.py --max-tokens``) makes the
three lengths paired: each is the same document's prefix, so only the context length
and field sizes change.  Records keep their ``doc_id``.  ``--out-manifest`` splits the
kept documents into dev-fit (0) and dev-eval (1) by ``hash_halves(doc_ids, 1)``, the
partition every other development analysis uses, for ``analyze_traces.py --split half``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.frame import hash_halves, nested_doc_subsets
from c3.provenance import collect as provenance, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--manifest", required=True, help="the committed split; fit half = 0")
    ap.add_argument("--model", default="llada-8b-base", help="whose tokenizer counts the tokens")
    ap.add_argument("--min-tokens", type=int, default=1024)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-manifest", default=None, help="dev-fit / dev-eval split of the kept documents")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from c3.traces.models import KNOWN_MODELS

    spec = KNOWN_MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(spec["path"], revision=spec["revision"],
                                        trust_remote_code=True)
    split = {int(k): int(v) for k, v in json.loads(Path(a.manifest).read_text()).items()}
    recs, long_enough = {}, []
    for line in Path(a.docs).read_text().splitlines():
        rec = json.loads(line)
        d = int(rec["doc_id"])
        if split.get(d) != 0:
            continue
        n_tok = len(tok(rec["text"], truncation=True, max_length=a.min_tokens,
                        add_special_tokens=False)["input_ids"])
        if n_tok >= a.min_tokens:
            recs[d] = line
            long_enough.append(d)
    keep = nested_doc_subsets(long_enough, [a.n], seed=a.seed)[a.n].tolist()
    out_docs = Path(a.out_docs)
    out_docs.parent.mkdir(parents=True, exist_ok=True)
    out_docs.write_text("".join(recs[d] + "\n" for d in keep))
    summary = {"what": "T13 ablation 4: fit-half documents of at least min_tokens tokens",
               "docs": a.docs, "docs_sha256": sha256_file(a.docs), "manifest": a.manifest,
               "model": a.model, "tokenizer_revision": spec["revision"],
               "min_tokens": a.min_tokens, "n_fit_long_enough": len(long_enough),
               "n": len(keep), "seed": a.seed, "doc_ids": keep,
               "out_docs": str(out_docs), "out_docs_sha256": sha256_file(out_docs),
               "provenance": provenance()}
    if a.out_manifest:
        Path(a.out_manifest).write_text(json.dumps(
            {str(d): int(h) for d, h in zip(keep, hash_halves(keep, 1))}))
        summary["dev_manifest"] = a.out_manifest
    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(f"[long] {len(long_enough)} fit-half documents reach {a.min_tokens} tokens; "
          f"kept {len(keep)} -> {out_docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
