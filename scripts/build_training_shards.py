#!/usr/bin/env python3
"""Tokenise the 110M training stream, with every evaluation document removed.

Writes document-aligned shards (no packing), a small validation split, and an
index.json whose ``exclusion`` block records how many held-out documents were
matched by id / text hash / first-256-token hash.  Fails if a held-out document
that was sampled from this stream is never matched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.training.data import build_token_shards
from c3.util import hard_exit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="node-local /scratch is the right place")
    ap.add_argument("--manifest", required=True, nargs="+",
                    help="exclusion manifest(s): v2 JSON from build_exclusion.py, or the "
                         "proposal-era exclusion_manifest.json")
    ap.add_argument("--target-tokens", type=int, default=10_000_000_000)
    ap.add_argument("--shard-tokens", type=int, default=200_000_000)
    ap.add_argument("--val-fraction", type=float, default=0.0005)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--allow-unmatched", action="store_true",
                    help="write shards even if some held-out documents were never matched")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    build_token_shards(a.out_dir, a.manifest, tok, target_tokens=a.target_tokens,
                       shard_tokens=a.shard_tokens, val_fraction=a.val_fraction,
                       fail_on_unmatched=not a.allow_unmatched)
    return 0


if __name__ == "__main__":
    # hard_exit: HF streaming leaves threads that abort at finalization (see c3.util)
    hard_exit(main())
