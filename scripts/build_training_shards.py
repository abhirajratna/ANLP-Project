#!/usr/bin/env python3
"""Tokenise the 110M training stream, with the calibration documents removed."""
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
    ap.add_argument("--manifest", required=True, help="exclusion_manifest.json")
    ap.add_argument("--target-tokens", type=int, default=6_000_000_000)
    ap.add_argument("--shard-tokens", type=int, default=200_000_000)
    ap.add_argument("--tokenizer", default="gpt2")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    build_token_shards(a.out_dir, a.manifest, tok,
                       target_tokens=a.target_tokens, shard_tokens=a.shard_tokens)
    return 0


if __name__ == "__main__":
    # hard_exit: HF streaming leaves threads that abort at finalization (see c3.util)
    hard_exit(main())
