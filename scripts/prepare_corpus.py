#!/usr/bin/env python3
"""Sample the held-out calibration corpus and write the exclusion manifest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.data.fineweb import CorpusSpec, sample_documents
from c3.util import hard_exit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-documents", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--pool", type=int, default=200_000)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    spec = CorpusSpec(n_documents=a.n_documents, max_tokens=a.max_tokens,
                      seed=a.seed, reservoir_pool=a.pool)
    sample_documents(spec, a.out_dir)
    return 0


if __name__ == "__main__":
    # hard_exit: HF streaming leaves threads that abort at finalization (see c3.util)
    hard_exit(main())
