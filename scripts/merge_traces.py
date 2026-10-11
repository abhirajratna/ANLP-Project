#!/usr/bin/env python3
"""Merge the shards of one collection run into a single parquet trace.

    python scripts/merge_traces.py --stem runs/traces/llada-base/llada-base \\
        --out runs/traces/llada-base.parquet

With ``--out <run>.dataset.json`` the merge is zero-copy: the same checks, and a
manifest of the validated parts (path, sha256, rows) instead of a second copy of
the data.  Every reader (``c3.assemble.load_traces``) accepts the manifest.

Fails, without writing, if the shard metas disagree on config, model, variant,
revision, precision, dataset or schema; if a shard is incomplete; or if any
document or (state_id, position) row appears twice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.traces.shards import MergeError, merge


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True, help="the --out stem given to collect_traces.py")
    ap.add_argument("--out", required=True,
                    help="merged parquet path, or <run>.dataset.json for a zero-copy merge")
    ap.add_argument("--allow-mixed-commits", action="store_true",
                    help="accept shards collected at different git commits")
    a = ap.parse_args()
    try:
        merge(a.stem, a.out, allow_mixed_commits=a.allow_mixed_commits)
    except MergeError as exc:
        print(f"[merge] REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
