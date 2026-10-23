#!/usr/bin/env python3
"""T13 ablation 3: manifests that fit C3's calibrators on fewer documents.

    python scripts/make_size_manifests.py --manifest runs/dev/main-110m-v4.fit.manifest.json \\
        --sizes 100 300 all --out-dir runs/t13/sizes

From a {doc_id: 0 fit | 1 score} manifest, writes ``n<size>.manifest.json`` for every
size: the first <size> fit documents of a hashed order (``c3.frame.nested_doc_subsets``,
so smaller sets sit inside larger ones), and every score document unchanged.  Fit
documents left out are dropped, not moved to the scored side, so every size is scored
on the same documents.  ``all`` keeps every fit document.  Feed each to
``analyze_setcal.py --manifest ... --save-calibrators`` and the saved calibrators to
``eval_infill.py --cal-tag n<size>``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.frame import nested_doc_subsets
from c3.provenance import sha256_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="{doc_id: 0 fit | 1 score}")
    ap.add_argument("--sizes", nargs="+", required=True, help="numbers of fit documents, or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    split = {int(k): int(v) for k, v in json.loads(Path(a.manifest).read_text()).items()}
    fit = [d for d, h in split.items() if h == 0]
    scored = {d: 1 for d, h in split.items() if h == 1}
    sizes = sorted({len(fit) if s == "all" else int(s) for s in a.sizes})
    subsets = nested_doc_subsets(fit, sizes, seed=a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = {"source": a.manifest, "source_sha256": sha256_file(a.manifest), "seed": a.seed,
             "n_scored": len(scored), "sizes": {}}
    for n, ids in subsets.items():
        p = out / f"n{n}.manifest.json"
        p.write_text(json.dumps({**{str(d): 0 for d in ids.tolist()},
                                 **{str(d): 1 for d in scored}}))
        index["sizes"][str(n)] = str(p)
        print(f"[sizes] {n} fit + {len(scored)} scored documents -> {p}")
    (out / "sizes.json").write_text(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
