#!/usr/bin/env python3
"""B: the proper-score field test (``c3.endpoints.fieldtest``) on every arm of a trace.

    python scripts/field_test.py --traces runs/dev/main-110m-v4.fit.parquet \\
        --manifest runs/dev/main-110m-v4.fit.manifest.json --out runs/dev/fieldtest.json

The manifest maps every document to 0 (fit the calibrators) or 1 (score them).
Development uses a re-split of the fit half (``make_dev_subset.py``); the frozen
``artifacts/split_manifest.json`` is refused unless C3_REPORT_HALF_OK=1.

Each arm is read on its own with pyarrow filters (a full 110M trace is ~120M
rows), scored with two feature sets -- all field features, and the ones a
decoder can compute at inference time -- and the gate of prereg/deviations.md
item 8 is evaluated: context-aware recalibration keeps its field features only if
the decode-feature log-loss gain is declared positive on a primary arm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3 import DEFAULT_KNOTS, N_BOOT
from c3.assemble import read_arm, trace_doc_ids
from c3.endpoints.fieldtest import DECODE_FEATURES, EPS, field_effect_test
from c3.frame import FIELD_FEATURES, PositionFrame
from c3.provenance import collect as provenance
from c3.stats import doc_boot_counts, holm
from c3.util import require_report_ok

DEFAULT_ARMS = ("uniform", "decoder:topk_conf:k2", "decoder:topk_conf:k4",
                "decoder:topk_conf:k8", "uniform_matched:topk_conf:k4")
PRIMARY_ARMS = ("uniform", "decoder:topk_conf:k4")      # as prereg/analysis_commands.sh
COLUMNS = ["doc_id", "state_id", "position", "ell", "conf", "agree", "mask_kind",
           "mask_rate", "n_masked", "step", "policy", "k"]


def load_arm(path: str, arm: str) -> PositionFrame:
    return PositionFrame.from_frame(read_arm(path, arm, COLUMNS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="parquet trace or *.dataset.json")
    ap.add_argument("--manifest", required=True, help="{doc_id: 0 fit | 1 score}")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    ap.add_argument("--primary-arms", nargs="+", default=list(PRIMARY_ARMS))
    ap.add_argument("--knots", type=int, default=DEFAULT_KNOTS)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--additive", action="store_true", help="no l x feature interactions")
    ap.add_argument("--weighting", choices=["pool", "position"], default="pool")
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot-label", default="c3set-fieldtest")
    ap.add_argument("--floor-ll", type=float, default=0.0)
    ap.add_argument("--floor-brier", type=float, default=0.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    split = {int(k): int(v) for k, v in json.loads(Path(a.manifest).read_text()).items()}
    require_report_ok(a.manifest)
    docs = trace_doc_ids(a.traces)
    unlisted = sorted(set(docs.tolist()) - set(split))
    if unlisted:
        raise SystemExit(f"{len(unlisted)} trace documents are not in the manifest, e.g. {unlisted[:5]}")
    codes = np.array(sorted(d for d in docs.tolist() if split[d] == 1), dtype=np.int64)
    counts = doc_boot_counts(codes.size, a.n_boot, a.seed, label=a.boot_label)
    print(f"[fieldtest] {docs.size} documents: {codes.size} scored, "
          f"{docs.size - codes.size} fitted; B = {a.n_boot}", flush=True)

    feature_sets = {"all": tuple(FIELD_FEATURES), "decode": tuple(DECODE_FEATURES)}
    arms = {}
    for arm in a.arms:
        t0 = time.time()
        pos = load_arm(a.traces, arm)
        half = np.array([split[int(d)] for d in pos.doc_id], dtype=np.int64)
        arms[arm] = {}
        for name, feats in feature_sets.items():
            arms[arm][name] = field_effect_test(
                pos, half == 0, half == 1, codes, counts, features=feats, n_knots=a.knots,
                interactions=not a.additive, ridge=a.ridge, weighting=a.weighting, eps=a.eps,
                floor_ll=a.floor_ll, floor_brier=a.floor_brier)
        ll = arms[arm]["decode"]["spline_vs_context"]["log_loss"]
        print(f"[fieldtest] {arm}: {len(pos):,} positions, decode-feature log-loss gain "
              f"{ll['gain']:+.5f} [{ll['ci'][0]:+.5f}, {ll['ci'][1]:+.5f}] nats "
              f"({ll['rel_gain']:+.2%}), helps={ll['helps']}  ({time.time() - t0:.0f} s)", flush=True)

    primary = [arm for arm in a.primary_arms if arm in arms]
    gate_rows = {arm: arms[arm]["decode"]["spline_vs_context"]["log_loss"] for arm in primary}
    adj, _ = holm([r["p_boot"] for r in gate_rows.values()]) if gate_rows else ([], [])
    gate = {
        "rule": "prereg/deviations.md item 8: field features are kept if the decode-feature "
                "log-loss gain is declared positive on a primary arm",
        "arms": {arm: {"gain": r["gain"], "ci": r["ci"], "helps": r["helps"],
                       "p_boot": r["p_boot"], "p_holm": float(p)}
                 for (arm, r), p in zip(gate_rows.items(), adj)},
        "use_field_features": any(r["helps"] for r in gate_rows.values()),
    }
    print(f"[fieldtest] gate: use field features = {gate['use_field_features']}", flush=True)

    res = {"label": a.label, "input": {"traces": a.traces, "manifest": a.manifest},
           "config": {k: v for k, v in vars(a).items() if k not in ("traces", "manifest", "out")},
           "n_docs": int(docs.size), "n_scored_docs": int(codes.size),
           "feature_sets": {k: list(v) for k, v in feature_sets.items()},
           "arms": arms, "gate": gate, "provenance": provenance()}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"[fieldtest] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
