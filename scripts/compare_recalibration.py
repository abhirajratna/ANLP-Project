#!/usr/bin/env python3
"""T13 ablation 2: which recalibration method is enough? (``c3.recal``)

    python scripts/compare_recalibration.py --traces runs/dev/main-110m-v4.fit.parquet \\
        --manifest runs/dev/main-110m-v4.fit.manifest.json --out runs/t13/recal.json \\
        --save-calibrators runs/t13/cal

The manifest maps every document to 0 (fit) or 1 (score); in development it is the
re-split of the fit half (``make_dev_subset.py``), and the frozen split is refused
unless C3_REPORT_HALF_OK=1.  For every arm, each method of ``c3.recal.METHODS`` is
fitted on the arm's fit positions (pool weights) and scored on its scored positions:

* equal-mass ECE (15 bins, document-bootstrap interval) and smooth ECE;
* log-loss and Brier of correctness, document-first with pool weights, each paired
  with ``none`` (gain > 0: the method beats the raw confidence) and with ``context``;
* the committed-token gap left after recalibration, E_m[pi_hat - a] over the T8 pool
  winners (pools still won by the raw ell: the decoder's own choice), with
  Delta_hat(32), the E1-style contrast Delta_hat(32) - Delta_hat(1), and Delta_hat(32)
  paired with the raw gap.

A method "is enough" where its remaining gap is indistinguishable from 0 at every m.

``--save-calibrators DIR`` also fits every method on the fit documents of the
``--save-arms`` (default the teacher-forced decoder arms, pooled, as
``analyze_setcal.py`` fits ``tok.json``) and writes ``DIR/<method>/tok.json`` for
``eval_infill.py --calibrator ... --cal-tag <method>``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3 import N_BOOT, N_POOLS, POOL_SIZES
from c3 import recal
from c3.assemble import position_and_pool_frames, read_arm, trace_doc_ids
from c3.calibrators import pool_weights
from c3.endpoints.e1 import winner_gap
from c3.endpoints.fieldtest import EPS, paired_score_test
from c3.frame import PositionFrame
from c3.metrics import ece_equal_mass, smooth_ece
from c3.provenance import collect as provenance
from c3.stats import boot_pvalue, doc_boot_counts, percentile_ci
from c3.util import require_report_ok

HI, LO = max(POOL_SIZES), min(POOL_SIZES)
PRIMARY_ARMS = ("uniform", "decoder:topk_conf:k4")
SAVE_ARMS = ("decoder:topk_conf:k2", "decoder:topk_conf:k4", "decoder:topk_conf:k8")
COLUMNS = ["doc_id", "state_id", "position", "ell", "conf", "agree", "mask_kind",
           "mask_rate", "n_masked", "step", "policy", "k"]


def _est(point, draws) -> dict:
    return {"value": float(point), "ci": list(percentile_ci(draws)), "p_boot": boot_pvalue(draws)}


def score_arm(df, split: dict, methods, codes, counts, a) -> dict:
    """Every method fitted on the arm's fit positions and scored on its scored ones."""
    pos, pools = position_and_pool_frames(df, pool_sizes=POOL_SIZES, n_pools_per_size=N_POOLS,
                                          seed=a.seed, retain=1, min_state_size=max(POOL_SIZES))
    half = np.array([split[int(d)] for d in pos.doc_id], dtype=np.int64)
    fit_mask, rep_mask = half == 0, half == 1
    tr, pr = pos.select(fit_mask), pos.select(rep_mask)
    w_fit, w_rep = pool_weights(tr), pool_weights(pr)
    doc_idx = np.searchsorted(codes, pr.doc_id)
    common = dict(agree=pr.agree, doc_idx=doc_idx, n_docs=codes.size, counts=counts,
                  weights=w_rep, eps=a.eps)

    preds, fits = {}, {}
    for m in methods:
        t0 = time.time()
        cal = recal.make(m).fit(tr, w_fit)
        preds[m] = np.clip(cal.predict(pos), 0.0, 1.0)
        d = cal.to_dict()
        fits[m] = {k: v for k, v in d.items() if k in ("kind", "T", "features", "converged")}
        fits[m]["seconds"] = round(time.time() - t0, 1)

    raw_gap = winner_gap(pools, pos, pos.conf, pos.agree, codes, counts, rep_mask)
    i_hi = int(np.flatnonzero(raw_gap.levels == HI)[0])
    out = {"n_fit_positions": int(len(tr)), "n_scored_positions": int(len(pr)),
           "raw_gap": raw_gap.to_records("gap"), "methods": {}}
    for m in methods:
        p = preds[m][rep_mask]
        ece = ece_equal_mass(p, pr.agree, n_bins=15, doc_idx=doc_idx, counts=counts)
        sece, _ = smooth_ece(p, pr.agree)
        g = winner_gap(pools, pos, preds[m], pos.agree, codes, counts, rep_mask)
        rec = {"fit": fits[m],
               "ece": {"value": ece.ece, "ci": list(ece.ci)}, "smooth_ece": float(sece),
               "gap": g.to_records("gap"),
               "gap_hi": _est(g.point[i_hi], g.draws[:, i_hi]),
               "gap_contrast": _est(*g.contrast(HI, LO)),
               "gap_hi_vs_raw": _est(g.point[i_hi] - raw_gap.point[i_hi],
                                     g.draws[:, i_hi] - raw_gap.draws[:, i_hi])}
        for ref in ("none", "context"):
            if ref in preds and ref != m:
                rec[f"vs_{ref}"] = {
                    metric: paired_score_test(preds[ref][rep_mask], p, metric=metric, **common)
                    for metric in ("log_loss", "brier")}
        out["methods"][m] = rec
    return out


def save_calibrators(a, split: dict, methods) -> dict:
    """Every method fitted on the fit documents of the pooled ``--save-arms``."""
    import pandas as pd

    fit_docs = sorted(d for d, h in split.items() if h == 0)
    df = pd.concat([read_arm(a.traces, arm, COLUMNS, doc_ids=fit_docs) for arm in a.save_arms],
                   ignore_index=True)
    pos = PositionFrame.from_frame(df)
    w = pool_weights(pos)
    out = {}
    for m in methods:
        cal = recal.make(m).fit(pos, w)
        d = Path(a.save_calibrators) / m
        d.mkdir(parents=True, exist_ok=True)
        recal.save(cal, d / "tok.json")
        out[m] = str(d / "tok.json")
    print(f"[recal] {len(methods)} calibrators on {len(pos):,} positions of {a.save_arms} "
          f"-> {a.save_calibrators}", flush=True)
    return {"arms": list(a.save_arms), "n_positions": int(len(pos)), "n_docs": len(fit_docs),
            "paths": out}


def latex_table(res: dict) -> str:
    def tex(s: str) -> str:
        return s.replace("_", "\\_")

    def cell(v: float) -> str:
        return f"${v:+.4f}$"

    arms = list(res["arms"])
    lines = ["% Generated by scripts/compare_recalibration.py -- do not edit by hand.",
             "\\begin{tabular}{l" + "rrr" * len(arms) + "}", "\\toprule",
             "Method & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{tex(x)}}}" for x in arms) + " \\\\",
             " & " + " & ".join(["ECE & $\\hat\\Delta(32)$ & $\\hat\\Delta(32)-\\hat\\Delta(1)$"]
                               * len(arms)) + " \\\\", "\\midrule"]
    for m in res["methods"]:
        cells = []
        for arm in arms:
            r = res["arms"][arm]["methods"][m]
            cells += [f"${r['ece']['value']:.4f}$", cell(r["gap_hi"]["value"]),
                      cell(r["gap_contrast"]["value"])]
        lines.append(tex(m) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="parquet trace or *.dataset.json")
    ap.add_argument("--manifest", required=True, help="{doc_id: 0 fit | 1 score}")
    ap.add_argument("--arms", nargs="+", default=list(PRIMARY_ARMS))
    ap.add_argument("--methods", nargs="+", choices=list(recal.METHODS), default=list(recal.METHODS))
    ap.add_argument("--save-calibrators", default=None,
                    help="also write <dir>/<method>/tok.json fitted on the --save-arms")
    ap.add_argument("--save-arms", nargs="+", default=list(SAVE_ARMS))
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot-label", default="t13-recal")
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
    print(f"[recal] {docs.size} documents: {codes.size} scored; methods {a.methods}; "
          f"B = {a.n_boot}", flush=True)

    res = {"label": a.label, "input": {"traces": a.traces, "manifest": a.manifest},
           "config": {k: v for k, v in vars(a).items() if k not in ("traces", "manifest", "out")},
           "n_docs": int(docs.size), "n_scored_docs": int(codes.size),
           "methods": list(a.methods), "arms": {}}
    for arm in a.arms:
        t0 = time.time()
        df = read_arm(a.traces, arm, COLUMNS).reset_index(drop=True)
        res["arms"][arm] = r = score_arm(df, split, a.methods, codes, counts, a)
        for m in a.methods:
            x = r["methods"][m]
            print(f"[recal] {arm:24s} {m:16s} ECE {x['ece']['value']:.4f}  "
                  f"gap(32) {x['gap_hi']['value']:+.4f} [{x['gap_hi']['ci'][0]:+.4f}, "
                  f"{x['gap_hi']['ci'][1]:+.4f}]  contrast {x['gap_contrast']['value']:+.4f}",
                  flush=True)
        print(f"[recal] {arm}: {r['n_scored_positions']:,} scored positions "
              f"({time.time() - t0:.0f} s)", flush=True)
    if a.save_calibrators:
        res["saved_calibrators"] = save_calibrators(a, split, a.methods)
    res["provenance"] = provenance()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    out.with_suffix(".tex").write_text(latex_table(res))
    print(f"[recal] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
