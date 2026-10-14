#!/usr/bin/env python3
"""A: set-level calibration (``c3.setcal``) on decoder states, fitted on one half, scored on the other.

    python scripts/analyze_setcal.py --traces runs/dev/main-110m-v4.fit.parquet \\
        --manifest runs/dev/main-110m-v4.fit.manifest.json --out runs/dev/setcal.json \\
        --save-calibrators runs/cal/dev-traces

Two sources of states:

* ``--traces``: the teacher-forced decoder arms of a trace (default
  decoder:topk_conf:k2/k4/k8);
* ``--onpolicy``: free-running step logs from ``eval_infill.py --record-steps``
  (every masked position at every step, with ``agree`` = top-1 equals the
  original token).

The manifest maps documents to 0 (fit) and 1 (score).  On the fit documents the
token calibrator pi_hat(ell, s) is fitted (pool weights), its out-of-fold c_hat
(two folds of the fit documents) builds the prefix-set table, and the
``SetCalibrator`` is fitted on that.  On the scored documents three estimates of
P(every token of S_j right) are compared, paired per document with the document
bootstrap: the set model, the independence product of c_hat (what C3-indep
assumes), and the product of raw sigma(ell).  ``--save-calibrators`` writes
``tok.json`` (fitted on every fit document) and ``set.json`` for the decoder;
``--use-calibrators`` scores given ones instead, so calibrators from two sources
(traces, on-policy logs) can be compared on the same free-running states.
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
from c3.endpoints.fieldtest import DECODE_FEATURES, EPS, paired_score_test
from c3.metrics import ece_equal_mass
from c3.provenance import collect as provenance
from c3.setcal import SetCalibrator, build_set_table, crossfit_token_probs
from c3.stats import doc_boot_counts, grouped_doc_mean
from c3.util import require_report_ok

DEFAULT_ARMS = ("decoder:topk_conf:k2", "decoder:topk_conf:k4", "decoder:topk_conf:k8")
COLUMNS = ["doc_id", "state_id", "position", "ell", "conf", "entropy", "agree", "mask_kind",
           "mask_rate", "n_masked", "step", "policy", "k"]
J_EDGES = (1, 2, 3, 5, 9, 17, 33, 129)        # prefix-size bins [1] [2] [3,4] ... [33,128]


def j_bin(j: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(J_EDGES), np.asarray(j), side="right") - 1


def load_states(a, split: dict):
    import pandas as pd

    frames = []
    if a.traces:
        for arm in a.arms:
            t0 = time.time()
            df = read_arm(a.traces, arm, COLUMNS, doc_ids=sorted(split))
            df["arm"] = arm
            frames.append(df)
            print(f"[setcal] {arm}: {len(df):,} rows ({time.time() - t0:.0f} s)", flush=True)
    else:
        paths = []
        for src in a.onpolicy:          # files, or directories of step-log parts
            paths += sorted(Path(src).glob("*.parquet")) if Path(src).is_dir() else [Path(src)]
        for path in paths:
            df = pd.read_parquet(path)
            df = df[df["doc_id"].isin(list(split))]
            if "arm" not in df.columns:
                df["arm"] = Path(path).stem
            frames.append(df)
        print(f"[setcal] {len(paths)} step-log parts, {sum(map(len, frames)):,} rows", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["arm"] = df["arm"].astype(str)
    df["half"] = df["doc_id"].map(split).astype(np.int64)
    return df


def token_probs(df, a):
    """c_hat for every row: out of fold on fit documents, the full fit elsewhere."""
    try:
        return crossfit_token_probs(df, df["half"].to_numpy() == 0, features=tuple(a.token_features),
                                    n_knots=a.knots, ridge=a.ridge, interactions=not a.additive)
    except ValueError as exc:
        raise SystemExit(f"{exc}; add documents")


def score_block(t, p: dict, codes, counts, a) -> dict:
    """Paired scores, ECE and the gap by prefix size, on one block of scored prefix rows."""
    doc_idx = np.searchsorted(codes, t["doc_id"].to_numpy())
    y = t["y"].to_numpy(np.float64)
    w = t["weight"].to_numpy(np.float64)
    out = {"n_rows": int(len(t)), "n_states": int(t["state_id"].nunique()),
           "all_right_rate": float(np.average(y, weights=w)), "compare": {}, "ece": {}, "gap_by_j": {}}
    for base, alt in (("indep", "set"), ("raw", "set"), ("raw", "indep")):
        out["compare"][f"{alt}_vs_{base}"] = {
            m: paired_score_test(p[base], p[alt], y, doc_idx, codes.size, counts, weights=w,
                                 metric=m, eps=a.eps,
                                 floor=a.floor_ll if m == "log_loss" else a.floor_brier)
            for m in ("log_loss", "brier")}
    jb = j_bin(t["j"].to_numpy())
    labels = [f"{J_EDGES[i]}-{J_EDGES[i + 1] - 1}" for i in range(len(J_EDGES) - 1)]
    for name, q in p.items():
        out["ece"][name] = ece_equal_mass(q, y, n_bins=15, doc_idx=doc_idx, counts=counts).to_dict()
        g = grouped_doc_mean(q - y, doc_idx, codes.size, jb, counts, weights=w)
        out["gap_by_j"][name] = [dict(r, j=labels[int(r["level"])]) for r in g.to_records("gap")]
    g = grouped_doc_mean(y, doc_idx, codes.size, jb, counts, weights=w)
    out["all_right_by_j"] = [dict(r, j=labels[int(r["level"])]) for r in g.to_records("rate")]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--traces", help="parquet trace or *.dataset.json")
    src.add_argument("--onpolicy", nargs="+", help="step logs from eval_infill.py --record-steps (files or directories)")
    ap.add_argument("--manifest", required=True, help="{doc_id: 0 fit | 1 score}")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    ap.add_argument("--token-features", nargs="*", default=list(DECODE_FEATURES),
                    help="field features of the token calibrator (none: spline only); the "
                         "deviations.md item-8 gate decides")
    ap.add_argument("--no-set-field", action="store_true",
                    help="the set model without the state's field features")
    ap.add_argument("--knots", type=int, default=DEFAULT_KNOTS)
    ap.add_argument("--set-knots", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--additive", action="store_true")
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot-label", default="c3set-setcal")
    ap.add_argument("--floor-ll", type=float, default=0.0)
    ap.add_argument("--floor-brier", type=float, default=0.0)
    ap.add_argument("--fit-only", action="store_true",
                    help="fit and save the calibrators on every manifest-0 document; no scoring")
    ap.add_argument("--save-calibrators", default=None, help="directory for tok.json / set.json")
    ap.add_argument("--use-calibrators", default=None,
                    help="score the tok.json / set.json in this directory instead of fitting "
                         "(compares calibrator sources on the same scored states)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    split = {int(k): int(v) for k, v in json.loads(Path(a.manifest).read_text()).items()}
    require_report_ok(a.manifest, reads_report=not a.fit_only)
    if a.fit_only:
        split = {d: h for d, h in split.items() if h == 0}
    if a.traces:
        present = set(trace_doc_ids(a.traces).tolist())
        split = {d: h for d, h in split.items() if d in present}
    df = load_states(a, split)
    t0 = time.time()
    given = Path(a.use_calibrators) if a.use_calibrators else None
    if given:
        from c3.calibrators import ContextLogistic
        from c3.frame import PositionFrame

        tok = ContextLogistic.load(given / "tok.json")
        c_hat = tok.predict(PositionFrame.from_frame(df))
        print(f"[setcal] scoring the calibrators in {given} (no fitting)", flush=True)
    else:
        c_hat, tok = token_probs(df, a)
        print(f"[setcal] token calibrator {tok.used} on {int((df['half'] == 0).sum()):,} fit rows "
              f"({time.time() - t0:.0f} s)", flush=True)

    tables = []
    for arm, rows in df.groupby("arm", sort=True).groups.items():
        rows = np.asarray(rows)
        t = build_set_table(df.iloc[rows], c_hat[rows])
        t["arm"] = arm
        tables.append(t)
    import pandas as pd

    table = pd.concat(tables, ignore_index=True)
    table["half"] = table["doc_id"].map(split).astype(np.int64)
    fit = table[table["half"] == 0]
    if given:
        setcal = SetCalibrator.load(given / "set.json")
    else:
        setcal = SetCalibrator(use_field=not a.no_set_field, n_knots=a.set_knots,
                               ridge=a.ridge).fit(fit, fit["weight"].to_numpy())
        print(f"[setcal] set calibrator on {len(fit):,} prefix rows; converged={setcal.converged}; "
              f"coef {json.dumps({k: round(v, 3) for k, v in setcal.coef().items() if not k.startswith('f_u')})}",
              flush=True)

    res = {"label": a.label, "source": "traces" if a.traces else "onpolicy",
           "input": {"traces": a.traces, "onpolicy": a.onpolicy, "manifest": a.manifest},
           "config": {k: v for k, v in vars(a).items()
                      if k not in ("traces", "onpolicy", "manifest", "out")},
           "token_calibrator": tok.to_dict(), "set_calibrator": setcal.to_dict(),
           "set_coef": setcal.coef(), "n_fit_rows": int(len(fit)), "arms": {}}
    if given:
        res["calibrator_dir"] = str(given)
    elif a.save_calibrators:
        d = Path(a.save_calibrators)
        d.mkdir(parents=True, exist_ok=True)
        tok.save(d / "tok.json")
        setcal.save(d / "set.json")
        res["calibrator_dir"] = str(d)
        print(f"[setcal] calibrators -> {d}", flush=True)

    if not a.fit_only:
        rep = table[table["half"] == 1]
        codes = np.unique(rep["doc_id"].to_numpy())
        counts = doc_boot_counts(codes.size, a.n_boot, a.seed, label=a.boot_label)
        p_all = {"set": setcal.predict(rep), "indep": np.exp(rep["log_p_ind"].to_numpy()),
                 "raw": np.exp(rep["log_p_raw"].to_numpy())}
        arm_col = rep["arm"].to_numpy()
        blocks = {arm: arm_col == arm for arm in sorted(set(arm_col))}
        blocks["pooled"] = np.ones(len(rep), dtype=bool)
        for arm, m in blocks.items():
            t0 = time.time()
            res["arms"][arm] = score_block(rep[m], {k: v[m] for k, v in p_all.items()}, codes,
                                           counts, a)
            ll = res["arms"][arm]["compare"]["set_vs_indep"]["log_loss"]
            print(f"[setcal] {arm}: set vs independence log-loss gain {ll['gain']:+.5f} "
                  f"[{ll['ci'][0]:+.5f}, {ll['ci'][1]:+.5f}] ({ll['rel_gain']:+.2%}), "
                  f"helps={ll['helps']} ({time.time() - t0:.0f} s)", flush=True)
        res["n_scored_docs"] = int(codes.size)
    res["provenance"] = provenance()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"[setcal] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
