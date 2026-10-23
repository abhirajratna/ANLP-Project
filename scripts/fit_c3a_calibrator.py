#!/usr/bin/env python3
"""T11: fit and freeze the C3a calibrator on the fit half of a task trace.

    python scripts/fit_c3a_calibrator.py \\
        --traces /share1/abhiraj.ratna/traces/c1-llada-instruct-gsm8k/c1-llada-instruct-gsm8k.dataset.json \\
        --manifest artifacts/split_manifest_gsm8k.json \\
        --out artifacts/c3/llada-8b-instruct_fp16_c3a-gsm8k.json

The calibrator is ``c3.calibrators.ContextLogistic`` on the decode-time field features
(``DECODE_FEATURES``: median ell, mask rate, log |M|), fitted with pool weights on the
teacher-forced decoder states (``decoder:topk_conf:k2/k4/k8``, pooled) of the fit-half
documents (manifest value 0).  Only those rows are read: the filters are applied while
reading, so no report-half row is ever loaded.

Before the final fit, a development check re-splits the fit half (``hash_halves(., 1)``,
as every other development analysis): fit on dev-fit, score on dev-eval, against the
raw confidence sigma(ell).  It is a sanity check, not a reported number.  The final
calibrator is then fitted on every fit-half document and written, with its inputs'
digests and the provenance stamp, to ``--out`` for ``C3a(alpha, ContextLogistic.load(out))``.

Two diagnostics travel with it.  ``feature_ranges`` (inside the calibrator) is the
fit-half range of each field feature over the decoder states, so C3a can count, or clamp,
the decode steps that fall outside it: free-running generation starts from 256 masks,
while a fit state's region is the reference answer (median ~90 tokens on GSM8K).
``feature_quantiles`` gives the 5/50/95th percentiles for reporting that shift.
``monotone_share`` is the share of fit states in which c_hat never decreases with ell; C3a
ranks by ell and uses c_hat only for the count, so non-monotone states change the count's
meaning, not the order.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3 import DEFAULT_KNOTS
from c3.assemble import read_trace_table, trace_meta
from c3.calibrators import DECODE_FEATURES, ContextLogistic, pool_weights
from c3.frame import PositionFrame, hash_halves
from c3.metrics import ece_equal_mass
from c3.provenance import collect as provenance, sha256_file
from c3.stats import doc_boot_counts, grouped_doc_mean, percentile_ci

ARMS = ("decoder:topk_conf:k2", "decoder:topk_conf:k4", "decoder:topk_conf:k8")
COLUMNS = ["doc_id", "state_id", "position", "ell", "conf", "agree", "mask_kind", "policy",
           "k", "mask_rate", "n_masked", "step"]
EPS = 1e-6


def arm_filter(arm: str) -> list:
    kind, policy, k = arm.split(":")
    return [("mask_kind", "=", kind), ("policy", "=", policy), ("k", "=", int(k.lstrip("k")))]


def load_fit_rows(path: str, arms, fit_docs) -> "PositionFrame":
    import pandas as pd

    ids = [int(d) for d in fit_docs]
    parts = [read_trace_table(path, columns=COLUMNS,
                              filters=arm_filter(arm) + [("doc_id", "in", ids)]).to_pandas()
             for arm in arms]
    df = pd.concat(parts, ignore_index=True)
    if not len(df):
        raise SystemExit(f"no fit-half rows for {list(arms)} in {path}")
    leaked = set(df["doc_id"].unique().tolist()) - set(ids)
    if leaked:                                       # the read filter is the guard; check it
        raise SystemExit(f"{len(leaked)} non-fit documents read; refusing")
    return PositionFrame.from_frame(df)


def scores(p: np.ndarray, pos: PositionFrame, counts: np.ndarray, codes: np.ndarray) -> dict:
    """Document-first log-loss and Brier (pool weights) and equal-mass ECE, with intervals."""
    doc_idx = np.searchsorted(codes, pos.doc_id)
    w = pool_weights(pos)
    q = np.clip(p, EPS, 1 - EPS)
    zero = np.zeros(len(pos), np.int64)
    out = {}
    for name, loss in (("log_loss", -(pos.agree * np.log(q) + (1 - pos.agree) * np.log1p(-q))),
                       ("brier", (p - pos.agree) ** 2)):
        g = grouped_doc_mean(loss, doc_idx, codes.size, zero, counts, weights=w)
        out[name] = {"value": float(g.point[0]), "draws": g.draws[:, 0]}
    e = ece_equal_mass(p, pos.agree, n_bins=15, doc_idx=doc_idx, counts=counts)
    out["ece"] = {"value": e.ece, "ci": list(e.ci)}
    return out


def state_features(pos: PositionFrame, names) -> dict:
    """Each field feature once per state (states are the rows' unit of the fit)."""
    from c3.calibrators import _features_exact

    X = _features_exact(pos, names)
    _, first = np.unique(pos.state_id, return_index=True)
    return {n: X[first, j] for j, n in enumerate(names)}


def feature_summary(pos: PositionFrame, names) -> tuple[dict, dict]:
    """(fit-half range per feature, its 5/50/95th percentiles), over states."""
    per_state = state_features(pos, names)
    ranges = {n: {"min": float(v.min()), "max": float(v.max())} for n, v in per_state.items()}
    quants = {n: dict(zip(("p05", "p50", "p95"), map(float, np.percentile(v, [5, 50, 95]))))
              for n, v in per_state.items()}
    return ranges, quants


def monotone_share(cal: ContextLogistic, pos: PositionFrame) -> float:
    """Share of states in which c_hat is non-decreasing in ell."""
    c = cal.predict(pos)
    order = np.lexsort((pos.ell, pos.state_id))
    s, cc = pos.state_id[order], c[order]
    drop = (np.diff(cc) < -1e-12) & (s[1:] == s[:-1])
    bad = np.unique(s[1:][drop])
    return float(1.0 - bad.size / np.unique(s).size)


def dev_check(pos: PositionFrame, features, knots: int, n_boot: int, seed: int) -> dict:
    """Fit on dev-fit, score on dev-eval: calibrated against raw sigma(ell)."""
    docs = np.unique(pos.doc_id)
    half = hash_halves(docs, 1)
    dev_fit, dev_eval = docs[half == 0], docs[half == 1]
    tr = pos.select(np.isin(pos.doc_id, dev_fit))
    te = pos.select(np.isin(pos.doc_id, dev_eval))
    cal = ContextLogistic(features=tuple(features), n_knots=knots).fit(tr, pool_weights(tr))
    codes = np.sort(dev_eval)
    counts = doc_boot_counts(codes.size, n_boot, seed, label="t11-c3a-dev")
    raw = scores(1.0 / (1.0 + np.exp(-te.ell)), te, counts, codes)
    fit = scores(cal.predict(te), te, counts, codes)
    out = {"n_dev_fit_docs": int(dev_fit.size), "n_dev_eval_docs": int(dev_eval.size),
           "features_used": list(cal.used), "converged": cal.converged}
    for m in ("log_loss", "brier"):
        gain = raw[m]["draws"] - fit[m]["draws"]
        out[m] = {"raw": raw[m]["value"], "calibrated": fit[m]["value"],
                  "gain": raw[m]["value"] - fit[m]["value"], "gain_ci": list(percentile_ci(gain))}
    out["ece"] = {"raw": raw["ece"], "calibrated": fit["ece"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="*.dataset.json or parquet of one task run")
    ap.add_argument("--manifest", required=True, help="the committed task split (0 = fit half)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--features", nargs="*", default=list(DECODE_FEATURES))
    ap.add_argument("--knots", type=int, default=DEFAULT_KNOTS)
    ap.add_argument("--n-boot", type=int, default=2000, help="for the development check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    split = {int(k): int(v) for k, v in json.loads(Path(a.manifest).read_text()).items()}
    fit_docs = sorted(d for d, h in split.items() if h == 0)
    pos = load_fit_rows(a.traces, a.arms, fit_docs)
    n_docs = int(np.unique(pos.doc_id).size)
    print(f"[c3a] {len(pos):,} decoder-state positions from {n_docs} fit-half documents "
          f"({', '.join(a.arms)})", flush=True)

    check = dev_check(pos, a.features, a.knots, a.n_boot, a.seed)
    print(f"[c3a] dev check: log-loss {check['log_loss']['raw']:.4f} -> "
          f"{check['log_loss']['calibrated']:.4f}, ECE {check['ece']['raw']['value']:.4f} -> "
          f"{check['ece']['calibrated']['value']:.4f}", flush=True)

    cal = ContextLogistic(features=tuple(a.features), n_knots=a.knots).fit(pos, pool_weights(pos))
    if not cal.converged:
        print("[c3a] WARNING: the final fit did not converge", flush=True)
    cal.feature_ranges, quantiles = feature_summary(pos, cal.used)
    mono = monotone_share(cal, pos)
    print(f"[c3a] c_hat monotone in ell in {mono:.1%} of fit states; feature ranges "
          f"{json.dumps(cal.feature_ranges)}", flush=True)
    meta = trace_meta(a.traces)
    rec = {
        "what": "C3a calibrator (plan v2 T11), frozen: pi_hat(ell, s) on the decode-time field "
                "features, fitted on the fit half only",
        "calibrator": cal.to_dict(),
        "features_requested": list(a.features), "arms": list(a.arms),
        "weights": "pool (calibrators.pool_weights)",
        "n_fit_docs": n_docs, "n_fit_positions": int(len(pos)),
        "n_fit_states": int(np.unique(pos.state_id).size),
        "feature_quantiles": quantiles, "monotone_share": mono,
        "traces": {"path": a.traces, "sha256": sha256_file(a.traces),
                   **{k: meta.get(k) for k in ("model", "variant", "precision", "dataset",
                                               "git_commit", "revision") if k in meta}},
        "split": {"path": a.manifest, "sha256": sha256_file(a.manifest), "fit_half": 0},
        "dev_check": check,
        "provenance": provenance(),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=float))
    print(f"[c3a] calibrator ({', '.join(cal.used)}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
