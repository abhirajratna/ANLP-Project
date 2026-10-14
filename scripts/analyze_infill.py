#!/usr/bin/env python3
"""Infilling accuracy at matched speed: does set-level C3 beat the best decoder?

    python scripts/analyze_infill.py --runs runs/infill/dev --out runs/infill/dev/analysis.json \\
        --plot runs/infill/dev/curves.png

Reads every ``infill.shard*.jsonl`` under ``--runs`` (``eval_infill.py``); where two runs
hold the same (configuration, document), the later ``--runs`` entry wins (e.g. the
baselines of sweep 1 with C3 policies rerun under other calibrators), and differing
outputs are counted in ``duplicate_conflicts``.  For each
configuration: document-first mean accuracy and per-step all-right rate, and
tokens/step = tokens committed / forward passes (the T12 ratio).  All of them
are recomputed on one shared document-bootstrap matrix.

Each policy's configurations trace a curve of accuracy against log2(tokens/step);
accuracy at the speeds ``--speeds`` is read off by linear interpolation
(``c3.infill.interp_at``), NaN where a policy never reached that speed.  The
primary statistic at speed t is

    Delta_max(t) = acc_new(t) - max over baselines b of acc_b(t),

with the max taken inside every bootstrap draw, so the interval accounts for
picking the best baseline after the fact.  Holm over the speeds; an improvement
is declared when the Holm-adjusted p <= 0.05, the 95% interval excludes 0, the
gain exceeds 2 x ``--floor`` and is positive.  Secondary contrasts (no
multiplicity correction) isolate the parts: C3a - Fast-dLLM threshold (the field
effect alone), C3-set - C3-indep (the learned dependence), C3-set - EB-Sampler
and C3-set - C3a (set vs token targets).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3 import N_BOOT
from c3.infill import interp_at, interp_draws
from c3.provenance import collect as provenance
from c3.stats import boot_pvalue, declare, doc_boot_counts, holm, percentile_ci

BASELINES = ("topk_conf", "margin", "pc_sampler", "threshold", "entropy_bounded", "klass")
SECONDARY = (("c3a", "threshold"), ("c3set", "c3indep"), ("c3set", "entropy_bounded"),
             ("c3set", "c3a"), ("c3indep", "threshold"))
KNOBS = ("k", "tau", "gamma", "eps_kl", "alpha", "lam")


def load_records(dirs):
    import pandas as pd

    rows = []
    for d in dirs:
        for p in sorted(Path(d).glob("infill.shard*.jsonl")):
            for line in p.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        raise SystemExit(f"no infill records under {dirs}")
    df = pd.DataFrame(rows)
    dup = df.duplicated(["config", "doc_id"], keep=False)
    n_conflict = 0
    if dup.any():
        n_conflict = int((df[dup].groupby(["config", "doc_id"])["tokens_sha1"].nunique() > 1).sum())
        df = df.drop_duplicates(["config", "doc_id"], keep="last")
    return df.reset_index(drop=True), n_conflict


def summarise(x: float, draws: np.ndarray, alpha: float = 0.05) -> dict:
    lo, hi = percentile_ci(draws, alpha)
    return {"value": float(x), "ci": [lo, hi], "se": float(np.nanstd(draws, ddof=1)),
            "p_boot": boot_pvalue(draws)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--speeds", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    ap.add_argument("--new", default="c3set")
    ap.add_argument("--baselines", nargs="+", default=list(BASELINES))
    ap.add_argument("--floor", type=float, default=0.0,
                    help="accuracy non-determinism floor (0 when repeated runs are identical)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="keep only documents every configuration finished")
    ap.add_argument("--compare-to", nargs="*", default=None,
                    help="infill dirs of a repeated run: report the share of identical outputs")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot-label", default="c3set-infill")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df, n_conflict = load_records(a.runs)
    configs = sorted(df["config"].unique())
    docs_by_cfg = df.groupby("config")["doc_id"].apply(set)
    common = set.intersection(*docs_by_cfg.tolist())
    every = set(df["doc_id"])
    if common != every and not a.allow_incomplete:
        short = {c: len(every - s) for c, s in docs_by_cfg.items() if s != every}
        raise SystemExit(f"incomplete: configurations missing documents {short}; "
                         f"rerun or pass --allow-incomplete")
    docs = np.array(sorted(common), dtype=np.int64)
    df = df[df["doc_id"].isin(common)]
    D, C = docs.size, len(configs)
    counts = doc_boot_counts(D, a.n_boot, a.seed, label=a.boot_label).astype(np.float64)

    def mat(col):
        p = df.pivot(index="config", columns="doc_id", values=col).loc[configs, docs]
        return p.to_numpy(np.float64)

    A, F, N, S = mat("acc"), mat("n_forward"), mat("n_committed"), mat("step_all_right")
    acc_pt, sar_pt = A.mean(1), S.mean(1)
    tps_pt = N.sum(1) / F.sum(1)
    acc_dr, sar_dr = counts @ A.T / D, counts @ S.T / D
    tps_dr = (counts @ N.T) / (counts @ F.T)
    first = df.drop_duplicates("config").set_index("config").loc[configs]
    policy_of = first["policy"].to_numpy()
    secs = df.groupby("config")["seconds"].mean().loc[configs].to_numpy()
    has_ppl = "gpt2_nll" in df.columns and df["gpt2_nll"].notna().all()
    ppl = df.groupby("config")["gpt2_nll"].mean().loc[configs].to_numpy() if has_ppl else None

    curves = []
    for i, c in enumerate(configs):
        row = {"config": c, "policy": policy_of[i],
               **{k: (None if k not in first.columns or first[k].iloc[i] != first[k].iloc[i]
                      else float(first[k].iloc[i])) for k in KNOBS},
               "tokens_per_step": float(tps_pt[i]), "acc": float(acc_pt[i]),
               "acc_ci": list(percentile_ci(acc_dr[:, i])), "step_all_right": float(sar_pt[i]),
               "step_all_right_ci": list(percentile_ci(sar_dr[:, i])), "seconds_per_doc": float(secs[i])}
        if has_ppl:
            row["gpt2_nll"] = float(ppl[i])
        curves.append(row)

    policies = sorted(set(policy_of))
    at, at_draws = {}, {}
    for p in policies:
        idx = np.flatnonzero(policy_of == p)
        at[p], at_draws[p] = {}, {}
        for t in a.speeds:
            pt = interp_at(tps_pt[idx], acc_pt[idx], t)
            dr = interp_draws(tps_dr[:, idx], acc_dr[:, idx], t)
            sp = interp_at(tps_pt[idx], sar_pt[idx], t)
            at_draws[p][t] = dr
            at[p][str(t)] = {"acc": pt, "acc_ci": list(percentile_ci(dr)), "step_all_right": sp,
                             "tps_range": [float(tps_pt[idx].min()), float(tps_pt[idx].max())]}

    base = [b for b in a.baselines if b in at_draws]
    primary, pvals = {}, []
    for t in a.speeds:
        if a.new not in at_draws or not base:
            break
        B = np.column_stack([at_draws[b][t] for b in base])
        bpt = np.array([at[b][str(t)]["acc"] for b in base])
        with np.errstate(all="ignore"), _quiet():
            best_dr = np.nanmax(B, axis=1)
        best_pt = np.nanmax(bpt) if np.isfinite(bpt).any() else float("nan")
        best_name = base[int(np.nanargmax(bpt))] if np.isfinite(bpt).any() else None
        new_pt = at[a.new][str(t)]["acc"]
        rec = summarise(new_pt - best_pt, at_draws[a.new][t] - best_dr)
        rec.update({"new": a.new, "new_acc": new_pt, "best_baseline": best_name,
                    "best_baseline_acc": float(best_pt),
                    "mde": 2.8 * rec["se"] if np.isfinite(rec["se"]) else float("nan"),
                    "speedup_at_matched_accuracy": _speedup(tps_pt, acc_pt, policy_of, a.new,
                                                            best_pt, t)})
        primary[str(t)] = rec
        pvals.append(rec["p_boot"])
    finite = [k for k, r in primary.items() if np.isfinite(r["value"]) and np.isfinite(r["p_boot"])]
    if finite:
        adj, rej = holm([primary[k]["p_boot"] for k in finite])
        for k, p, r in zip(finite, adj, rej):
            rec = primary[k]
            rec["p_holm"] = float(p)
            rec["declared"] = bool(r and declare(rec["value"], tuple(rec["ci"]), a.floor))
            rec["improves"] = bool(rec["declared"] and rec["value"] > 0)
    for k, rec in primary.items():
        rec.setdefault("p_holm", float("nan"))
        rec.setdefault("declared", False)
        rec.setdefault("improves", False)

    secondary = {}
    for x, y in SECONDARY:
        if x in at_draws and y in at_draws:
            secondary[f"{x}-{y}"] = {
                str(t): summarise(at[x][str(t)]["acc"] - at[y][str(t)]["acc"],
                                  at_draws[x][t] - at_draws[y][t]) for t in a.speeds}

    coverage = []
    for i, c in enumerate(configs):
        if policy_of[i] in ("c3a", "c3indep", "c3set"):
            target = 1.0 - float(first["alpha"].iloc[i])
            got = float(acc_pt[i]) if policy_of[i] == "c3a" else float(sar_pt[i])
            coverage.append({"config": c, "policy": policy_of[i], "target": target,
                             "quantity": "token accuracy" if policy_of[i] == "c3a"
                             else "per-step all-right rate", "achieved": got,
                             "tokens_per_step": float(tps_pt[i])})

    repeat = None
    if a.compare_to:
        other, _ = load_records(a.compare_to)
        m = df.merge(other, on=["config", "doc_id"], suffixes=("", "_rep"))
        repeat = {"n_pairs": int(len(m)),
                  "identical_share": float((m["tokens_sha1"] == m["tokens_sha1_rep"]).mean()),
                  "max_abs_acc_change": float((m["acc"] - m["acc_rep"]).abs().max())}

    res = {"label": a.label, "runs": a.runs, "n_docs": int(D), "n_configs": int(C),
           "speeds": a.speeds, "new": a.new, "baselines": base, "floor": a.floor,
           "duplicate_conflicts": n_conflict, "curves": curves, "at_speed": at,
           "primary": primary, "secondary": secondary, "coverage": coverage,
           "repeat_check": repeat, "config": {"n_boot": a.n_boot, "seed": a.seed,
                                              "boot_label": a.boot_label},
           "provenance": provenance()}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    for t, r in primary.items():
        print(f"[infill] t={t}: {a.new} {r['new_acc']:.4f} vs best {r['best_baseline']} "
              f"{r['best_baseline_acc']:.4f}: Delta_max {r['value']:+.4f} "
              f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] p_holm={r['p_holm']:.3g} "
              f"improves={r['improves']}")
    if a.plot:
        plot_curves(curves, a.plot, a.new)
    print(f"[infill] {D} documents, {C} configurations -> {out}")
    return 0


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def _speedup(tps, acc, policy_of, new, target_acc, t) -> float:
    """tokens/step at which ``new`` reaches ``target_acc``, over t (point estimate; NaN if
    its curve is not monotone decreasing or never reaches that accuracy)."""
    idx = np.flatnonzero(policy_of == new)
    if not np.isfinite(target_acc) or idx.size < 2:
        return float("nan")
    o = np.argsort(tps[idx])
    x, y = np.log2(tps[idx][o]), acc[idx][o]
    if np.any(np.diff(y) > 0) or not (y[-1] <= target_acc <= y[0]):
        return float("nan")
    return float(2 ** np.interp(target_acc, y[::-1], x[::-1]) / t)


def plot_curves(curves, path, new):
    try:
        import matplotlib
    except ImportError:                 # the cluster env has no matplotlib; W&B plots instead
        print("[infill] matplotlib not installed; no plot")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in sorted({c["policy"] for c in curves}):
        pts = sorted((c["tokens_per_step"], c["acc"]) for c in curves if c["policy"] == p)
        x, y = zip(*pts)
        style = dict(lw=2.5, marker="o") if p == new else dict(lw=1, ls="--", marker=".")
        ax.plot(np.log2(x), y, label=p, **style)
    ax.set_xlabel("log2 tokens per step")
    ax.set_ylabel("infilling token accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
