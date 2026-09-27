#!/usr/bin/env python3
"""The plan v2 endpoints on one trace file: E1, E2, ECE, Eq. 2/3 with and without FE.

Input is either a parquet trace written by the harness, or ``sim:<scenario>``
for a synthetic trace with known truth (``c3.sim.positions``).  For every mask
kind in the trace it reports:

* Delta(m) = E_m[c - a] for m in POOL_SIZES, document-first, with 10k-resample
  percentile intervals, plus the contrast Delta(32) - Delta(1) and its declare
  decision (CI excludes 0 AND > 2x floor);
* the 15-bin equal-mass ECE (with interval) and smooth ECE over all masked
  positions, with reliability bins;
* the Prop. 2 decomposition: Delta_own, the field term, the residual, Delta_ctx,
  from cross-fitted calibrators, and the ICC of residuals within states;
* Eq. 2 on winners and Eq. 3 on the top ranks, each with and without state FE;
* sigma_d of the per-document contrast and N = 7.85 sigma_d^2 / delta_min^2
  (labelled provisional: G1 is read off the v2 T2 pilot, not this script);
* for decoder-induced states, the native gap over the tokens the decoder committed.

Outputs: ``results.json``, figures (PNG), and ``macros.tex`` with one
``\\newcommand`` per headline number for the human-written report.

Split modes: ``kfold-doc`` (leave-one-document-out calibrators; the pilot, which
is too small for halves), ``half`` (fit on half 1, report half 2; the main
study), ``none`` (in-sample; diagnostics only).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3 import DEFAULT_KNOTS, DELTA_MIN, N_BOOT, N_POOLS, POOL_SIZES, RETAIN
from c3.assemble import SCALAR_COLUMNS, arm_labels, load_traces, position_and_pool_frames
from c3.calibrators import crossfit_predictions, fit_on_half
from c3.endpoints.e1 import delta_m, gap_slope_log2, native_gap
from c3.endpoints.e2 import decompose, icc_residuals
from c3.frame import hash_halves
from c3.metrics import ece_equal_mass, smooth_ece
from c3.power import required_documents_sd
from c3.provenance import collect as provenance, sha256_file
from c3.regression import fit_and_report
from c3.stats import declare, doc_boot_counts, percentile_ci

_WORDS = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen", 32: "ThirtyTwo",
          64: "SixtyFour", 128: "OneTwentyEight", 256: "TwoFiftySix"}


def load_input(spec: str):
    if spec.startswith("sim:"):
        from c3.sim.positions import simulate_positions
        parts = spec[4:].split(",")
        kw = {}
        for p in parts[1:]:
            k, v = p.split("=")
            kw[k] = type(_SIM_DEFAULTS.get(k, 0.0))(v) if k in _SIM_DEFAULTS else float(v)
        return simulate_positions(parts[0], **kw), {"source": spec}
    df = load_traces(spec, columns=SCALAR_COLUMNS)
    meta = {"source": spec, "sha256": sha256_file(spec)}
    meta_path = Path(spec).with_suffix(".meta.json")
    if meta_path.exists():
        meta["trace_meta"] = json.loads(meta_path.read_text())
    return df, meta


_SIM_DEFAULTS = {"n_docs": 200, "states_per_doc": 4, "n_positions": 128, "seed": 0}


def fold_assignment(split: str, codes: np.ndarray, seed: int, manifest: str | None):
    """Returns (fold_of_doc or None, fit_docs or None, report_docs)."""
    if split == "kfold-doc":
        return {int(d): int(d) for d in codes}, None, codes
    if split == "none":
        return None, codes, codes
    if split == "half":
        if manifest:
            m = json.loads(Path(manifest).read_text())
            half = np.array([int(m[str(int(d))]) for d in codes])
        else:
            half = hash_halves(codes, seed)
        return None, codes[half == 0], codes[half == 1]
    raise ValueError(split)


def regression_block(pools, report_docs, fit_docs, split, n_knots, n_boot, seed, n_jobs,
                     ridge_beta=0.0):
    """Eq. 2 on winners and Eq. 3 on the retained top ranks, with and without FE."""
    out = {}
    keep = np.isin(pools.doc_id, report_docs if split != "half" else np.concatenate([fit_docs, report_docs]))
    fr = pools.select(keep)
    cross = split == "half"
    for eq, frame in (("eq2", fr.winners()), ("eq3", fr)):
        for fe in (False, True):
            key = f"{eq}_{'fe' if fe else 'nofe'}"
            try:
                est = fit_and_report(frame, eq, n_knots=n_knots, n_boot=n_boot, seed=seed,
                                     n_jobs=n_jobs, cross_fit=cross, fe=fe,
                                     fit_docs=fit_docs if cross else None,
                                     max_rank=RETAIN, ridge_beta=ridge_beta)
                out[key] = {name: {"value": e.value, "ci": [e.ci_low, e.ci_high], "se": e.se,
                                   "n_failed_draws": int(n_boot - (e.boot.size if e.boot is not None else 0)),
                                   "point_converged": e.converged, "point_iters": e.n_iter}
                            for name, e in est.items()}
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                out[key] = {"error": str(exc)}
    return out


def analyze_kind(df_k, kind, codes, counts, args, fit_info):
    fold_of_doc, fit_docs, report_docs = fit_info
    pos, pools = position_and_pool_frames(df_k, pool_sizes=POOL_SIZES, n_pools_per_size=N_POOLS,
                                          seed=args.seed, retain=RETAIN,
                                          min_state_size=max(POOL_SIZES))
    report_mask = np.isin(pos.doc_id, report_docs)
    res: dict = {"mask_kind": kind,
                 "n_docs": int(np.unique(pos.doc_id[report_mask]).size),
                 "n_states": int(np.unique(pos.state_id[report_mask]).size),
                 "n_positions": int(report_mask.sum()),
                 "saturated_share": float(np.mean((pos.conf[report_mask] >= 1 - 1e-6)
                                                  | (np.abs(pos.ell[report_mask]) >= 13.8)))}

    # ---- calibrators (out-of-sample for every reported position)
    if fold_of_doc is not None:
        pi_own, pi_ctx, fits = crossfit_predictions(pos, fold_of_doc, n_knots=args.knots,
                                                    interactions=args.context_model == "interacted")
        res["calibrators"] = {"mode": "leave-one-document-out", "folds": len(fits),
                              "context_features": sorted({f for x in fits for f in x["context_features"]}),
                              "all_converged": all(x["context_converged"] for x in fits)}
    else:
        pi_own, pi_ctx, fitted = fit_on_half(pos, fit_docs, n_knots=args.knots,
                                             interactions=args.context_model == "interacted")
        res["calibrators"] = {"mode": "fit-half" if args.split == "half" else "in-sample",
                              "fitted": fitted}

    # ---- E1
    ex = dict(method=args.pools, pool_sizes=POOL_SIZES, min_state_size=max(POOL_SIZES))
    delta = delta_m(pools, pos, codes, counts, pos_mask=report_mask, **ex)
    res["delta"] = delta.to_records("delta")
    hi, lo = max(POOL_SIZES), min(POOL_SIZES)
    c_point, c_draws = delta.contrast(hi, lo)
    c_ci = percentile_ci(c_draws)
    per_doc = delta.per_doc_contrast(hi, lo)
    sigma_d = float(np.std(per_doc, ddof=1)) if per_doc.size > 1 else float("nan")
    slope, slope_draws = gap_slope_log2(delta)
    res["contrast"] = {"name": f"Delta({hi}) - Delta({lo})", "value": c_point, "ci": list(c_ci),
                       "declared": declare(c_point, c_ci, args.floor), "floor": args.floor,
                       "slope_log2m": slope, "slope_ci": list(percentile_ci(slope_draws))}
    res["power_provisional"] = {
        "note": "provisional, not the pre-registered G1 (that is read off the v2 T2 pilot)",
        "sigma_d": sigma_d, "delta_min": args.delta_min, "n_docs_used": int(per_doc.size),
        "n_required": (required_documents_sd(sigma_d, args.delta_min)
                       if np.isfinite(sigma_d) else None)}

    # ---- ECE over all masked positions
    pr = pos.select(report_mask)
    doc_idx = np.searchsorted(codes, pr.doc_id)
    ece = ece_equal_mass(pr.conf, pr.agree, 15, doc_idx, counts)
    res["ece"] = ece.to_dict()
    if not args.no_smooth_ece:
        sm, bw = smooth_ece(pr.conf, pr.agree)
        res["smooth_ece"] = {"value": sm, "bandwidth": bw}

    # ---- E2
    dec = decompose(pools, pos, pi_own, pi_ctx, codes, counts, pos_mask=report_mask, **ex)
    res["decomposition"] = dec.to_records()
    f_point, f_draws = dec.field.contrast(hi, lo)
    f_ci = percentile_ci(f_draws)
    f_per_doc = dec.field.per_doc_contrast(hi, lo)
    sigma_f = float(np.std(f_per_doc, ddof=1)) if f_per_doc.size > 1 else float("nan")
    res["field_contrast"] = {"name": f"field({hi}) - field({lo})", "value": f_point,
                             "ci": list(f_ci), "declared": declare(f_point, f_ci, args.floor),
                             "sigma_d": sigma_f,
                             "n_required_provisional": (required_documents_sd(sigma_f, args.delta_min)
                                                        if np.isfinite(sigma_f) else None)}
    res["icc_residuals"] = icc_residuals(pos, pi_ctx, codes, counts, pos_mask=report_mask)

    # ---- Eq. 2 / 3
    if not args.skip_regression:
        res["regression"] = regression_block(pools, report_docs, fit_docs, args.split,
                                             args.knots, args.eq_boot, args.seed, args.n_jobs,
                                             ridge_beta=args.ridge)
        if args.split != "half":
            res["regression"]["note"] = ("pooled fit on the reported documents (no half split): "
                                         "feasibility estimate, not a reported result")

    # ---- native gap for decoder-induced states
    native = None
    if kind.split(":")[0] == "decoder" and args.native_k:
        # v2 traces store the commits; v1 commit sets are rebuilt from k.
        committed = (df_k["committed"].to_numpy(bool) if "committed" in df_k.columns else None)
        k_arm = int(df_k["k"].iloc[0]) if "k" in df_k.columns else args.native_k
        try:
            native = native_gap(pos, codes, counts, k=k_arm, min_masked=args.min_masked,
                                pos_mask=report_mask, committed=committed)
            res["native"] = {"k": k_arm, "grouped_by": "2^floor(log2 |M_t|)",
                             "commits": "stored" if committed is not None else "rebuilt (v1)",
                             "rows": native.to_records("delta")}
        except ValueError as exc:
            res["native"] = {"error": str(exc)}
    return res, {"delta": delta, "dec": dec, "ece": ece, "native": native}


# --------------------------------------------------------------------------- #
# Figures and macros
# --------------------------------------------------------------------------- #

def make_figures(out: Path, objs: dict, results: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from c3.metrics import plot_reliability

    written = []
    for kind, o in objs.items():
        dec, ece = o["dec"], o["ece"]
        ms = dec.delta.levels
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for est, lab, style in ((dec.delta, "observed gap $\\Delta(m)$", "-o"),
                                (dec.delta_own, "own-confidence calibrator", "--s"),
                                (dec.delta_ctx, "context-aware calibrator", ":^")):
            ci = est.ci()
            ax.errorbar(ms, est.point, yerr=[est.point - ci[:, 0], ci[:, 1] - est.point],
                        fmt=style, ms=4, lw=1.3, capsize=2, label=lab)
        ax.axhline(ece.ece, color="0.5", lw=0.8, ls=":")
        ax.text(ms[-1], ece.ece, f" ECE {ece.ece:.3f}", va="bottom", ha="right", fontsize=8, color="0.4")
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ms, [str(int(m)) for m in ms])
        ax.set_xlabel("field size $m$")
        ax.set_ylabel("confidence $-$ accuracy")
        ax.set_title(f"{kind} masks")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        p = out / f"fig_gap_{kind.replace(':', '_')}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p.name)

    if objs:
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_reliability(ax, {k: o["ece"] for k, o in objs.items()}, "reliability, all masked positions")
        fig.tight_layout()
        p = out / "fig_reliability.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p.name)

    for kind, o in objs.items():
        nat = o.get("native")
        if nat is None:
            continue
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ci = nat.ci()
        ax.errorbar(nat.levels, nat.point, yerr=[nat.point - ci[:, 0], ci[:, 1] - nat.point],
                    fmt="-o", ms=4, capsize=2)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("masked positions $|M_t|$ when committed")
        ax.set_ylabel("confidence $-$ accuracy (committed)")
        ax.set_title(f"native gap, {kind}")
        fig.tight_layout()
        p = out / f"fig_native_{kind.replace(':', '_')}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p.name)
    return written


_KIND_TAGS = {"uniform": "Uni", "decoder": "Dec", "uniform_matched": "UniMatched"}


def arm_tag(arm: str) -> str:
    """Digit-free macro tag of an arm: 'decoder:topk_conf:k4' -> 'DecTopkConfKFour'."""
    kind, *rest = arm.split(":")
    tag = _KIND_TAGS.get(kind, kind.title().replace("_", ""))
    for part in rest:
        if part[:1] == "k" and part[1:].isdigit():
            tag += "K" + _WORDS[int(part[1:])]
        else:
            tag += "".join(w.title() for w in part.split("_"))
    return tag


def _macro_name(prefix: str, *parts) -> str:
    name = prefix + "".join(p[:1].upper() + p[1:] for p in parts)
    if any(ch.isdigit() for ch in name):
        raise ValueError(f"macro name {name!r} contains a digit")
    return name


def write_macros(path: Path, prefix: str, results: dict) -> None:
    lines = ["% Generated by scripts/analyze_traces.py -- do not edit by hand.",
             f"% source: {results['input']['source']}  commit: {results['provenance']['git_commit']}"]

    def add(name, value, fmt="{:.3f}"):
        lines.append(f"\\newcommand{{\\{name}}}{{{fmt.format(value)}}}")

    for kind, r in results["by_mask_kind"].items():
        k = arm_tag(kind)
        for row in r["delta"]:
            m = _WORDS[int(row["level"])]
            add(_macro_name(prefix, "Gap", m, k), row["delta"])
            add(_macro_name(prefix, "Gap", m, k, "Lo"), row["ci_low"])
            add(_macro_name(prefix, "Gap", m, k, "Hi"), row["ci_high"])
        add(_macro_name(prefix, "Contrast", k), r["contrast"]["value"])
        add(_macro_name(prefix, "Contrast", k, "Lo"), r["contrast"]["ci"][0])
        add(_macro_name(prefix, "Contrast", k, "Hi"), r["contrast"]["ci"][1])
        add(_macro_name(prefix, "ECE", k), r["ece"]["ece"])
        for row in r["decomposition"]:
            m = _WORDS[int(row["m"])]
            add(_macro_name(prefix, "Field", m, k), row["field"])
            add(_macro_name(prefix, "Own", m, k), row["delta_own"])
        if r["power_provisional"]["n_required"] is not None:
            add(_macro_name(prefix, "NReq", k), r["power_provisional"]["n_required"], "{:d}")
        reg = r.get("regression", {})
        for key, tag in (("eq2_nofe", "NoFE"), ("eq2_fe", "FE")):
            bz = reg.get(key, {}).get("beta_z")
            if bz:
                add(_macro_name(prefix, "BetaZ", tag, k), bz["value"])
        add(_macro_name(prefix, "NDocs", k), r["n_docs"], "{:d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="parquet trace, or sim:<scenario>[,key=value...]")
    ap.add_argument("--mask-kinds", nargs="+", default=None)
    ap.add_argument("--split", choices=["kfold-doc", "half", "none"], default="kfold-doc")
    ap.add_argument("--manifest", default=None, help="JSON {doc_id: 0|1}; 0 = fit half (split=half)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--eq-boot", type=int, default=None, help="resamples for Eq. 2/3 (default: --n-boot)")
    ap.add_argument("--knots", type=int, default=DEFAULT_KNOTS)
    ap.add_argument("--floor", type=float, default=0.0, help="non-determinism floor, Delta units")
    ap.add_argument("--delta-min", type=float, default=DELTA_MIN)
    ap.add_argument("--native-k", type=int, default=None,
                    help="tokens per step of the teacher-forced decoder (default: from .meta.json, else 8)")
    ap.add_argument("--min-masked", type=int, default=None)
    ap.add_argument("--pools", choices=["mc", "exact"], default="mc",
                    help="E1/E2 over the 20 drawn pools per size (plan v2) or the exact "
                         "expectation over all pools (c3.pools); Eq. 2/3 always use drawn pools")
    ap.add_argument("--context-model", choices=["interacted", "additive"], default="interacted",
                    help="context calibrator: spline(l) + features (+ l x features); see c3.calibrators")
    ap.add_argument("--ridge", type=float, default=1e-6,
                    help="ridge on the standardised Eq. 2/3 coefficients; keeps the fit finite "
                         "when high-confidence positions are all correct (quasi-separation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--label", default="", help="free-text label stored with the results")
    ap.add_argument("--macro-prefix", default="Res")
    ap.add_argument("--skip-regression", action="store_true")
    ap.add_argument("--no-smooth-ece", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.eq_boot is None:
        args.eq_boot = args.n_boot

    df, meta = load_input(args.traces)
    tcfg = meta.get("trace_meta", {}).get("config", {})
    if args.native_k is None:
        args.native_k = int(tcfg.get("tokens_per_step", 8))
    if args.min_masked is None:
        args.min_masked = int(tcfg.get("min_masked", 32))

    df["arm"] = arm_labels(df)
    kinds = args.mask_kinds or sorted(df["arm"].unique().tolist())
    codes = np.unique(df["doc_id"].to_numpy(np.int64))
    counts = doc_boot_counts(codes.size, args.n_boot, seed=args.seed, label="analyze-traces")
    fit_info = fold_assignment(args.split, codes, args.seed, args.manifest)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"label": args.label, "input": meta, "config": {
        "split": args.split, "pool_sizes": list(POOL_SIZES), "n_pools": N_POOLS, "retain": RETAIN,
        "n_boot": args.n_boot, "eq_boot": args.eq_boot, "knots": args.knots, "floor": args.floor,
        "delta_min": args.delta_min, "seed": args.seed, "native_k": args.native_k,
        "min_masked": args.min_masked, "context_model": args.context_model,
        "pools": args.pools, "ridge": args.ridge,
        "calibrator_weighting": "pool (1 / (|state| x states per doc))",
        "sign": "conf - agree (positive = overconfident)",
        "resamples_shared_across": "all pool sizes, estimands and mask kinds"},
        "by_mask_kind": {}}
    objs = {}
    for kind in kinds:
        df_k = df[df["arm"] == kind]
        if len(df_k) == 0:
            continue
        print(f"[analyze] {kind}: {len(df_k)} rows", flush=True)
        res, o = analyze_kind(df_k, kind, codes, counts, args, fit_info)
        results["by_mask_kind"][kind] = res
        objs[kind] = o
        c = res["contrast"]
        print(f"[analyze]   {c['name']} = {c['value']:+.4f}  CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"
              f"  ECE {res['ece']['ece']:.4f}", flush=True)

    results["provenance"] = provenance()
    if not args.no_figures:
        results["figures"] = make_figures(out, objs, results)
    results["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    write_macros(out / "macros.tex", args.macro_prefix, results)
    print(f"[analyze] wrote {out / 'results.json'} and {out / 'macros.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
