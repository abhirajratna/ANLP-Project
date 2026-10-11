#!/usr/bin/env python3
"""E3 (plan v2 T7): decoder-visited vs random states at the same confidence.

    # development: fit half only (the fit half is split again, so nuisance
    # choices and estimates still come from different documents)
    python scripts/analyze_e3.py --traces runs/traces/llada-base.parquet \\
        --split half --manifest artifacts/split_manifest.json --half fit --out runs/e3-dev

    # the pre-registered run (prereg/analysis_commands.sh): report half
    python scripts/analyze_e3.py --traces ... --split half --manifest ... --half report ...

    # synthetic traces with known truth (c3.sim.trajectories)
    python scripts/analyze_e3.py --traces sim-design:decoder_shift,n_docs=80 --split half \\
        --half report --out /tmp/e3

For every decoder arm (policy, k) in the trace it compares the decoder states
with (a) the matched uniform states of the same arm (primary: |M| equal state by
state) and (b) the uniform grid, reweighted to the decoder's mask-rate mix, using
``c3.endpoints.e3.e3_compare``: the type effect at equal confidence, the joint
test of the type terms, the difference curve with pointwise and simultaneous
intervals, the effect by mask-rate bin, and side-by-side reliability bins.

Nuisance choices (spline knots, ell-bin edges, reliability bins) are frozen on
the fit half; estimates come from the other documents.  ``--split half``
requires ``--half``: there is no default, so a report-half number is never
computed by accident.

Outputs ``results.json``, figures and ``macros.tex`` in ``--out``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3 import DEFAULT_KNOTS, N_BOOT, POOL_SIZES
from c3.endpoints.e3 import RATE_CENTRES, E3Nuisance, contrast_of_contrasts, e3_compare
from c3.frame import PositionFrame, hash_halves
from c3.provenance import collect as provenance, sha256_file
from c3.stats import doc_boot_counts

COLUMNS = ["doc_id", "state_id", "position", "ell", "conf", "agree", "mask_kind", "policy",
           "k", "mask_rate", "n_masked", "step"]
KINDS = ("decoder", "uniform", "uniform_matched")
_WORDS = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen", 32: "ThirtyTwo"}


def load_input(spec: str):
    """A trace restricted to E3's columns and mask kinds, or a simulated one."""
    if spec.startswith("sim-design:"):
        from c3.sim.trajectories import simulate_design
        name, *kv = spec.split(":", 1)[1].split(",")
        kw = {}
        for p in kv:
            k, v = p.split("=")
            kw[k] = tuple(int(x) for x in v.split("+")) if k == "ks" else (
                int(v) if k in ("n_docs", "doc_len", "seed", "masks_per_rate") else float(v))
        return simulate_design(name, **kw), {"source": spec}
    import pyarrow.parquet as pq

    names = set(pq.read_schema(spec).names)
    cols = [c for c in COLUMNS if c in names]
    table = pq.read_table(spec, columns=cols, filters=[("mask_kind", "in", list(KINDS))],
                          read_dictionary=[c for c in ("mask_kind", "policy") if c in cols])
    df = table.to_pandas()
    for c in ("mask_kind", "policy"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    meta = {"source": spec, "sha256": sha256_file(spec)}
    mp = Path(spec).with_suffix(".meta.json")
    if mp.exists():
        m = json.loads(mp.read_text())
        meta["trace_meta"] = {k: m.get(k) for k in ("model", "variant", "precision", "dataset",
                                                     "git_commit", "n_docs", "config")}
    return df, meta


def halves(codes: np.ndarray, split: str, half: str | None, manifest: str | None, seed: int):
    """(nuisance documents, estimation documents, label)."""
    if split == "half":
        if half is None:
            raise SystemExit("--split half needs --half fit (development) or --half report "
                             "(the pre-registered run); there is no default on purpose")
        if manifest:
            m = json.loads(Path(manifest).read_text())
            missing = [int(d) for d in codes if str(int(d)) not in m]
            if missing:
                raise SystemExit(f"{len(missing)} documents are not in {manifest} (first: "
                                 f"{missing[:5]}); pilot documents are left out on purpose")
            h = np.array([int(m[str(int(d))]) for d in codes])
        else:
            print("[e3] WARNING: --split half without --manifest uses hash_halves, not the "
                  "committed split", file=sys.stderr, flush=True)
            h = hash_halves(codes, seed)
        fit, report = codes[h == 0], codes[h == 1]
        if half == "report":
            return fit, report, "report half (nuisance from the fit half)"
        # Development: split the fit half again so nothing is fitted and scored on
        # the same documents, and never touch the report half.
        h2 = hash_halves(fit, seed + 1)
        return fit[h2 == 0], fit[h2 == 1], "DEVELOPMENT: fit half only, not a reported number"
    if split == "none":
        return codes, codes, "in-sample (diagnostic only)"
    raise ValueError(split)


def arms_of(df) -> list[tuple[str, int]]:
    dec = df[df["mask_kind"] == "decoder"]
    return sorted({(str(p), int(k)) for p, k in zip(dec["policy"], dec["k"])})


def _pf(df) -> PositionFrame:
    return PositionFrame.from_frame(df.reset_index(drop=True))


def tag(policy: str, k: int, ref: str) -> str:
    pol = "".join(w.title() for w in policy.split("_"))
    return f"{pol}K{_WORDS.get(k, str(k))}{'Matched' if ref == 'matched' else 'Grid'}"


def make_figures(out: Path, results: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for key, r in results["comparisons"].items():
        if "error" in r:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
        ax = axes[0]
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", zorder=0)
        for name, style in (("reference", "-o"), ("decoder", "-s")):
            bins = [b for b in r["reliability"][name]["bins"] if b["conf"] is not None]
            x = [b["conf"] for b in bins]
            y = [b["acc"] for b in bins]
            lo = [b["acc"] - b["acc_ci"][0] for b in bins]
            hi = [b["acc_ci"][1] - b["acc"] for b in bins]
            ax.errorbar(x, y, yerr=[lo, hi], fmt=style, ms=3, lw=1.1, capsize=2,
                        label=f"{name} (wECE {r['reliability'][name]['weighted_ece']:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_title(f"reliability, {key}", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
        ax = axes[1]
        dc = r["difference_curve"]
        x = np.array(dc["conf"])
        v = np.array(dc["value"])
        ci = np.array(dc["ci"], dtype=float)
        ax.fill_between(x, ci[:, 0], ci[:, 1], alpha=0.25, lw=0, label="pointwise 95%")
        band = dc["simultaneous"].get("band")
        if band:
            b = np.array(band, dtype=float)
            ax.plot(x, b[:, 0], color="C0", lw=0.8, ls=":")
            ax.plot(x, b[:, 1], color="C0", lw=0.8, ls=":", label="simultaneous 95%")
        ax.plot(x, v, "-", color="C0", lw=1.4, label="$\\pi_{ref}(\\ell) - \\pi_{dec}(\\ell)$")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy difference (ref - dec)")
        te = r["type_effect"]
        ax.set_title(f"type effect {te['value']:+.3f} [{te['ci'][0]:+.3f}, {te['ci'][1]:+.3f}]",
                     fontsize=9)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        p = out / f"fig_e3_{key.replace(':', '_')}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p.name)
    return written


def write_macros(path: Path, prefix: str, results: dict) -> None:
    lines = ["% Generated by scripts/analyze_e3.py -- do not edit by hand.",
             f"% source: {results['input']['source']}  commit: {results['provenance']['git_commit']}"
             f"  scope: {results['scope']}"]

    def add(name, value, fmt="{:.3f}"):
        if any(ch.isdigit() for ch in name):
            raise ValueError(f"macro name {name!r} contains a digit")
        if value is not None and np.isfinite(value):
            lines.append(f"\\newcommand{{\\{name}}}{{{fmt.format(value)}}}")

    for key, r in results["comparisons"].items():
        if "error" in r:
            continue
        t = r["tag"]
        te = r["type_effect"]
        add(f"{prefix}TypeEffect{t}", te["value"])
        add(f"{prefix}TypeEffect{t}Lo", te["ci"][0])
        add(f"{prefix}TypeEffect{t}Hi", te["ci"][1])
        add(f"{prefix}TypeTestP{t}", r["type_test"].get("p"))
        cc = r.get("delta_contrast_of_contrasts")
        if cc:
            add(f"{prefix}GapContrastDiff{t}", cc["value"])
            add(f"{prefix}GapContrastDiff{t}Lo", cc["ci"][0])
            add(f"{prefix}GapContrastDiff{t}Hi", cc["ci"][1])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True,
                    help="merged parquet trace, or sim-design:<scenario>[,key=value...]")
    ap.add_argument("--split", choices=["half", "none"], default="half")
    ap.add_argument("--half", choices=["fit", "report"], default=None)
    ap.add_argument("--manifest", default=None, help="artifacts/split_manifest.json")
    ap.add_argument("--exclude-docs", default=None,
                    help="contamination screen output, or a JSON list of doc ids, to drop")
    ap.add_argument("--reference", choices=["matched", "grid", "both"], default="both")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="decoder arms as policy:k (default: every one in the trace)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--knots", type=int, default=DEFAULT_KNOTS)
    ap.add_argument("--n-ell-bins", type=int, default=100)
    ap.add_argument("--grid-size", type=int, default=25)
    ap.add_argument("--no-by-rate", action="store_true")
    ap.add_argument("--delta-contrast", action="store_true",
                    help="also [Delta(32)-Delta(1)] decoder minus reference (draws pools: slow)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--macro-prefix", default="EThree")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df, meta = load_input(a.traces)
    if a.exclude_docs:
        from c3.data.contamination import excluded_doc_ids
        drop = excluded_doc_ids(a.exclude_docs)
        present = sorted(set(df["doc_id"].unique().tolist()) & drop)
        df = df[~df["doc_id"].isin(drop)].reset_index(drop=True)
        meta["excluded_docs"] = {"source": a.exclude_docs, "n_dropped": len(present),
                                 "doc_ids": present}
    codes_all = np.unique(df["doc_id"].to_numpy(np.int64))
    nuis_docs, est_docs, scope = halves(codes_all, a.split, a.half, a.manifest, a.seed)
    print(f"[e3] {scope}: {nuis_docs.size} nuisance / {est_docs.size} estimation documents",
          flush=True)
    codes = np.sort(est_docs)
    counts = doc_boot_counts(codes.size, a.n_boot, seed=a.seed, label="analyze-e3")

    arms = arms_of(df)
    if a.arms:
        want = {(p.rsplit(":", 1)[0], int(p.rsplit(":", 1)[1])) for p in a.arms}
        arms = [x for x in arms if x in want]
    if not arms:
        raise SystemExit("no decoder arms in the trace")
    in_est = df["doc_id"].isin(est_docs).to_numpy()
    in_nuis = df["doc_id"].isin(nuis_docs).to_numpy()
    kind = df["mask_kind"].to_numpy()
    policy = df["policy"].astype(str).to_numpy()
    kk = df["k"].to_numpy(np.int64)

    results = {"input": meta, "scope": scope, "config": {
        "split": a.split, "half": a.half, "manifest": a.manifest, "n_boot": a.n_boot,
        "knots": a.knots, "n_ell_bins": a.n_ell_bins, "grid_size": a.grid_size, "seed": a.seed,
        "rate_bins": RATE_CENTRES.tolist(), "reference": a.reference,
        "n_nuisance_docs": int(nuis_docs.size), "n_estimation_docs": int(est_docs.size),
        "weights": "pool weights within each arm; grid reference reweighted to the decoder "
                   "arm's mask-rate-bin mix",
        "resamples_shared_across": "every comparison"}, "comparisons": {}}

    grid_sel = kind == "uniform"
    for pol, k in arms:
        dec_sel = (kind == "decoder") & (policy == pol) & (kk == k)
        mat_sel = (kind == "uniform_matched") & (policy == pol) & (kk == k)
        refs = []
        if a.reference in ("matched", "both") and mat_sel.any():
            refs.append(("matched", mat_sel, False))
        if a.reference in ("grid", "both") and grid_sel.any():
            refs.append(("grid", grid_sel, True))
        for ref_name, ref_sel, match_rates in refs:
            key = f"decoder:{pol}:k{k}_vs_{ref_name}"
            print(f"[e3] {key}", flush=True)
            nu_rows = (dec_sel | ref_sel) & in_nuis
            try:
                nu = E3Nuisance.from_positions(df["ell"].to_numpy()[nu_rows],
                                               df["conf"].to_numpy()[nu_rows],
                                               n_knots=a.knots, n_ell_bins=a.n_ell_bins)
                D = _pf(df[dec_sel & in_est])
                R = _pf(df[ref_sel & in_est])
                r = e3_compare(D, R, codes, counts, nu, match_rates=match_rates,
                               grid_size=a.grid_size, by_rate=not a.no_by_rate)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                results["comparisons"][key] = {"error": str(exc)}
                print(f"[e3]   failed: {exc}", flush=True)
                continue
            r["nuisance"] = nu.to_dict()
            r["tag"] = tag(pol, k, ref_name)
            if a.delta_contrast:
                from c3.assemble import position_and_pool_frames
                from c3.endpoints.e1 import delta_m
                g = []
                for P in (D, R):
                    sub = df[(dec_sel if P is D else ref_sel) & in_est].reset_index(drop=True)
                    pos, pools = position_and_pool_frames(sub, pool_sizes=POOL_SIZES,
                                                          seed=a.seed, retain=1,
                                                          min_state_size=max(POOL_SIZES))
                    g.append(delta_m(pools, pos, codes, counts, min_state_size=max(POOL_SIZES)))
                r["delta_contrast_of_contrasts"] = contrast_of_contrasts(
                    g[0], g[1], max(POOL_SIZES), min(POOL_SIZES))
            te = r["type_effect"]
            print(f"[e3]   type effect {te['value']:+.4f} [{te['ci'][0]:+.4f}, {te['ci'][1]:+.4f}]"
                  f"  joint p {r['type_test'].get('p')}", flush=True)
            results["comparisons"][key] = r

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results["provenance"] = provenance()
    results["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    if not a.no_figures:
        results["figures"] = make_figures(out, results)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float),
                                      encoding="utf-8")
    write_macros(out / "macros.tex", a.macro_prefix, results)
    print(f"[e3] wrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
