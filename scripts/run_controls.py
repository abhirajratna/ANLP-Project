#!/usr/bin/env python3
"""T8 controls (plan v2): one ablation table, every control a row.

    # development on the fit half (split again; the report half is never read)
    python scripts/run_controls.py --traces runs/traces/llada-base.parquet \\
        --split half --manifest artifacts/split_manifest.json --half fit \\
        --docs runs/corpus/calibration_docs.jsonl --out runs/t8-dev

    # precision: the 200-document 4-bit run against the same documents in fp16
    python scripts/run_controls.py --traces <fp16 trace> --precision-trace <4-bit trace> \\
        --controls precision --split half --manifest ... --half fit --out ...

    # synthetic traces (c3.sim.trajectories)
    python scripts/run_controls.py --traces sim-design:position,n_docs=80 --split half \\
        --half report --out /tmp/t8

Controls (``c3.controls``): ``position`` (absolute position, distance to the
nearest unmasked token, PC-Sampler's weight as covariates), ``token_type``
(punctuation / stop-word / content: covariates, and the split by type),
``temperature`` (one per mask-rate bin, fitted on the fit half; log-odds and
softmax, one row each), ``precision``
(fp16 vs 4-bit on the same documents), and ``statistic`` (T13 ablation 1, not run
by default: pools won by the highest margin or lowest entropy instead of ell).  Every row gives the main effect -- E1
Delta(32) - Delta(1), E2 field(32) - field(1) -- without and with the control,
and the paired difference, on one resample matrix per document set.

Outputs ``ablation.json``, ``ablation.tex`` and ``macros.tex`` in ``--out``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from c3 import DEFAULT_KNOTS, N_BOOT, N_POOLS, POOL_SIZES
from c3 import controls as C
from c3.assemble import arm_labels, position_and_pool_frames
from c3.endpoints.e1 import winner_gap
from c3.provenance import collect as provenance, sha256_file
from c3.stats import doc_boot_counts, grouped_doc_mean

HI, LO = max(POOL_SIZES), min(POOL_SIZES)
BASE_COLUMNS = ["doc_id", "state_id", "state_idx", "position", "ell", "conf", "agree", "top1",
                "mask_kind", "policy", "k", "mask_rate", "n_masked", "step", "doc_len",
                "span_start", "model", "dataset"]
_WORDS = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen", 32: "ThirtyTwo"}


def _sim_kw(spec):
    name, *kv = spec.split(":", 1)[1].split(",")
    kw = {}
    for p in kv:
        k, v = p.split("=")
        kw[k] = tuple(int(x) for x in v.split("+")) if k == "ks" else (
            int(v) if k in ("n_docs", "doc_len", "seed", "masks_per_rate") else float(v))
    return name, kw


def load(spec: str, extra_cols=(), arms=None):
    """The trace's scalar columns (and ``extra_cols``), or a simulated trace.  With
    ``arms``, only rows of those arms' mask kinds are read (a 110M or 8B fit half is
    ~60M rows)."""
    if spec.startswith("sim-design:"):
        from c3.sim.trajectories import simulate_design
        name, kw = _sim_kw(spec)
        return simulate_design(name, **kw), {"source": spec}
    from c3.assemble import read_trace_table, trace_meta

    filters = [("mask_kind", "in", sorted({x.split(":")[0] for x in arms}))] if arms else None
    df = read_trace_table(spec, columns=list(BASE_COLUMNS) + list(extra_cols),
                          filters=filters).to_pandas()
    for c in ("mask_kind", "policy", "model", "dataset"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    meta = {"source": spec, "sha256": sha256_file(spec)}
    m = trace_meta(spec)
    if m:
        meta["trace_meta"] = {k: m.get(k) for k in ("model", "variant", "precision", "dataset",
                                                     "git_commit", "n_docs", "path", "revision")}
    return df, meta


def halves(codes, split, half, manifest, seed):
    from analyze_e3 import halves as h
    return h(codes, split, half, manifest, seed)


def arm_tag(arm: str) -> str:
    kind, *rest = arm.split(":")
    tag = {"uniform": "Uni", "decoder": "Dec", "uniform_matched": "UniMatched"}.get(
        kind, kind.title().replace("_", ""))
    for part in rest:
        if part[:1] == "k" and part[1:].isdigit():
            tag += "K" + _WORDS[int(part[1:])]
        else:
            tag += "".join(w.title() for w in part.split("_"))
    return tag


# --------------------------------------------------------------------------- #
# token classes
# --------------------------------------------------------------------------- #

def slot_classes(df, args, meta) -> np.ndarray:
    """TOKEN_CLASSES index of every row's true token."""
    if args.traces.startswith("sim-design:"):
        from c3.sim.trajectories import true_token_ids
        _, kw = _sim_kw(args.traces)
        x0 = true_token_ids(kw.get("n_docs", 40), kw.get("doc_len", 64), kw.get("seed", 0))
        ids = C.slot_token_ids(df, x0)
        # The simulator's ids below 10 are its trivial tokens: 0-4 punctuation, 5-9 stop-words.
        return np.where(ids < 5, 1, np.where(ids < 10, 2, 0)).astype(np.int8)
    from transformers import AutoTokenizer

    from c3.traces.models import KNOWN_MODELS
    model = str(df["model"].iloc[0])
    spec = KNOWN_MODELS.get(model, {})
    path = args.tokenizer or spec.get("path")
    if path is None:
        raise SystemExit(f"no tokenizer known for model {model!r}; pass --tokenizer")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True,
                                        revision=None if args.tokenizer else spec.get("revision"))
    if args.task:
        from c3.data.tasks import format_documents, load_task
        docs = format_documents(args.task, load_task(args.task, args.task_jsonl), tok)
        x0 = {d.doc_id: d.ids for d in docs}
    else:
        if not args.docs:
            raise SystemExit("the token-type control needs --docs (the corpus) or --task")
        from c3.data.fineweb import tokenize_documents
        x0 = dict(tokenize_documents(args.docs, tok, args.max_tokens))
    table = C.token_class_table(tok)
    return table[C.slot_token_ids(df, x0)]


# --------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------- #

def frames(df, seed):
    pos, pools = position_and_pool_frames(df, pool_sizes=POOL_SIZES, n_pools_per_size=N_POOLS,
                                          seed=seed, retain=1, min_state_size=max(POOL_SIZES))
    return pos, pools


def control_rows(arm, df, fit_docs, est_docs, codes, counts, args, classes=None) -> tuple[list, dict]:
    pos, pools = frames(df, args.seed)
    fit_mask = np.isin(pos.doc_id, fit_docs)
    rep_mask = np.isin(pos.doc_id, est_docs)
    n_docs = int(np.unique(pos.doc_id[rep_mask]).size)
    rows, extra = [], {}

    delta = winner_gap(pools, pos, pos.conf, pos.agree, codes, counts, rep_mask)
    field_pr = C.preregistered_field(pools, pos, fit_mask, rep_mask, codes, counts, args.knots)
    extra["baseline"] = {"E1": C._est(*delta.contrast(HI, LO)),
                         "E2_preregistered": C._est(*field_pr.contrast(HI, LO)),
                         "delta": delta.to_records("delta")}

    def covariate_rows(name, X, note):
        ce = C.covariate_control(pools, pos, X, fit_mask, rep_mask, codes, counts, args.knots)
        out = [C.ablation_row(name, "E1_contrast", arm, ce.delta.contrast(HI, LO),
                              ce.delta_adj.contrast(HI, LO), n_docs,
                              "with = Delta(m) - E_m[pi(l) - pi(l, x)]; " + note)]
        if ce.field0 is not None:
            out.append(C.ablation_row(name, "E2_field_contrast", arm, ce.field0.contrast(HI, LO),
                                      ce.field_x.contrast(HI, LO), n_docs,
                                      "logistic calibrators, x in both; " + note))
        extra[name] = {"fits": ce.fits, "control_term": ce.control_term.to_records("term")}
        return out

    if "position" in args.controls:
        log_freq = np.load(args.log_freq) if args.log_freq else None
        X, names = C.position_covariates(df, log_freq=log_freq)
        rows += covariate_rows("position", X, "x = " + ", ".join(names) +
                               ("" if log_freq is not None else " (PC-Sampler weight: position-only)"))
        strata = C.position_strata(df)
        r = stratified_row("position_standardised", arm, strata, delta, pools, pos, rep_mask,
                           codes, counts, n_docs, "strata: distance to unmasked (1, 2, 3-4, 5+) "
                           "x absolute-position tercile")
        if r is not None:
            rows.append(r[0])
            extra["position_strata"] = r[1]

    if "token_type" in args.controls and classes is not None:
        X, names = C.token_covariates(classes)
        rows += covariate_rows("token_type", X, "x = " + ", ".join(names) + " (true slot token)")
        # The split: Delta(m) within each class, and the class-standardised contrast.
        groups, shares = {}, {}
        w_all = np.zeros(len(C.TOKEN_CLASSES))
        for ci, cname in enumerate(C.TOKEN_CLASSES):
            sel = rep_mask & (classes == ci)
            if not sel.any():
                continue
            try:
                g = winner_gap(pools, pos, pos.conf, pos.agree, codes, counts, sel)
                g.contrast(HI, LO)
            except (KeyError, ValueError):
                continue
            groups[cname] = g
            w_all[ci] = float(sel.sum())
            rows.append(C.ablation_row(f"token_type_{cname}_only", "E1_contrast", arm,
                                       delta.contrast(HI, LO), g.contrast(HI, LO), n_docs,
                                       f"with = winners whose true token is {cname}"))
        if groups:
            tot = sum(w_all[C.TOKEN_CLASSES.index(c)] for c in groups)
            shares = {c: w_all[C.TOKEN_CLASSES.index(c)] / tot for c in groups}
            rows.append(C.ablation_row("token_type_standardised", "E1_contrast", arm,
                                       delta.contrast(HI, LO),
                                       C.standardised_contrast(groups, shares), n_docs,
                                       "with = sum_c share_c x contrast_c, shares from m = 1"))
            extra["token_type_shares"] = shares

    if "temperature" in args.controls:
        tr = pos.select(fit_mask)
        from c3.calibrators import pool_weights
        w = pool_weights(tr)
        extra["temperature"] = {"bins": C.TEMPERATURE_EDGES.tolist()}
        i = int(np.flatnonzero(delta.levels == HI)[0])
        for kind in args.temperature:
            if kind == "softmax" and not {"topk_p", "p_true"} <= set(df.columns):
                extra["temperature"]["softmax"] = {"skipped": "no top-k probabilities in this trace"}
                continue
            kw = {}
            if kind == "softmax":
                topk = np.stack(df["topk_p"].to_numpy())
                pt = df["p_true"].to_numpy(np.float64)
                vocab = int(args.vocab or _vocab_of(df))
                T = C.fit_temperatures("softmax", tr.mask_rate, w, topk_p=topk[fit_mask],
                                       p_true=pt[fit_mask], vocab=vocab)
                conf_t, ell_t = C.apply_temperatures("softmax", T, pos.mask_rate, topk_p=topk,
                                                     p_true=pt, vocab=vocab)
                kw["vocab"] = vocab
            else:
                T = C.fit_temperatures("logodds", tr.mask_rate, w, ell=tr.ell, agree=tr.agree)
                conf_t, ell_t = C.apply_temperatures("logodds", T, pos.mask_rate, ell=pos.ell)
            # Same pool memberships (they never look at ell), re-ranked on the tempered ell.
            pools_t = C.reranked_pools(pos, ell_t, args.seed)
            delta_t = winner_gap(pools_t, pos, conf_t, pos.agree, codes, counts, rep_mask)
            name = f"temperature_{kind}"
            note = f"{kind} temperature per mask-rate bin, fitted on the fit half"
            rows.append(C.ablation_row(name, "E1_contrast", arm, delta.contrast(HI, LO),
                                       delta_t.contrast(HI, LO), n_docs, note))
            rows.append(C.ablation_row(name, f"Delta({HI})", arm,
                                       (float(delta.point[i]), delta.draws[:, i]),
                                       (float(delta_t.point[i]), delta_t.draws[:, i]), n_docs, note))
            extra["temperature"][kind] = {"T": T.tolist(), **kw}

    if "statistic" in args.controls:
        extra["statistic"] = {}
        i = int(np.flatnonzero(delta.levels == HI)[0])
        for stat in args.statistics:
            if stat not in df.columns:
                extra["statistic"][stat] = {"skipped": f"no {stat} column in this trace"}
                continue
            se = C.statistic_control(pools, pos, C.rank_score(df, stat), fit_mask, rep_mask,
                                     codes, counts, args.seed, args.knots)
            name = f"statistic_{stat}"
            note = (f"pools won by the {'highest margin' if stat == 'margin' else 'lowest entropy'}"
                    " instead of the highest ell (same pools); gap = own confidence - correct")
            rows.append(C.ablation_row(name, "E1_contrast", arm, delta.contrast(HI, LO),
                                       se.delta.contrast(HI, LO), n_docs, note))
            rows.append(C.ablation_row(name, f"Delta({HI})", arm,
                                       (float(delta.point[i]), delta.draws[:, i]),
                                       (float(se.delta.point[i]), se.delta.draws[:, i]), n_docs, note))
            rows.append(C.ablation_row(name, "E2_field_contrast", arm, field_pr.contrast(HI, LO),
                                       se.field.contrast(HI, LO), n_docs,
                                       note + "; pre-registered calibrators"))
            extra["statistic"][stat] = {"winner_agreement": se.agreement,
                                        "delta": se.delta.to_records("delta")}
    return rows, extra


def stratified_row(name, arm, strata, delta, pools, pos, rep_mask, codes, counts, n_docs, note):
    """E1 contrast standardised over strata, shares fixed at their m = 1 (all-position) mass.

    A stratum with no winners at the largest pool size cannot contribute a
    contrast; it is left out and the shares of the others renormalised (recorded).
    """
    groups, mass = {}, {}
    for s in np.unique(strata[rep_mask]):
        sel = rep_mask & (strata == s)
        try:
            g = winner_gap(pools, pos, pos.conf, pos.agree, codes, counts, sel)
            g.contrast(HI, LO)
        except (KeyError, ValueError):
            continue
        groups[int(s)] = g
        mass[int(s)] = float(sel.sum())
    if not groups:
        return None
    tot = sum(mass.values())
    shares = {s: m / tot for s, m in mass.items()}
    covered = tot / float(rep_mask.sum())
    row = C.ablation_row(name, "E1_contrast", arm, delta.contrast(HI, LO),
                         C.standardised_contrast(groups, shares), n_docs,
                         f"{note}; {len(groups)} strata covering {covered:.3f} of positions")
    return row, {"shares": {str(k): v for k, v in shares.items()}, "coverage": covered}


def _vocab_of(df) -> int:
    from c3.traces.models import KNOWN_MODELS
    model = str(df["model"].iloc[0])
    return {"llada-8b-base": 126464, "llada-8b-instruct": 126464, "llada-1.5": 126464,
            "dream-7b-base": 152064, "dream-7b-instruct": 152064}.get(model) or \
        int(KNOWN_MODELS.get(model, {}).get("vocab_size", 0)) or 50258


# --------------------------------------------------------------------------- #
# precision
# --------------------------------------------------------------------------- #

def precision_section(df_a, df_b, args):
    """fp16 (a) vs 4-bit (b) on their common documents; rows and the paired logit shift."""
    from c3.regression import fit_and_report

    common = np.intersect1d(df_a["doc_id"].unique(), df_b["doc_id"].unique())
    fit_docs, est_docs, scope = halves(common, args.split, args.half, args.manifest, args.seed)
    codes = np.sort(est_docs)
    counts = doc_boot_counts(codes.size, args.n_boot, seed=args.seed, label="t8-precision")
    rows, extra = [], {"scope": scope, "n_common_docs": int(common.size),
                       "n_fit_docs": int(fit_docs.size), "n_report_docs": int(est_docs.size)}

    # Uniform masks are seeded by document, so both precisions scored the same states.
    ua = df_a[(df_a.mask_kind == "uniform") & df_a.doc_id.isin(est_docs)]
    ub = df_b[(df_b.mask_kind == "uniform") & df_b.doc_id.isin(est_docs)]
    j = ua.merge(ub, on=["doc_id", "state_idx", "position"], suffixes=("_a", "_b"))
    if len(j):
        di = np.searchsorted(codes, j["doc_id"].to_numpy())
        stats = {"ell_shift": j["ell_b"] - j["ell_a"],
                 "abs_ell_shift": (j["ell_b"] - j["ell_a"]).abs(),
                 "top1_agreement": (j["top1_a"] == j["top1_b"]).astype(float),
                 "agree_change": j["agree_b"] - j["agree_a"]}
        extra["paired_uniform"] = {"n_positions": int(len(j))}
        for k, v in stats.items():
            g = grouped_doc_mean(v.to_numpy(np.float64), di, codes.size, np.zeros(len(j)), counts)
            extra["paired_uniform"][k] = C._est(float(g.point[0]), g.draws[:, 0])
        extra["paired_uniform"]["ell_corr"] = float(np.corrcoef(j["ell_a"], j["ell_b"])[0, 1])

    la, lb = arm_labels(df_a), arm_labels(df_b)
    arms = sorted(set(la) & set(lb))
    if args.arms:
        arms = [x for x in arms if x in args.arms]
    for arm in arms:
        out = {}
        for name, df, lab in (("a", df_a, la), ("b", df_b, lb)):
            sub = df[(lab == arm) & df.doc_id.isin(common).to_numpy()].reset_index(drop=True)
            pos, pools = frames(sub, args.seed)
            fit_mask, rep_mask = np.isin(pos.doc_id, fit_docs), np.isin(pos.doc_id, est_docs)
            out[name] = {
                "delta": winner_gap(pools, pos, pos.conf, pos.agree, codes, counts, rep_mask),
                "field": C.preregistered_field(pools, pos, fit_mask, rep_mask, codes, counts,
                                               args.knots)}
            if args.eq_boot:
                win = pools.select(np.isin(pools.doc_id, np.concatenate([fit_docs, est_docs])))
                try:
                    est = fit_and_report(win.winners(), "eq2", n_knots=args.knots,
                                         n_boot=args.eq_boot, seed=args.seed, n_jobs=args.n_jobs,
                                         cross_fit=True, fe=False, fit_docs=fit_docs,
                                         ridge_beta=1e-6)
                    out[name]["beta_z"] = est["beta_z"]
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    extra.setdefault("errors", []).append(f"{arm} {name} beta_z: {exc}")
        n = int(codes.size)
        note = "without = fp16, with = 4-bit, same documents"
        rows.append(C.ablation_row("precision", "E1_contrast", arm, out["a"]["delta"].contrast(HI, LO),
                                   out["b"]["delta"].contrast(HI, LO), n, note))
        rows.append(C.ablation_row("precision", "E2_field_contrast", arm,
                                   out["a"]["field"].contrast(HI, LO),
                                   out["b"]["field"].contrast(HI, LO), n, note))
        if "beta_z" in out["a"] and "beta_z" in out["b"]:
            ea, eb = out["a"]["beta_z"], out["b"]["beta_z"]
            if ea.boot is not None and eb.boot is not None and ea.boot.size == eb.boot.size:
                rows.append(C.ablation_row("precision", "beta_z_noFE", arm, (ea.value, ea.boot),
                                           (eb.value, eb.boot), n,
                                           note + "; paired resamples (same seed, same documents)"))
    return rows, extra


# --------------------------------------------------------------------------- #

def write_macros(path: Path, prefix: str, rows: list[dict], header: str) -> None:
    lines = ["% Generated by scripts/run_controls.py -- do not edit by hand.", f"% {header}"]
    eff = {"E1_contrast": "EOne", "E2_field_contrast": "ETwo", "beta_z_noFE": "BetaZ",
           f"Delta({HI})": "GapThirtyTwo"}
    seen = set()
    for r in rows:
        ctrl = "".join(w.title() for w in r["control"].split("_"))
        base = f"{prefix}{ctrl}{eff.get(r['effect'], 'X')}{arm_tag(r['arm'])}"
        if base in seen or any(ch.isdigit() for ch in base):
            continue
        seen.add(base)
        for part in ("without", "with", "difference"):
            e = r[part]
            nm = base + part.title()
            for suffix, v in (("", e["value"]), ("Lo", e["ci"][0]), ("Hi", e["ci"][1])):
                if v is not None and np.isfinite(v):
                    lines.append(f"\\newcommand{{\\{nm}{suffix}}}{{{v:.4f}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--precision-trace", default=None, help="the 4-bit trace for --controls precision")
    ap.add_argument("--controls", nargs="+",
                    default=["position", "token_type", "temperature"],
                    choices=["position", "token_type", "temperature", "precision", "statistic"])
    ap.add_argument("--statistics", nargs="+", choices=list(C.RANK_STATISTICS),
                    default=list(C.RANK_STATISTICS),
                    help="for --controls statistic (T13): rank pools by these instead of ell")
    ap.add_argument("--split", choices=["half", "none"], default="half")
    ap.add_argument("--half", choices=["fit", "report"], default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--exclude-docs", default=None)
    ap.add_argument("--arms", nargs="+", default=None, help="arm labels (default: all)")
    ap.add_argument("--temperature", nargs="+", choices=["logodds", "softmax"],
                    default=["logodds", "softmax"],
                    help="temperature kinds, each its own row: logodds (exact, sigma(ell/T), the "
                         "primary) and softmax (top-20 + an even tail; a sensitivity row)")
    ap.add_argument("--vocab", type=int, default=None, help="vocabulary size for softmax temperature")
    ap.add_argument("--docs", default=None, help="the corpus, for the true tokens (token type)")
    ap.add_argument("--task", choices=["gsm8k", "mbpp"], default=None)
    ap.add_argument("--task-jsonl", default=None)
    ap.add_argument("--tokenizer", default=None, help="tokenizer path/id (default: the trace's model)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--log-freq", default=None,
                    help="log unigram frequencies (.npy) for PC-Sampler's full weight")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--eq-boot", type=int, default=0, help="resamples for beta_z (precision; 0 = skip)")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--knots", type=int, default=DEFAULT_KNOTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--macro-prefix", default="TEight")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    extra_cols = (["topk_p", "p_true"] if "temperature" in a.controls and "softmax" in a.temperature
                  else [])
    if "statistic" in a.controls:
        extra_cols += ["margin", "entropy"]
    df, meta = load(a.traces, extra_cols, arms=a.arms)
    if a.exclude_docs:
        from c3.data.contamination import excluded_doc_ids
        drop = excluded_doc_ids(a.exclude_docs)
        df = df[~df["doc_id"].isin(drop)].reset_index(drop=True)
        meta["excluded_docs"] = {"source": a.exclude_docs, "n_listed": len(drop)}
    codes_all = np.unique(df["doc_id"].to_numpy(np.int64))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"input": meta, "config": {k: (v if not isinstance(v, Path) else str(v))
                                        for k, v in vars(a).items()}, "rows": [], "arms": {}}

    other = [c for c in a.controls if c != "precision"]
    if other:
        fit_docs, est_docs, scope = halves(codes_all, a.split, a.half, a.manifest, a.seed)
        print(f"[t8] {scope}: {fit_docs.size} fit / {est_docs.size} estimation documents", flush=True)
        result["scope"] = scope
        codes = np.sort(est_docs)
        counts = doc_boot_counts(codes.size, a.n_boot, seed=a.seed, label="t8-controls")
        df = df[df["doc_id"].isin(np.concatenate([fit_docs, est_docs]))].reset_index(drop=True)
        labels = arm_labels(df)
        arms = sorted(set(labels)) if not a.arms else [x for x in sorted(set(labels)) if x in a.arms]
        for arm in arms:
            sub = df[labels == arm].reset_index(drop=True)
            print(f"[t8] {arm}: {len(sub)} rows", flush=True)
            classes = slot_classes(sub, a, meta) if "token_type" in a.controls else None
            try:
                rows, extra = control_rows(arm, sub, fit_docs, est_docs, codes, counts, a, classes)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                result["arms"][arm] = {"error": str(exc)}
                print(f"[t8]   failed: {exc}", flush=True)
                continue
            result["rows"] += rows
            result["arms"][arm] = extra
            for r in rows:
                w, x = r["without"], r["with"]
                print(f"[t8]   {r['control']:24s} {r['effect']:18s} without {w['value']:+.4f} "
                      f"with {x['value']:+.4f}  diff {r['difference']['value']:+.4f} "
                      f"[{r['difference']['ci'][0]:+.4f}, {r['difference']['ci'][1]:+.4f}]", flush=True)

    if "precision" in a.controls:
        if not a.precision_trace:
            raise SystemExit("--controls precision needs --precision-trace")
        df_a, _ = load(a.traces, ["top1"])
        df_b, meta_b = load(a.precision_trace, ["top1"])
        rows, extra = precision_section(df_a, df_b, a)
        result["rows"] += rows
        result["precision"] = {**extra, "input_4bit": meta_b}

    result["provenance"] = provenance()
    result["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    (out / "ablation.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    (out / "ablation.tex").write_text(C.latex_table(result["rows"]), encoding="utf-8")
    write_macros(out / "macros.tex", a.macro_prefix, result["rows"],
                 f"source: {meta['source']}  commit: {result['provenance']['git_commit']}  "
                 f"scope: {result.get('scope', result.get('precision', {}).get('scope'))}")
    print(f"[t8] {len(result['rows'])} rows -> {out / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
