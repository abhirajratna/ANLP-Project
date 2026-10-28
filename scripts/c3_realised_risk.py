#!/usr/bin/env python3
"""Realised risk of C3a against the raw 1 - alpha rule on report-half decoder states (EXPLORATORY).

    python scripts/c3_realised_risk.py --data-dir <dir holding c1-llada-instruct-{gsm8k,mbpp-v2}/>

C3a commits every eligible position with c_hat >= 1 - alpha; the raw rule (Fast-dLLM's threshold
at tau = 1 - alpha) commits every one with conf >= 1 - alpha.  If c_hat is calibrated the first
should make per-token error <= alpha on the tokens it commits; the second only if conf is.  This
re-plays ONE commit step on each teacher-forced decoder state of the report half (the traces
``c1-llada-instruct-{gsm8k,mbpp-v2}``, arms ``decoder:topk_conf:k{2,4,8}``) and scores the
committed set against the reference token (``agree``).  CPU only, numpy / pandas / pyarrow.

  * Per-token error against the reference, not T12 accuracy.  The states are those a top-k
    decoder visits on the reference answer; committed slots hold the reference, so contexts
    are cleaner than in free-running decoding.  A one-step counterfactual per state, weighted
    by how often top-k visits it.  Hence "exploratory".
  * Eligible set.  The traces have no blocks (they are unblocked decodes of the answer span).
    ``block32`` (primary) keeps the masked positions of the leftmost non-empty 32-block of the
    region, as T12's decode loop does; ``region`` keeps every masked position (the trace's own
    trajectory).  Order in the eligible set: ell descending, ties to the lower position.
  * The calibrators are the frozen T11 fits (``artifacts/c3/llada-8b-instruct_fp16_c3a-<task>.json``);
    c_hat is evaluated unclamped (T12 runs without --clamp) on the state features over ALL masked
    positions of the state, exactly as ``c3.decode.loop.decode`` builds them.
  * The raw rule compares the float32 conf with the float32 of 1 - alpha, as ``Threshold.select``
    does on T12's float32 statistics.  The count of states where sigma(ell) would change n is
    written to the JSON.
  * Only report-half documents (split manifest at ``prereg-v1``, value 1) are read: the filter is
    applied at read time and checked after it.  The sanity panel on the fit half is not built.
  * Before any number is written, the simulation is checked against ``C3a.select`` and
    ``Threshold.select`` on random real states (all alphas, both rules, both eligible sets).

Metrics per task x arm x eligible set x rule x alpha: risk = 1 - correct/committed pooled over
states (also over unforced and forced states separately; a state is forced when no position
passes and one is committed anyway), tokens per state, forced share.  Intervals resample whole
documents (one matrix per task, shared by every estimand, ``c3.stats.doc_boot_counts``).

Both rules commit a prefix of the same ell order, so at equal alpha they differ mostly in how many
tokens they commit.  The main figure therefore plots realised error against tokens per state, tracing
each rule along a dense alpha grid (``DENSE_ALPHAS``; curves only, not a knob), and the JSON holds the
paired difference c3a - raw at fixed tokens-per-state targets (``TARGETS``), interpolated inside every
bootstrap draw.  The reported 9-point alpha grid gives the equal-alpha view (``_alpha`` figures).

Writes ``artifacts/c3/realised_risk_llada-8b-instruct_fp16_<tag>.{json,pdf,png}`` (error vs tokens),
``..._alpha.{pdf,png}``, ``..._arms.pdf``, ``..._alpha_arms.pdf`` and ``..._macros.tex`` (prefix
CThreeTEleven, naming scheme in its header); refuses to overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from c3.calibrators import ContextLogistic, state_field_features  # noqa: E402
from c3.decode.policies import C3a, StepContext, Threshold  # noqa: E402
from c3.metrics import ece_equal_mass  # noqa: E402
from c3.provenance import collect as provenance, sha256_file  # noqa: E402
from c3.stats import doc_boot_counts, percentile_ci  # noqa: E402

BLOCK = 32
ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.12, 0.20, 0.25, 0.40, 0.60)
"""Fixed before any report-half number was read: T12's C3a alphas {.02,.05,.12,.25,.4,.6}
(prereg/t12_cal.json) together with its tau grid as 1 - tau {.01,.05,.1,.2,.4,.6}.  The per-task
C3a alphas of prereg/t12_grid_<task>.json were not committed when this was run, so none is added.
Nothing else may be added after results are seen."""
DENSE_ALPHAS = tuple(np.unique(np.round(np.geomspace(0.005, 0.8, 160), 5)))
"""Only traces the error-vs-tokens curves (a different estimand from the reported grid above,
not a knob: no number is read off it except by matching tokens per state)."""
TARGETS = (1.5, 2.0, 2.5, 3.0, 4.0)
"""Tokens-per-state levels at which C3a and the raw rule are compared (matched speed)."""
ARMS = (2, 4, 8)
PRIMARY_ARM = 4
VARIANTS = ("block32", "region")
RULES = ("c3a", "raw")
N_BOOT = 2000
HF_REVISION = "1ae83e3b3ee2a3c952fe378cef34b39427c1a069"
MANIFEST_REF = "prereg-v1"
TASKS = {"gsm8k": {"dir": "c1-llada-instruct-gsm8k", "word": "Gsm"},
         "mbpp": {"dir": "c1-llada-instruct-mbpp-v2", "word": "Mbpp"}}
COLUMNS = ["doc_id", "state_id", "position", "ell", "conf", "agree", "mask_kind", "policy", "k",
           "mask_rate", "n_masked", "step", "span_start", "span_len"]
STAMP = "exploratory; report half; teacher-forced decoder states"


# --- states ---------------------------------------------------------------------------------
@dataclass
class Rows:
    """Rows of one arm, sorted by (doc, state, position); every row of a state is a masked position."""

    doc_id: np.ndarray
    state_id: np.ndarray
    position: np.ndarray
    span_start: np.ndarray
    span_len: np.ndarray
    n_masked: np.ndarray
    ell: np.ndarray
    conf: np.ndarray          # float32, as T12's statistics
    agree: np.ndarray         # float64 0/1
    sid: np.ndarray           # dense state index per row
    state_start: np.ndarray   # (S + 1,) row offsets

    @property
    def n_states(self) -> int:
        return int(self.state_start.size - 1)

    @property
    def n_rows_state(self) -> np.ndarray:
        return np.diff(self.state_start)

    @property
    def state_doc(self) -> np.ndarray:
        return self.doc_id[self.state_start[:-1]]


def make_rows(doc_id, state_id, position, span_start, span_len, n_masked, ell, conf, agree) -> Rows:
    doc_id, state_id, position = (np.asarray(a, np.int64) for a in (doc_id, state_id, position))
    order = np.lexsort((position, state_id, doc_id))
    d, s = doc_id[order], state_id[order]
    first = np.ones(d.size, bool)
    first[1:] = (d[1:] != d[:-1]) | (s[1:] != s[:-1])
    sid = np.cumsum(first) - 1
    pick = lambda a, dt: np.asarray(a, dt)[order]  # noqa: E731
    return Rows(doc_id=d, state_id=s, position=position[order],
                span_start=pick(span_start, np.int64), span_len=pick(span_len, np.int64),
                n_masked=pick(n_masked, np.int64), ell=pick(ell, np.float64),
                conf=pick(conf, np.float32), agree=pick(agree, np.float64), sid=sid,
                state_start=np.append(np.flatnonzero(first), d.size))


def state_features(rows: Rows) -> dict:
    """The decode-time field features of every state, vectorised ``state_field_features``."""
    start, n = rows.state_start[:-1], rows.n_rows_state
    v = rows.ell[np.lexsort((rows.ell, rows.sid))]
    median = (v[start + (n - 1) // 2] + v[start + n // 2]) / 2.0     # np.median's two-middle mean
    n_masked = rows.n_masked[start]
    return {"median_ell": median,
            "mask_rate": n_masked.astype(np.float64) / rows.span_len[start].astype(np.float64),
            "log_n_masked": np.log(np.maximum(n_masked, 1).astype(np.float64))}


def chat_for(rows: Rows, cal: ContextLogistic) -> np.ndarray:
    """c_hat of every row, unclamped; it does not depend on alpha."""
    feats = state_features(rows)
    return cal.predict_arrays(rows.ell, {n: feats[n][rows.sid] for n in cal.used})


def eligible(rows: Rows, variant: str) -> np.ndarray:
    if variant == "region":
        return np.ones(rows.ell.size, bool)
    if variant != "block32":
        raise ValueError(f"unknown eligible set {variant!r}")
    b = (rows.position - rows.span_start) // BLOCK
    return b == np.minimum.reduceat(b, rows.state_start[:-1])[rows.sid]


# --- the simulation -------------------------------------------------------------------------
@dataclass
class Sim:
    """One commit step per state, for every alpha and both rules; arrays are (alphas, states)."""

    variant: str
    alphas: tuple
    size: np.ndarray            # |E| per state
    order: np.ndarray           # row indices of E, by state then ell descending, position ascending
    start: np.ndarray           # offset of each state in ``order``
    n: dict                     # rule -> committed count
    correct: dict               # rule -> committed rows with agree = 1
    forced: dict                # rule -> no position passed (one committed anyway)
    sigma_diff: np.ndarray      # states whose raw n would change with sigma(ell) in place of conf

    def committed(self, s: int, rule: str, a: int) -> np.ndarray:
        """Row indices committed in state ``s`` (rule, alpha index ``a``), best first."""
        return self.order[self.start[s]:self.start[s] + self.n[rule][a, s]]


def simulate(rows: Rows, chat: np.ndarray, variant: str, alphas=ALPHAS) -> Sim:
    S = rows.n_states
    idx = np.flatnonzero(eligible(rows, variant))
    sid = rows.sid[idx]
    order = idx[np.lexsort((rows.position[idx], -rows.ell[idx], sid))]   # _stable_order, per state
    sid_o = rows.sid[order]
    size = np.bincount(sid_o, minlength=S)
    start = np.cumsum(size) - size
    cum = np.concatenate([[0.0], np.cumsum(rows.agree[order])])
    sigma = 1.0 / (1.0 + np.exp(-rows.ell[order]))
    score = {"c3a": chat[order], "raw": rows.conf[order]}
    sim = Sim(variant, tuple(alphas), size, order, start, {}, {}, {}, np.zeros(len(alphas), np.int64))
    for rule in RULES:
        for table in (sim.n, sim.correct, sim.forced):
            table[rule] = np.zeros((len(alphas), S), np.int64 if table is not sim.forced else bool)
    for a, alpha in enumerate(alphas):
        npass = {}
        for rule in RULES:
            npass[rule] = np.bincount(sid_o, weights=score[rule] >= 1.0 - alpha, minlength=S).astype(np.int64)
            n = np.maximum(1, npass[rule])
            sim.n[rule][a] = n
            sim.correct[rule][a] = (cum[start + n] - cum[start]).astype(np.int64)
            sim.forced[rule][a] = npass[rule] == 0
        by_sigma = np.bincount(sid_o, weights=sigma >= 1.0 - alpha, minlength=S).astype(np.int64)
        sim.sigma_diff[a] = int(np.sum(np.maximum(1, by_sigma) != sim.n["raw"][a]))
    return sim


def verify_against_policies(rows: Rows, chat: np.ndarray, cal, alphas=ALPHAS, n_states: int = 300,
                            seed: int = 0) -> int:
    """The simulation against ``C3a.select`` and ``Threshold.select`` on random states, both
    eligible sets, every alpha: counts, committed sets and forced flags.  Raises on mismatch."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(rows.n_states, size=min(n_states, rows.n_states), replace=False)
    for variant in VARIANTS:
        sim = simulate(rows, chat, variant, alphas)
        for s in pick:
            sl = slice(rows.state_start[s], rows.state_start[s + 1])
            keep = eligible(rows, variant)[sl]
            pos = rows.position[sl][keep]
            stats = {"ell": rows.ell[sl][keep], "conf": rows.conf[sl][keep]}
            state = state_field_features(rows.ell[sl], int(rows.n_masked[sl][0]), int(rows.span_len[sl][0]))
            ctx = StepContext(step=0, positions=pos, gen_start=int(rows.span_start[sl][0]), state=state)
            for a, alpha in enumerate(alphas):
                for rule, policy in (("c3a", C3a(alpha, cal)), ("raw", Threshold(tau=1.0 - alpha))):
                    order, n = policy.select(stats, ctx)
                    score = policy.calibrated(stats, ctx)[0] if rule == "c3a" else stats["conf"]
                    if sim.forced[rule][a, s] != (not np.any(score >= 1.0 - alpha)):
                        raise AssertionError(f"{variant} {rule} alpha={alpha} state {s}: wrong forced flag")
                    want = np.sort(pos[order[:n]])
                    got = np.sort(rows.position[sim.committed(s, rule, a)])
                    if sim.n[rule][a, s] != n or not np.array_equal(got, want):
                        raise AssertionError(f"{variant} {rule} alpha={alpha} state {s}: simulated "
                                             f"{got.tolist()} but the policy commits {want.tolist()}")
    return int(pick.size)


def verify_features(rows: Rows, n_states: int = 200, seed: int = 0) -> int:
    """The vectorised state features against ``state_field_features`` on random states."""
    feats = state_features(rows)
    rng = np.random.default_rng(seed)
    pick = rng.choice(rows.n_states, size=min(n_states, rows.n_states), replace=False)
    for s in pick:
        sl = slice(rows.state_start[s], rows.state_start[s + 1])
        ref = state_field_features(rows.ell[sl], int(rows.n_masked[sl][0]), int(rows.span_len[sl][0]))
        for name, v in ref.items():
            if not np.isclose(feats[name][s], v, rtol=0, atol=1e-12):
                raise AssertionError(f"state {s}: {name} {feats[name][s]!r} != state_field_features {v!r}")
    return int(pick.size)


# --- estimands ------------------------------------------------------------------------------
def ratio_ci(num: np.ndarray, den: np.ndarray, counts: np.ndarray) -> dict:
    """Pooled ratio sum(num)/sum(den) over documents and its document bootstrap (shared counts)."""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    C = counts.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (C @ num) / (C @ den)
        value = num.sum() / den.sum()
    return {"value": float(value), "ci": percentile_ci(draws), "draws": draws}


def diff_ci(a: dict, b: dict) -> dict:
    draws = a["draws"] - b["draws"]
    return {"value": a["value"] - b["value"], "ci": percentile_ci(draws), "draws": draws}


def pub(e: dict) -> dict:
    """An estimate as written to the JSON: value and 95% interval (NaN -> null)."""
    nn = lambda x: None if not np.isfinite(x) else float(x)  # noqa: E731
    return {"value": nn(e["value"]), "ci_low": nn(e["ci"][0]), "ci_high": nn(e["ci"][1])}


def over_alpha(e: dict, alpha: float) -> dict:
    return pub({"value": e["value"] / alpha, "ci": tuple(x / alpha for x in e["ci"])})


def curves(rows: Rows, sim: Sim, state_doc: np.ndarray, counts: np.ndarray) -> dict:
    """Per rule and alpha: risk (all / unforced / forced), tokens per state, forced share, and the
    paired risk difference c3a - raw; document-bootstrap intervals throughout."""
    D = counts.shape[1]
    per_doc = lambda w: np.bincount(state_doc, weights=w, minlength=D)  # noqa: E731
    states = per_doc(np.ones(rows.n_states))
    out = {"singleton_share": pub(ratio_ci(per_doc((sim.size == 1).astype(float)), states, counts)),
           "mean_eligible_size": float(sim.size.mean()),
           "sigma_changes_raw_n_states": [int(x) for x in sim.sigma_diff],
           "notes": {"unforced": "risk_unforced is the error of the commits in states where at least one "
                                 "position passed the bar; the unforced subsets differ between the two rules "
                                 "(the raw rule has fewer forced states at the same alpha), so unforced risks "
                                 "are not a like-for-like comparison.  unforced_over_alpha = risk_unforced / "
                                 "alpha: under 1 means the nominal bound holds on those commits."},
           "rules": {r: [] for r in RULES}, "paired_risk_diff_c3a_minus_raw": []}
    risk = {r: [] for r in RULES}
    for a, alpha in enumerate(sim.alphas):
        for rule in RULES:
            n, c, f = sim.n[rule][a], sim.correct[rule][a], sim.forced[rule][a]
            err = (n - c).astype(float)
            r_all = ratio_ci(per_doc(err), per_doc(n.astype(float)), counts)
            risk[rule].append(r_all)
            rec = {"alpha": alpha, "risk": pub(r_all), "n_committed": int(n.sum()),
                   "risk_unforced": pub(ratio_ci(per_doc(err * ~f), per_doc(n * ~f * 1.0), counts)),
                   "unforced_over_alpha": over_alpha(
                       ratio_ci(per_doc(err * ~f), per_doc(n * ~f * 1.0), counts), alpha),
                   "risk_forced": pub(ratio_ci(per_doc(err * f), per_doc(n * f * 1.0), counts)),
                   "tokens_per_state": pub(ratio_ci(per_doc(n.astype(float)), states, counts)),
                   "forced_share": pub(ratio_ci(per_doc(f * 1.0), states, counts))}
            out["rules"][rule].append(rec)
        out["paired_risk_diff_c3a_minus_raw"].append(
            {"alpha": alpha, **pub(diff_ci(risk["c3a"][a], risk["raw"][a]))})
    return out


def interp_rows(T: float, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Row-wise linear interpolation of Y at X = T (X increasing along a row); NaN outside X's range."""
    out = np.full(X.shape[0], np.nan)
    for i in range(X.shape[0]):
        if X[i, 0] <= T <= X[i, -1]:
            out[i] = np.interp(T, X[i], Y[i])
    return out


def matched_estimate(T: float, X: np.ndarray, Y: np.ndarray) -> dict:
    """Y at X = T.  Row 0 of X, Y is the point curve, rows 1.. one bootstrap draw each: the
    interpolation is redone inside every draw (each has its own tokens-per-state axis)."""
    v = interp_rows(T, X, Y)
    draws = v[1:]
    return {"value": float(v[0]), "ci": percentile_ci(draws), "draws": draws,
            "share_in_range": float(np.isfinite(draws).mean())}


def matched_curves(rows: Rows, chat: np.ndarray, variant: str, state_doc: np.ndarray,
                   counts: np.ndarray, alphas=DENSE_ALPHAS, targets=TARGETS) -> dict:
    """Realised error against tokens per state for each rule along a dense alpha grid, and the
    paired difference c3a - raw at fixed tokens-per-state targets (same document draws)."""
    sim = simulate(rows, chat, variant, alphas)
    D = counts.shape[1]
    per_doc = lambda w: np.bincount(state_doc, weights=w, minlength=D)  # noqa: E731
    W = np.vstack([np.ones((1, D)), counts.astype(np.float64)])        # row 0: the point estimate
    states = W @ per_doc(np.ones(rows.n_states))
    X, Y = {}, {}
    for rule in RULES:
        n = sim.n[rule].astype(np.float64)
        N = np.stack([per_doc(x) for x in n])                           # (alphas, docs)
        E = np.stack([per_doc(x) for x in n - sim.correct[rule]])
        X[rule] = (W @ N.T) / states[:, None]
        Y[rule] = (W @ E.T) / (W @ N.T)
    curve = {}
    for rule in RULES:
        lo_x, hi_x = np.quantile(X[rule][1:], [0.025, 0.975], axis=0)
        lo_y, hi_y = np.quantile(Y[rule][1:], [0.025, 0.975], axis=0)
        curve[rule] = [{"alpha": float(a),
                        "tokens_per_state": {"value": float(X[rule][0, i]), "ci_low": float(lo_x[i]),
                                             "ci_high": float(hi_x[i])},
                        "risk": {"value": float(Y[rule][0, i]), "ci_low": float(lo_y[i]),
                                 "ci_high": float(hi_y[i])}} for i, a in enumerate(alphas)]
    matched = []
    for T in targets:
        est = {r: matched_estimate(T, X[r], Y[r]) for r in RULES}
        ok = all(np.isfinite(e["value"]) for e in est.values())
        rec = {"T": T, "in_range": ok,
               **{r: {"risk": pub(est[r]), "share_draws_in_range": est[r]["share_in_range"]} for r in RULES}}
        rec["diff"] = (pub(diff_ci(est["c3a"], est["raw"])) if ok
                       else pub({"value": np.nan, "ci": (np.nan, np.nan)}))
        matched.append(rec)
    return {"alphas": [float(a) for a in alphas], "curve": curve, "matched": matched,
            "note": "dense alpha grid: traces the curves only; matched = risk interpolated at fixed tokens/state "
                    "inside every bootstrap draw (draws whose tokens range misses T are dropped from the CI)"}


def monotone_share(rows: Rows, chat: np.ndarray) -> float:
    """Share of states in which c_hat is non-decreasing in ell (as in the fit's diagnostic)."""
    order = np.lexsort((rows.ell, rows.sid))
    s, c = rows.sid[order], chat[order]
    drop = (np.diff(c) < -1e-12) & (s[1:] == s[:-1])
    return float(1.0 - np.unique(s[1:][drop]).size / rows.n_states)


def outside_flags(rows: Rows, cal: ContextLogistic) -> dict:
    """Per feature: which states lie outside the calibrator's fit-half range."""
    feats = state_features(rows)
    return {n: (feats[n] < r["min"]) | (feats[n] > r["max"])
            for n, r in (cal.feature_ranges or {}).items() if n in feats}


# --- loading and the report-half guard ------------------------------------------------------
def report_docs(manifest: dict) -> list:
    docs = sorted(int(d) for d, h in manifest.items() if int(h) == 1)
    if not docs:
        raise SystemExit("the split manifest has no report-half documents")
    return docs


def check_report_only(doc_ids, docs) -> None:
    leaked = set(np.unique(np.asarray(doc_ids)).tolist()) - set(int(d) for d in docs)
    if leaked:                                       # the read filter is the guard; check it
        raise SystemExit(f"{len(leaked)} non-report documents read; refusing")


def read_report_rows(parts, report_docs, ks=ARMS):
    """The decoder-state rows (topk_conf, k in ``ks``) of report-half documents, part by part."""
    import pandas as pd

    from c3.assemble import read_trace_table

    ids = [int(d) for d in report_docs]
    filters = [("mask_kind", "=", "decoder"), ("policy", "=", "topk_conf"),
               ("k", "in", [int(k) for k in ks]), ("doc_id", "in", ids)]
    df = pd.concat([read_trace_table(p, columns=COLUMNS, filters=filters).to_pandas() for p in parts],
                   ignore_index=True)
    if not len(df):
        raise SystemExit("no report-half decoder rows read")
    check_report_only(df["doc_id"].to_numpy(), ids)
    return df


def lf_sha256(path) -> str:
    """sha256 of a text file as git stores it (LF): a CRLF checkout hashes differently."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pinned_hashes() -> dict:
    out = {}
    for line in (ROOT / "scripts" / "jarvis_t12.inputs.sha256").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, p = line.split()
            out[p] = h
    return out


def git_manifest(task: str) -> tuple[dict, str]:
    """The split manifest at the pre-registration tag, read through git (LF), with its sha256."""
    raw = subprocess.run(["git", "show", f"{MANIFEST_REF}:artifacts/split_manifest_{task}.json"],
                         cwd=ROOT, capture_output=True, check=True).stdout
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


# --- one task -------------------------------------------------------------------------------
def run_task(task: str, data_dir: Path, art_dir: Path, n_boot: int, seed: int) -> dict:
    from c3.traces.shards import dataset_files

    pins = pinned_hashes()
    stem = data_dir / TASKS[task]["dir"] / f"{TASKS[task]['dir']}.dataset.json"
    cal_path = art_dir / f"llada-8b-instruct_fp16_c3a-{task}.json"
    cal_rec = json.loads(cal_path.read_text())
    cal = ContextLogistic.from_dict(cal_rec["calibrator"])
    need(lf_sha256(cal_path) == pins[f"artifacts/c3/{cal_path.name}"],
         f"{cal_path.name} is not the calibrator pinned in scripts/jarvis_t12.inputs.sha256")
    need(sha256_file(stem) == cal_rec["traces"]["sha256"],
         f"{stem.name} is not the trace the calibrator was fitted on")
    manifest, man_sha = git_manifest(task)
    need(man_sha == pins[f"artifacts/split_manifest_{task}.json"], f"{task} manifest differs from the pin")
    docs = report_docs(manifest)
    parts = dataset_files(stem, verify=True)
    print(f"[{task}] {len(parts)} parts verified; {len(docs)} report-half documents", flush=True)

    df = read_report_rows(parts, docs)
    arms = {}
    for k in ARMS:
        d = df[df["k"] == k]
        rows = make_rows(*(d[c].to_numpy() for c in ("doc_id", "state_id", "position", "span_start",
                                                      "span_len", "n_masked", "ell", "conf", "agree")))
        need(np.array_equal(rows.n_masked[rows.state_start[:-1]], rows.n_rows_state),
             f"{task} k{k}: n_masked is not the stored row count of a state")
        need(np.allclose(d.sort_values(["doc_id", "state_id", "position"])["mask_rate"].to_numpy(),
                         rows.n_masked / rows.span_len, rtol=0, atol=1e-9),
             f"{task} k{k}: mask_rate != n_masked / span_len")
        need(np.array_equal(rows.conf.astype(np.float64), d.sort_values(["doc_id", "state_id", "position"])
                            ["conf"].to_numpy()), f"{task} k{k}: conf is not float32-exact")
        arms[k] = rows
    del df
    universe = np.unique(np.concatenate([r.doc_id for r in arms.values()]))
    counts = doc_boot_counts(universe.size, n_boot, seed, label=f"t11-realised-risk-{task}")
    print(f"[{task}] " + ", ".join(f"k{k}: {r.n_states:,} states / {r.ell.size:,} rows"
                                   for k, r in arms.items()) + f"; {universe.size} documents", flush=True)

    res = {"n_docs": int(universe.size), "n_report_docs_in_manifest": len(docs), "arms": {}, "ece": {},
           "outside_share": {}, "monotone_share": {}, "checked_states": {},
           "inputs": {"calibrator": {"path": f"artifacts/c3/{cal_path.name}", "sha256_lf": lf_sha256(cal_path)},
                      "traces_dataset_json": {"file": stem.name, "sha256": sha256_file(stem)},
                      "split_manifest": {"git": f"{MANIFEST_REF}:artifacts/split_manifest_{task}.json",
                                         "sha256": man_sha}},
           "feature_ranges": cal.feature_ranges}
    pooled = {"chat": [], "sigma": [], "agree": [], "doc": [], "outside": {}}
    for k, rows in arms.items():
        chat = chat_for(rows, cal)
        verify_features(rows, 200, seed)
        res["checked_states"][f"k{k}"] = verify_against_policies(rows, chat, cal, seed=seed)
        state_doc = np.searchsorted(universe, rows.state_doc)
        row_doc = state_doc[rows.sid]
        sigma = 1.0 / (1.0 + np.exp(-rows.ell))
        res["arms"][f"k{k}"] = {"n_states": rows.n_states, "n_rows": int(rows.ell.size),
                                "n_docs": int(np.unique(rows.doc_id).size),
                                **{v: {**curves(rows, simulate(rows, chat, v), state_doc, counts),
                                       "matched_speed": matched_curves(rows, chat, v, state_doc, counts)}
                                   for v in VARIANTS}}
        res["ece"][f"k{k}"] = {"calibrated": ece_equal_mass(chat, rows.agree, 15, row_doc, counts).to_dict(),
                               "raw_sigma_ell": ece_equal_mass(sigma, rows.agree, 15, row_doc, counts).to_dict()}
        res["monotone_share"][f"k{k}"] = monotone_share(rows, chat)
        flags = outside_flags(rows, cal)
        any_out = np.any(list(flags.values()), axis=0)
        res["outside_share"][f"k{k}"] = {"any": float(any_out.mean()),
                                         "per_feature": {n: float(f.mean()) for n, f in flags.items()}}
        for key, v in (("chat", chat), ("sigma", sigma), ("agree", rows.agree), ("doc", row_doc)):
            pooled[key].append(v)
        for n, f in flags.items():
            pooled["outside"].setdefault(n, []).append(f)
        print(f"[{task}] k{k}: verified {res['checked_states'][f'k{k}']} states against the policies; "
              f"OutsideShare {any_out.mean():.4f}; monotone {res['monotone_share'][f'k{k}']:.4f}", flush=True)
    cat = {k: np.concatenate(v) for k, v in pooled.items() if k != "outside"}
    res["ece"]["pooled"] = {
        "calibrated": ece_equal_mass(cat["chat"], cat["agree"], 15, cat["doc"], counts).to_dict(),
        "raw_sigma_ell": ece_equal_mass(cat["sigma"], cat["agree"], 15, cat["doc"], counts).to_dict()}
    out = {n: np.concatenate(f) for n, f in pooled["outside"].items()}
    res["outside_share"]["pooled"] = {"any": float(np.any(list(out.values()), axis=0).mean()),
                                      "per_feature": {n: float(f.mean()) for n, f in out.items()}}
    return res


# --- macros ---------------------------------------------------------------------------------
PREFIX = "CThreeTEleven"
RULE_WORD = {"c3a": "CThreeA", "raw": "Raw"}
ALPHA_WORD = {0.05: "AlphaFivePct", 0.10: "AlphaTenPct", 0.20: "AlphaTwentyPct"}
UNFORCED_ALPHA_WORD = {0.01: "AlphaOnePct", **ALPHA_WORD}
T_WORD = {1.5: "OnePtFive", 2.0: "Two", 2.5: "TwoPtFive", 3.0: "Three", 4.0: "Four"}
VARIANT_WORD = {"block32": "Block", "region": "Region"}


def at(curve: list, alpha: float) -> dict:
    return next(r for r in curve if abs(r["alpha"] - alpha) < 1e-12)


def macros(res: dict, json_name: str) -> str:
    lines = []
    names = set()

    def add(name, body):
        name = PREFIX + name
        if not (name.isascii() and name.isalpha()) or name in names:
            raise ValueError(f"macro name {name!r} is not letters only or is defined twice")
        names.add(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{body}}}")

    for line in ("Generated by scripts/c3_realised_risk.py -- do not edit by hand.",
                 "EXPLORATORY: realised per-token risk of C3a (c_hat >= 1 - alpha) against the raw rule",
                 "(conf >= 1 - alpha) on report-half teacher-forced LLaDA-8B-Instruct decoder states, arm",
                 f"decoder:topk_conf:k{PRIMARY_ARM}, eligible set block32 (leftmost non-empty 32-block).",
                 f"Source: artifacts/c3/{json_name}; all numbers on a 0-1 scale, three decimals.",
                 "  \\CThreeTEleven<Task>Risk<Rule><Alpha>[Lo|Hi]   1 - correct/committed, 95% document-bootstrap",
                 "  \\CThreeTEleven<Task>TokensPerState<Rule><Alpha>   committed tokens per state",
                 "  \\CThreeTEleven<Task>ReportECE<Raw|Cal>   equal-mass ECE of sigma(ell) / of c_hat, all positions",
                 "  \\CThreeTEleven<Task>RiskOutsideShare   share of states with a feature outside the fit range",
                 "  \\CThreeTEleven<Task>UnforcedRisk<Rule><Alpha>[Lo|Hi]   error of the commits in states where a",
                 "      position passed the bar (the unforced subsets differ between the rules);",
                 "  \\CThreeTEleven<Task>UnforcedRatio<Rule><Alpha>[Lo|Hi]   that error divided by alpha",
                 "  \\CThreeTEleven<Task>Matched<Block|Region>Diff<T>[Lo|Hi]   C3a minus raw realised error at equal",
                 "      tokens per state T (interpolated on a dense alpha grid inside every bootstrap draw; Block =",
                 "      leftmost 32-block, Region = whole region);  ...Matched<Block|Region>Risk<Rule><T>  each rule's error",
                 "  Task: Gsm, Mbpp;  Rule: CThreeA, Raw;  Alpha: AlphaFivePct, AlphaTenPct, AlphaTwentyPct",
                 "  (Unforced...: also AlphaOnePct);  T: OnePtFive, Two, TwoPtFive, Three, Four tokens per state"):
        lines.append(f"% {line}")
    k = f"k{PRIMARY_ARM}"
    for task, t in res["tasks"].items():
        w = TASKS[task]["word"]
        for rule in RULES:
            for alpha, aw in ALPHA_WORD.items():
                rec = at(t["arms"][k]["block32"]["rules"][rule], alpha)
                base = f"{w}Risk{RULE_WORD[rule]}{aw}"
                add(base, f"{rec['risk']['value']:.3f}")
                add(base + "Lo", f"{rec['risk']['ci_low']:.3f}")
                add(base + "Hi", f"{rec['risk']['ci_high']:.3f}")
                add(f"{w}TokensPerState{RULE_WORD[rule]}{aw}", f"{rec['tokens_per_state']['value']:.3f}")
        for rule in RULES:
            for alpha, aw in UNFORCED_ALPHA_WORD.items():
                rec = at(t["arms"][k]["block32"]["rules"][rule], alpha)
                for key, word, fmt in (("risk_unforced", "UnforcedRisk", "{:.3f}"),
                                       ("unforced_over_alpha", "UnforcedRatio", "{:.2f}")):
                    base = f"{w}{word}{RULE_WORD[rule]}{aw}"
                    add(base, fmt.format(rec[key]["value"]))
                    add(base + "Lo", fmt.format(rec[key]["ci_low"]))
                    add(base + "Hi", fmt.format(rec[key]["ci_high"]))
        for variant, vw in VARIANT_WORD.items():
            for m in t["arms"][k][variant]["matched_speed"]["matched"]:
                if not m["in_range"]:
                    lines.append(f"% {w} {vw} T={m['T']}: outside the tokens range of one rule; not written")
                    continue
                add(f"{w}Matched{vw}Diff{T_WORD[m['T']]}", f"{m['diff']['value']:+.3f}")
                add(f"{w}Matched{vw}Diff{T_WORD[m['T']]}Lo", f"{m['diff']['ci_low']:+.3f}")
                add(f"{w}Matched{vw}Diff{T_WORD[m['T']]}Hi", f"{m['diff']['ci_high']:+.3f}")
                for rule in RULES:
                    add(f"{w}Matched{vw}Risk{RULE_WORD[rule]}{T_WORD[m['T']]}", f"{m[rule]['risk']['value']:.3f}")
        add(f"{w}ReportECERaw", f"{t['ece'][k]['raw_sigma_ell']['ece']:.3f}")
        add(f"{w}ReportECECal", f"{t['ece'][k]['calibrated']['ece']:.3f}")
        add(f"{w}RiskOutsideShare", f"{t['outside_share'][k]['any']:.3f}")
    return "\n".join(lines) + "\n"


# --- figures --------------------------------------------------------------------------------
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
COLOR = {"c3a": "#0072B2", "raw": "#D55E00"}                 # blue / vermillion: CVD-safe pair
LABEL = {"c3a": "C3a (calibrated c-hat)", "raw": "raw rule (conf)"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.titlepad": 3,
        "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def _tokens_axis(ax) -> None:
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.set_yticklabels(["1", "2", "4", "8", "16"])
    ax.minorticks_off()


def _style(ax, alpha_axis: bool = True) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xscale("log")
    if alpha_axis:
        ax.set_xticks(ALPHAS)
        ax.set_xticklabels([f"{a:g}" for a in ALPHAS], rotation=60)
    else:
        ticks = [1, 1.5, 2, 3, 4, 6, 8, 12]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlim(1, 12)
    ax.minorticks_off()


def _tokens_line(ax, curve, rule, ls="-", band=True, lw=1.2, label=None):
    """Realised error against tokens per state along the dense alpha grid."""
    x = [r["tokens_per_state"]["value"] for r in curve]
    ax.plot(x, [r["risk"]["value"] for r in curve], ls, color=COLOR[rule], lw=lw, label=label)
    if band:
        ax.fill_between(x, [r["risk"]["ci_low"] for r in curve], [r["risk"]["ci_high"] for r in curve],
                        color=COLOR[rule], alpha=0.18, lw=0)


def matched_figure(res: dict, stem: Path) -> list:
    """MAIN figure: realised error against tokens per state (equal speed), one panel per task."""
    plt = _plt()
    k = f"k{PRIMARY_ARM}"
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for j, (task, t) in enumerate(res["tasks"].items()):
        ax = axes[j]
        _style(ax, alpha_axis=False)
        for rule in RULES:
            _tokens_line(ax, t["arms"][k]["block32"]["matched_speed"]["curve"][rule], rule,
                         label=LABEL[rule] + ", block 32")
            _tokens_line(ax, t["arms"][k]["region"]["matched_speed"]["curve"][rule], rule, ls="--", lw=0.9,
                         band=False, label=LABEL[rule] + ", whole region")
        ax.set_title(f"{'GSM8K' if task == 'gsm8k' else 'MBPP'}  (arm k{PRIMARY_ARM}, "
                     f"{t['arms'][k]['n_states']:,} states, {t['n_docs']} docs)", loc="left")
        ax.set_xlabel("committed tokens per state")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("realised error of committed tokens")
    axes[0].legend(frameon=False, loc="upper left", fontsize=5.5)
    fig.suptitle(f"Realised error at equal speed, C3a vs raw rule -- {STAMP}", fontsize=7.5, y=0.995)
    fig.text(0.01, 0.005, "Curves sweep alpha (dense grid); solid = leftmost 32-block, band = 95% document bootstrap; "
             "dashed = whole region. Error vs the reference token, not task accuracy.",
             fontsize=5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    paths = _save(fig, stem, "Realised error vs tokens per state, C3a vs raw (exploratory)")
    plt.close(fig)
    return paths


def matched_arms_figure(res: dict, stem: Path) -> list:
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.2), sharex=True)
    style = {"k2": ":", "k4": "-", "k8": "--"}
    for j, (task, t) in enumerate(res["tasks"].items()):
        for i, variant in enumerate(VARIANTS):
            ax = axes[i, j]
            _style(ax, alpha_axis=False)
            for arm in t["arms"]:
                for rule in RULES:
                    _tokens_line(ax, t["arms"][arm][variant]["matched_speed"]["curve"][rule], rule, ls=style[arm],
                                 band=False, lw=1.0, label=f"{arm}, {LABEL[rule]}" if (i, j) == (0, 0) else None)
            ax.set_title(f"{task}: {variant}", loc="left")
            ax.set_ylim(bottom=0)
        axes[1, j].set_xlabel("committed tokens per state")
        axes[0, 0].set_ylabel("realised error")
        axes[1, 0].set_ylabel("realised error")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=5, loc="upper left")
    fig.suptitle(f"All arms (k2 dotted, k4 solid, k8 dashed) -- {STAMP}", fontsize=7.5, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    paths = _save(fig, stem.parent / f"{stem.name}_arms", "Realised error vs tokens, all arms (exploratory)", png=False)
    plt.close(fig)
    return paths


def _line(ax, curve, key, rule, ls="-", band=False, lw=1.2, label=None):
    x = [r["alpha"] for r in curve]
    y = [r[key]["value"] for r in curve]
    ax.plot(x, y, ls, color=COLOR[rule], lw=lw, marker="o" if ls == "-" else None, ms=2.5, label=label)
    if band:
        ax.fill_between(x, [r[key]["ci_low"] for r in curve], [r[key]["ci_high"] for r in curve],
                        color=COLOR[rule], alpha=0.18, lw=0)


def _save(fig, stem: Path, title: str, png: bool = True) -> list:
    out = []
    for ext in ("pdf", "png") if png else ("pdf",):
        p = stem.parent / f"{stem.name}.{ext}"
        meta = {"Title": title}
        if ext == "pdf":
            meta["CreationDate"] = None
        fig.savefig(p, dpi=300, metadata=meta)
        out.append(p)
    return out


def alpha_figures(res: dict, stem: Path) -> list:
    """The equal-alpha view (secondary): realised error and tokens per state against alpha."""
    plt = _plt()
    k = f"k{PRIMARY_ARM}"
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True)
    for j, (task, t) in enumerate(res["tasks"].items()):
        top, bot = axes[0, j], axes[1, j]
        for ax in (top, bot):
            _style(ax)
        top.plot([ALPHAS[0], ALPHAS[-1]], [ALPHAS[0], ALPHAS[-1]], color=INK2, lw=0.8, ls=":", label="y = x")
        for rule in RULES:
            b32 = t["arms"][k]["block32"]["rules"][rule]
            reg = t["arms"][k]["region"]["rules"][rule]
            _line(top, b32, "risk", rule, band=True, label=LABEL[rule] + ", block 32")
            _line(top, reg, "risk", rule, ls="--", lw=0.9, label=LABEL[rule] + ", whole region")
            _line(bot, b32, "tokens_per_state", rule, band=True)
            _line(bot, reg, "tokens_per_state", rule, ls="--", lw=0.9)
        top.set_title(f"{task.upper() if task == 'gsm8k' else 'MBPP'}  (arm k{PRIMARY_ARM}, "
                      f"{t['arms'][k]['n_states']:,} states, {t['n_docs']} docs)", loc="left")
        top.set_ylabel("realised error (committed)" if j == 0 else "")
        bot.set_ylabel("tokens per state" if j == 0 else "")
        _tokens_axis(bot)
        bot.set_xlabel("alpha (nominal risk, 1 - threshold)")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Realised risk of C3a vs the raw rule at equal alpha -- {STAMP}", fontsize=7.5, y=0.995)
    fig.text(0.01, 0.005, "Solid: eligible set = leftmost 32-block, band = 95% document bootstrap. Dashed: whole "
             "region. Per-token error against the reference token, not task accuracy.", fontsize=5.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    paths = _save(fig, stem.parent / f"{stem.name}_alpha", "Realised risk of C3a vs raw at equal alpha (exploratory)")
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(6.4, 8.6), sharex=True)
    shade = {"k2": 0.55, "k4": 1.0, "k8": 0.75}
    style = {"k2": ":", "k4": "-", "k8": "--"}
    for j, (task, t) in enumerate(res["tasks"].items()):
        for i, (variant, key) in enumerate((("block32", "risk"), ("region", "risk"),
                                            ("block32", "tokens_per_state"), ("region", "tokens_per_state"))):
            ax = axes[i, j]
            _style(ax)
            if key == "risk":
                ax.plot([ALPHAS[0], ALPHAS[-1]], [ALPHAS[0], ALPHAS[-1]], color=INK2, lw=0.8, ls=":")
            else:
                _tokens_axis(ax)
            for arm in t["arms"]:
                for rule in RULES:
                    _line(ax, t["arms"][arm][variant]["rules"][rule], key, rule, ls=style[arm], lw=1.0 * shade[arm],
                          label=f"{arm}, {LABEL[rule]}" if (i, j) == (0, 0) else None)
            ax.set_title(f"{task}: {variant}, {'risk' if key == 'risk' else 'tokens per state'}", loc="left")
        axes[3, j].set_xlabel("alpha")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=5, loc="upper left")
    fig.suptitle(f"All arms (k2 dotted, k4 solid, k8 dashed) -- {STAMP}", fontsize=7.5, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    paths += _save(fig, stem.parent / f"{stem.name}_alpha_arms", "Realised risk, all arms (exploratory)", png=False)
    plt.close(fig)
    return paths


# --- main -----------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="holds c1-llada-instruct-gsm8k/ and c1-llada-instruct-mbpp-v2/ (HF revision "
                         f"{HF_REVISION[:8]})")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--art-dir", type=Path, default=ROOT / "artifacts" / "c3")
    ap.add_argument("--tag", default="v2", help="output version; an existing one is never overwritten")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    stem = a.art_dir / f"realised_risk_llada-8b-instruct_fp16_{a.tag}"
    outs = [stem.with_suffix(".json"), stem.with_suffix(".pdf"), stem.with_suffix(".png"),
            stem.parent / f"{stem.name}_alpha.pdf", stem.parent / f"{stem.name}_alpha.png",
            stem.parent / f"{stem.name}_arms.pdf", stem.parent / f"{stem.name}_alpha_arms.pdf",
            stem.parent / f"{stem.name}_macros.tex"]
    clash = [p.name for p in outs if p.exists()]
    need(not clash, f"refusing to overwrite {clash}; use a new --tag")

    res = {"what": "Realised per-token risk of C3a vs the raw 1 - alpha rule, one commit step per "
                   "report-half teacher-forced decoder state (LLaDA-8B-Instruct, fp16)",
           "status": "exploratory", "exploratory": True, "stamp": STAMP, "half": "report",
           "alphas": list(ALPHAS), "alpha_note": "fixed before report-half numbers were read; the per-task "
           "C3a alphas of prereg/t12_grid_<task>.json were not committed, none added",
           "eligible_sets": {"block32": f"masked rows of the leftmost non-empty {BLOCK}-block of the region "
                                       "(primary)", "region": "every masked row of the state"},
           "rules": {"c3a": "n = max(1, #{c_hat >= 1 - alpha}), unclamped", "raw": "n = max(1, #{conf >= 1 - alpha}), "
                                                                                  "float32 compare"},
           "dense_alphas": {"grid": "160 log-spaced alpha in [0.005, 0.8] (rounded to 5 decimals, unique)",
                            "targets_tokens_per_state": list(TARGETS),
                            "note": "traces the error-vs-tokens curves only; the reported grid is `alphas`"},
           "notes": ["Both rules commit a prefix of the same ell order, so the comparison at equal alpha is mostly a "
                     "token-count effect: compare at equal tokens per state (arms[*][variant].matched_speed).",
                     "risk_unforced is taken over states where a position passed the bar; the unforced subsets "
                     "differ between the rules, so it is not a like-for-like comparison.",
                     "Errors are against the reference token, not task accuracy."],
           "bootstrap": {"n_boot": a.n_boot, "seed": a.seed, "label": "t11-realised-risk-<task>",
                         "unit": "document"},
           "hf": {"repo": "ANLP-project/c3-traces-8b", "revision": HF_REVISION}, "tasks": {}}
    for task in a.tasks:
        res["tasks"][task] = run_task(task, a.data_dir, a.art_dir, a.n_boot, a.seed)
    res["provenance"] = provenance()
    stem.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    stem.parent.joinpath(f"{stem.name}_macros.tex").write_text(macros(res, stem.with_suffix(".json").name))
    for p in matched_figure(res, stem) + matched_arms_figure(res, stem) + alpha_figures(res, stem):
        print(f"[risk] wrote {p}")
    print(f"[risk] wrote {stem.with_suffix('.json')} and the macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
