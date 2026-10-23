"""T8 controls: rule out the boring explanations (plan v2 T8).

Each control becomes one row of a single ablation table: the main effect with
and without the control, and their difference, all with document-bootstrap
intervals on one shared resample matrix (so the difference has a valid
interval).  The main effects are the two primary endpoints,

* E1: Delta(32) - Delta(1), Delta(m) = E_m[c - a] over pool winners;
* E2: field(32) - field(1), field(m) = E_m[pi(l) - pi(l, s)].

**Position** and **token type** are covariate controls.  With x the control
covariates, Prop. 2's split extends by one term::

    Delta(m) = E_m[c - pi(l)] + E_m[pi(l) - pi(l, x)] + E_m[pi(l, x) - a]

The middle term is the part of the gap that the control explains beyond the
token's own confidence (winners of large pools sitting in particular positions,
or being particular kinds of token).  "With the control" removes it:
Delta_x(m) = Delta(m) - E_m[pi(l) - pi(l, x)], and for E2 the field term is
recomputed with x in both calibrators, E_m[pi(l, x) - pi(l, x, s)].  If the
effect disappears, it was positional (or a token-type artefact), not selection.
The calibrators here are all logistic in a spline of ell (x and the field
features added linearly, ell x feature interactions in the context model), so
"with" and "without" differ only by x; the pre-registered isotonic E2 estimate
is reported alongside for reference.  All are fitted on the fit half.

Position covariates: absolute position (as a fraction of the document), the
distance to the nearest unmasked token, and PC-Sampler's position-aware weight
exp(-0.25 i) (``c3.decode.policies.PCSampler``; times its frequency factor when a
unigram table is given).  Token type: the slot's true token is punctuation /
whitespace, an English stop-word, or content; the split by type is reported too.

**Temperature**: one temperature per mask-rate bin, fitted on the fit half, then
Delta(m) recomputed.  ``softmax`` divides the logits by T, reconstructed from the
stored top-20 probabilities with the rest of the vocabulary as one tail bucket
spread evenly over the remaining tokens (exact at T = 1), and is fitted by the
NLL of the true token, as in Guo et al. (2017); a tempered confidence can rank
positions differently, so pools are re-ranked on it.  ``logodds`` rescales the
top-1 log-odds, c = sigma(ell / T), fitted by the NLL of correctness; exact
from the trace, and it never changes a ranking.

**Precision**: the same documents in fp16 and 4-bit (``precision_rows``).

**Confidence statistic** (T13 ablation 1): pools won by the highest margin or the
lowest entropy instead of the highest ell, on the same pool draws, with the gap
still the committed token's own confidence minus its correctness.  Only the pool
version: decoder-induced states exist for confidence top-k only (deviations item 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import DEFAULT_KNOTS, N_POOLS, POOL_SIZES
from .calibrators import ContextLogistic, IsotonicOwn, _features_exact, pool_weights
from .endpoints.e1 import winner_gap
from .frame import FIELD_FEATURES, AnalysisFrame, PositionFrame
from .logit import _sigmoid, fit_fe_logit
from .splines import Spline
from .stats import GroupedMean, boot_pvalue, percentile_ci

# A fixed English stop-word list (NLTK's), reproduced so nothing is downloaded.
STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before
being below between both but by can did do does doing don down during each few for from
further had has have having he her here hers herself him himself his how i if in into is
it its itself just me more most my myself no nor not now of off on once only or other our
ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we
were what when where which while who whom why will with you your yours yourself yourselves
""".split())

TOKEN_CLASSES = ("content", "punct", "stop")

TEMPERATURE_EDGES = np.array([0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0 + 1e-9])
"""Mask-rate bins for the temperature control: centred on 0.1, ..., 0.9, the end
bins taking the rest of (0, 1] (decoder states reach rate 1.0 and ~0)."""


def temperature_bin(rate: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(np.asarray(rate, np.float64), TEMPERATURE_EDGES) - 1, 0,
                   TEMPERATURE_EDGES.size - 2)


# --------------------------------------------------------------------------- #
# Position covariates
# --------------------------------------------------------------------------- #

def distance_to_unmasked(state_id: np.ndarray, position: np.ndarray,
                         doc_len: np.ndarray) -> np.ndarray:
    """Per row: distance from the masked position to the nearest unmasked slot of its state.

    The rows of a state are exactly its masked positions, so everything else in
    [0, doc_len) is visible (for tasks that includes the prompt).  A fully masked
    state has no visible slot; its rows get the document length.
    """
    state_id = np.asarray(state_id, np.int64)
    position = np.asarray(position, np.int64)
    doc_len = np.asarray(doc_len, np.int64)
    order = np.lexsort((position, state_id))
    s, p, L = state_id[order], position[order], doc_len[order]
    out_sorted = np.empty(order.size, dtype=np.int64)
    bounds = np.flatnonzero(np.diff(s)) + 1
    for rows in np.split(np.arange(order.size), bounds):
        pp = p[rows]
        n = int(L[rows[0]])
        # Runs of consecutive masked positions; inside a run [a, b] the nearest
        # visible slots are a - 1 and b + 1, when they exist.
        brk = np.flatnonzero(np.diff(pp) != 1) + 1
        starts = np.r_[0, brk]
        ends = np.r_[brk, pp.size] - 1
        run = np.repeat(np.arange(starts.size), ends - starts + 1)
        a, b = pp[starts][run], pp[ends][run]
        left = np.where(a > 0, pp - (a - 1), np.iinfo(np.int64).max)
        right = np.where(b < n - 1, (b + 1) - pp, np.iinfo(np.int64).max)
        d = np.minimum(left, right)
        d[d == np.iinfo(np.int64).max] = n
        out_sorted[rows] = d
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def pc_sampler_weight(position: np.ndarray, span_start: np.ndarray | int = 0,
                      lam: float = 0.25, alpha: float = 10.0,
                      log_freq: np.ndarray | None = None,
                      top1: np.ndarray | None = None) -> np.ndarray:
    """PC-Sampler's weight exp(-lambda i) [* min(-log p_D(top1), alpha)], i in the generated region."""
    i = np.asarray(position, np.int64) - np.asarray(span_start, np.int64)
    w = np.exp(-lam * i.astype(np.float64))
    if log_freq is not None:
        if top1 is None:
            raise ValueError("the frequency factor needs the top-1 token ids")
        w = w * np.minimum(-np.asarray(log_freq)[np.asarray(top1, np.int64)], alpha)
    return w


POSITION_COVARIATES = ("abs_position", "log1p_dist_unmasked", "pc_sampler_weight")

DIST_STRATA = (1, 2, 3, 5)
"""Lower edges of the distance-to-unmasked strata: 1, 2, 3-4, 5+."""


def position_strata(df) -> np.ndarray:
    """Stratum of every row: distance to unmasked (1, 2, 3-4, 5+) x absolute-position tercile.

    The nonparametric companion of the covariate control: Delta(m) within strata,
    averaged with fixed weights, cannot be moved by winners merely sitting in
    different positions.  (PC-Sampler's weight is a function of position, so the
    position terciles cover it.)
    """
    dist = distance_to_unmasked(df["state_id"].to_numpy(), df["position"].to_numpy(),
                                df["doc_len"].to_numpy())
    db = np.searchsorted(np.asarray(DIST_STRATA), dist, side="right") - 1
    frac = df["position"].to_numpy(np.float64) / np.maximum(df["doc_len"].to_numpy(np.float64), 1)
    tb = np.minimum((frac * 3).astype(np.int64), 2)
    return db * 3 + tb


def position_covariates(df, log_freq: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """The T8 position controls for every row of one arm's trace (unstandardised)."""
    L = df["doc_len"].to_numpy(np.float64)
    start = df["span_start"].to_numpy(np.int64) if "span_start" in df.columns else 0
    dist = distance_to_unmasked(df["state_id"].to_numpy(), df["position"].to_numpy(),
                                df["doc_len"].to_numpy())
    top1 = df["top1"].to_numpy(np.int64) if "top1" in df.columns else None
    X = np.column_stack([df["position"].to_numpy(np.float64) / np.maximum(L, 1.0),
                         np.log1p(dist.astype(np.float64)),
                         pc_sampler_weight(df["position"].to_numpy(), start, log_freq=log_freq,
                                           top1=top1)])
    return X, list(POSITION_COVARIATES)


# --------------------------------------------------------------------------- #
# Token type
# --------------------------------------------------------------------------- #

def classify_string(s: str) -> str:
    """'punct' (no letters or digits: punctuation, symbols, whitespace), 'stop' or 'content'."""
    t = s.strip()
    if not t or not any(ch.isalnum() for ch in t):
        return "punct"
    return "stop" if t.lower() in STOP_WORDS else "content"


def token_class_table(tokenizer, vocab_size: int | None = None) -> np.ndarray:
    """Class index (into TOKEN_CLASSES) of every token id, from its decoded string."""
    V = int(vocab_size or len(tokenizer))
    strings = tokenizer.batch_decode([[i] for i in range(V)])
    return np.array([TOKEN_CLASSES.index(classify_string(s)) for s in strings], dtype=np.int8)


def slot_token_ids(df, x0_by_doc: dict) -> np.ndarray:
    """The true token at every row's (document, position)."""
    doc = df["doc_id"].to_numpy(np.int64)
    pos = df["position"].to_numpy(np.int64)
    out = np.empty(doc.size, dtype=np.int64)
    for d in np.unique(doc):
        sel = doc == d
        out[sel] = np.asarray(x0_by_doc[int(d)])[pos[sel]]
    return out


def token_covariates(classes: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Indicators for punctuation and stop-words (content is the reference level)."""
    c = np.asarray(classes)
    return np.column_stack([(c == 1).astype(float), (c == 2).astype(float)]), ["is_punct", "is_stop"]


# --------------------------------------------------------------------------- #
# Calibrators with control covariates
# --------------------------------------------------------------------------- #

@dataclass
class CovariateLogistic:
    """pi(l, x) = sigma(a + f(l) + x @ b [+ l * x @ c]), x standardised on the fit rows."""

    n_knots: int = DEFAULT_KNOTS
    ridge: float = 1e-4
    interactions: bool = False
    spline: Spline | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    intercept: float = 0.0
    beta: np.ndarray | None = None
    converged: bool = False
    keep_: np.ndarray | None = field(default=None, repr=False)

    def _design(self, ell, X):
        cols = [self.spline.transform(ell)]
        if X is not None and X.shape[1]:
            Z = (X[:, self.keep_] - self.mean_) / self.scale_
            cols.append(Z)
            if self.interactions:
                cols.append(ell[:, None] * Z)
        return np.column_stack(cols)

    def fit(self, ell, X, agree, weights=None) -> "CovariateLogistic":
        ell = np.asarray(ell, np.float64)
        self.spline = Spline.fit(ell, n_knots=self.n_knots)
        if X is not None and X.shape[1]:
            X = np.asarray(X, np.float64)
            sd = X.std(axis=0)
            self.keep_ = np.isfinite(sd) & (sd > 0)       # constant columns carry nothing
            self.mean_ = X[:, self.keep_].mean(axis=0)
            self.scale_ = sd[self.keep_]
        D = self._design(ell, X)
        w = None
        if weights is not None:
            w = np.asarray(weights, np.float64)
            w = w * (w.size / w.sum())
        r = fit_fe_logit(np.asarray(agree, np.float64), D, np.zeros(ell.size, np.int64),
                         n_groups=1, ridge_fe=0.0, ridge_beta=self.ridge, weights=w)
        self.intercept, self.beta, self.converged = float(r.alpha[0]), r.beta, bool(r.converged)
        return self

    def predict(self, ell, X) -> np.ndarray:
        X = None if X is None else np.asarray(X, np.float64)
        return _sigmoid(self.intercept + self._design(np.asarray(ell, np.float64), X) @ self.beta)


def _field_matrix(pos: PositionFrame, fit_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Field features for every row, with the feature set chosen on the fit rows."""
    _, used = pos.select(fit_mask).field_features(FIELD_FEATURES)
    return _features_exact(pos, used), used


def _contrast(g: GroupedMean, hi: int, lo: int) -> tuple[float, np.ndarray]:
    return g.contrast(hi, lo)


@dataclass
class ControlEffects:
    """Per-m estimates with and without one covariate control, on shared resamples."""

    delta: GroupedMean           # E_m[c - a]
    delta_adj: GroupedMean       # Delta - E_m[pi(l) - pi(l, x)]
    control_term: GroupedMean    # E_m[pi(l) - pi(l, x)]
    field0: GroupedMean | None   # E_m[pi(l) - pi(l, s)]
    field_x: GroupedMean | None  # E_m[pi(l, x) - pi(l, x, s)]
    fits: dict


def covariate_control(pools: AnalysisFrame, pos: PositionFrame, X_ctrl: np.ndarray,
                      fit_mask: np.ndarray, report_mask: np.ndarray, codes: np.ndarray,
                      counts: np.ndarray, n_knots: int = DEFAULT_KNOTS,
                      with_field: bool = True) -> ControlEffects:
    """Delta(m) and the field term with and without the control covariates ``X_ctrl``.

    Calibrators are fitted on ``fit_mask`` rows with pool weights and evaluated
    everywhere; estimates use ``report_mask`` rows only.
    """
    fit_mask = np.asarray(fit_mask, bool)
    tr = pos.select(fit_mask)
    w = pool_weights(tr)
    Xc_tr = np.asarray(X_ctrl)[fit_mask]
    pi0 = CovariateLogistic(n_knots).fit(tr.ell, None, tr.agree, w).predict(pos.ell, None)
    pix_m = CovariateLogistic(n_knots).fit(tr.ell, Xc_tr, tr.agree, w)
    pix = pix_m.predict(pos.ell, X_ctrl)
    c, a = pos.conf, pos.agree

    def gap(q, t):
        return winner_gap(pools, pos, q, t, codes, counts, report_mask)

    delta = gap(c, a)
    ctrl = gap(pi0, pix)
    adj = gap(c - pi0 + pix, a)
    fits = {"control_model_converged": pix_m.converged,
            "control_columns_used": int(pix_m.keep_.sum()) if pix_m.keep_ is not None else 0}
    f0 = fx = None
    if with_field:
        Xf, used = _field_matrix(pos, fit_mask)
        if Xf.shape[1]:
            pis = CovariateLogistic(n_knots, interactions=True).fit(
                tr.ell, Xf[fit_mask], tr.agree, w).predict(pos.ell, Xf)
            Xfx = np.column_stack([Xf, X_ctrl])
            pixs = CovariateLogistic(n_knots, interactions=True).fit(
                tr.ell, Xfx[fit_mask], tr.agree, w).predict(pos.ell, Xfx)
            f0, fx = gap(pi0, pis), gap(pix, pixs)
            fits["field_features"] = used
    return ControlEffects(delta, adj, ctrl, f0, fx, fits)


def preregistered_field(pools, pos, fit_mask, report_mask, codes, counts,
                        n_knots: int = DEFAULT_KNOTS) -> GroupedMean:
    """The E2 field term exactly as analyze_traces computes it (isotonic own, interacted context)."""
    tr = pos.select(np.asarray(fit_mask, bool))
    w = pool_weights(tr)
    own = IsotonicOwn().fit(tr.ell, tr.agree, w)
    ctx = ContextLogistic(n_knots=n_knots).fit(tr, w)
    return winner_gap(pools, pos, own.predict(pos.ell), ctx.predict(pos), codes, counts,
                      report_mask)


# --------------------------------------------------------------------------- #
# Temperature
# --------------------------------------------------------------------------- #

def _log_z(logp_topk: np.ndarray, tail: np.ndarray, vocab: int, T: float) -> np.ndarray:
    k = logp_topk.shape[1]
    rest = max(vocab - k, 1)
    tail = np.clip(tail, 1e-12, 1.0)
    terms = np.column_stack([logp_topk / T, np.log(rest) + (np.log(tail) - np.log(rest)) / T])
    mx = terms.max(axis=1, keepdims=True)
    return (mx + np.log(np.exp(terms - mx).sum(axis=1, keepdims=True)))[:, 0]


def tempered_softmax(topk_p: np.ndarray, p_true: np.ndarray, vocab: int, T: float):
    """(top-1 confidence, true-token probability) after dividing the logits by T.

    Rebuilt from the stored top-k probabilities plus one tail bucket whose mass is
    spread evenly over the other V - k tokens; exact at T = 1.  The top-1 after
    tempering is still the top-1 (dividing by T > 0 keeps the order).
    """
    p = np.clip(np.asarray(topk_p, dtype=np.float64), 1e-30, 1.0)
    logp = np.log(p)
    tail = 1.0 - p.sum(axis=1)
    lz = _log_z(logp, tail, vocab, T)
    conf = np.exp(logp[:, 0] / T - lz)
    pt = np.exp(np.log(np.clip(np.asarray(p_true, np.float64), 1e-30, 1.0)) / T - lz)
    return conf, pt


def _minimise_log_t(nll, bounds=(0.05, 20.0)) -> float:
    from scipy.optimize import minimize_scalar

    r = minimize_scalar(nll, bounds=np.log(bounds), method="bounded",
                        options={"xatol": 1e-5})
    return float(np.exp(r.x))


def fit_temperatures(kind: str, rate: np.ndarray, weights: np.ndarray, *, ell=None, agree=None,
                     topk_p=None, p_true=None, vocab: int | None = None,
                     min_rows: int = 50) -> np.ndarray:
    """One temperature per mask-rate bin (``TEMPERATURE_EDGES``); T = 1 where a bin is too thin."""
    b = temperature_bin(rate)
    w = np.asarray(weights, np.float64)
    T = np.ones(TEMPERATURE_EDGES.size - 1)
    for j in range(T.size):
        sel = b == j
        if sel.sum() < min_rows:
            continue
        ws = w[sel] / w[sel].sum()
        if kind == "logodds":
            e, y = np.asarray(ell, np.float64)[sel], np.asarray(agree, np.float64)[sel]

            def nll(lt, e=e, y=y, ws=ws):
                z = e / np.exp(lt)
                return float(np.sum(ws * (np.logaddexp(0, z) - y * z)))
        elif kind == "softmax":
            tk, pt = np.asarray(topk_p)[sel], np.asarray(p_true)[sel]

            def nll(lt, tk=tk, pt=pt, ws=ws):
                _, q = tempered_softmax(tk, pt, vocab, float(np.exp(lt)))
                return -float(np.sum(ws * np.log(np.clip(q, 1e-30, 1.0))))
        else:
            raise ValueError(f"temperature kind must be 'softmax' or 'logodds', got {kind!r}")
        T[j] = _minimise_log_t(nll)
    return T


def apply_temperatures(kind: str, T: np.ndarray, rate: np.ndarray, *, ell=None, topk_p=None,
                       p_true=None, vocab: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(tempered confidence, its log-odds) for every row."""
    b = temperature_bin(rate)
    Tr = np.asarray(T)[b]
    if kind == "logodds":
        e = np.asarray(ell, np.float64) / Tr
        return _sigmoid(e), e
    conf = np.empty(b.size)
    for j in np.unique(b):
        sel = b == j
        conf[sel], _ = tempered_softmax(np.asarray(topk_p)[sel], np.asarray(p_true)[sel],
                                        vocab, float(T[j]))
    conf = np.clip(conf, 1e-12, 1 - 1e-12)
    return conf, np.log(conf) - np.log1p(-conf)


# --------------------------------------------------------------------------- #
# Confidence statistic (plan v2 T13 ablation 1)
# --------------------------------------------------------------------------- #

RANK_STATISTICS = ("margin", "entropy")


def rank_score(df, statistic: str) -> np.ndarray:
    """What a decoder ranks positions by under ``statistic``, highest committed first:
    the margin p_top1 - p_top2 (the Margin decoder) or minus the predictive entropy
    (EB-Sampler's order)."""
    if statistic == "margin":
        return df["margin"].to_numpy(np.float64)
    if statistic == "entropy":
        return -df["entropy"].to_numpy(np.float64)
    raise ValueError(f"statistic must be one of {RANK_STATISTICS}, got {statistic!r}")


def reranked_pools(pos: PositionFrame, score: np.ndarray, seed: int) -> AnalysisFrame:
    """The T8 pools of ``pos`` ranked by ``score`` instead of ell.

    Pool memberships never look at the score, so with the same seed these are the
    same pools as the ell-ranked ones; only who wins them changes, and the two
    Delta(m) curves are paired draw by draw.
    """
    from .assemble import frame_from_arrays

    return frame_from_arrays(pos.doc_id, pos.state_id, pos.position, score, pos.agree,
                             pool_sizes=POOL_SIZES, n_pools_per_size=N_POOLS, seed=seed,
                             retain=1, min_state_size=max(POOL_SIZES))


def winner_agreement(pools_a: AnalysisFrame, pools_b: AnalysisFrame) -> dict:
    """Per pool size, the share of pools whose winner is the same position under both rankings."""
    wa, wb = pools_a.winners(), pools_b.winners()
    if len(wa) != len(wb) or not np.array_equal(wa.m, wb.m):
        raise ValueError("the two pool frames do not hold the same pools")
    same = wa.src_row == wb.src_row
    return {int(m): float(same[wa.m == m].mean()) for m in np.unique(wa.m)}


@dataclass
class StatisticEffects:
    """Delta(m) and the E2 field term when pools are won by another confidence statistic."""

    delta: GroupedMean           # E_m[c - a], winners by the statistic, c the stored confidence
    field: GroupedMean           # E_m[pi(l) - pi(l, s)] over the same winners
    agreement: dict              # m -> share of pools with the same winner as under ell


def statistic_control(pools: AnalysisFrame, pos: PositionFrame, score: np.ndarray,
                      fit_mask: np.ndarray, report_mask: np.ndarray, codes: np.ndarray,
                      counts: np.ndarray, seed: int,
                      n_knots: int = DEFAULT_KNOTS) -> StatisticEffects:
    """The E1 gap and the pre-registered E2 field term with pools ranked by ``score``.

    The gap is still the committed token's own probability minus its correctness:
    the statistic changes which token is committed, not what the model claims for it.
    The calibrators are the pre-registered ones (fitted on all fit-half positions,
    which do not depend on the ranking).
    """
    pools_s = reranked_pools(pos, score, seed)
    delta = winner_gap(pools_s, pos, pos.conf, pos.agree, codes, counts, report_mask)
    field = preregistered_field(pools_s, pos, fit_mask, report_mask, codes, counts, n_knots)
    return StatisticEffects(delta, field, winner_agreement(pools, pools_s))


# --------------------------------------------------------------------------- #
# Table rows
# --------------------------------------------------------------------------- #

def _est(point: float, draws: np.ndarray, alpha: float = 0.05) -> dict:
    return {"value": float(point), "ci": list(percentile_ci(draws, alpha)),
            "p_boot": boot_pvalue(draws)}


def ablation_row(control: str, effect: str, arm: str, without: tuple, with_: tuple,
                 n_docs: int, note: str = "", alpha: float = 0.05) -> dict:
    """One row: (point, draws) without and with the control, and their paired difference."""
    (p0, d0), (p1, d1) = without, with_
    return {"control": control, "effect": effect, "arm": arm,
            "without": _est(p0, d0, alpha), "with": _est(p1, d1, alpha),
            "difference": _est(p1 - p0, d1 - d0, alpha), "n_docs": int(n_docs), "note": note}


def standardised_contrast(groups: dict, shares: dict, hi: int = max(POOL_SIZES),
                          lo: int = min(POOL_SIZES)) -> tuple[float, np.ndarray]:
    """sum_c share_c x [est_c(hi) - est_c(lo)] over strata c, with joint draws."""
    point, draws = 0.0, 0.0
    for c, g in groups.items():
        p, d = g.contrast(hi, lo)
        point += shares[c] * p
        draws = draws + shares[c] * d
    return point, draws


def latex_table(rows: list[dict]) -> str:
    """The ablation table as a LaTeX tabular (numbers only from ``rows``)."""
    def cell(e):
        return f"${e['value']:+.4f}$ \\scriptsize[{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}]"

    lines = ["% Generated by scripts/run_controls.py -- do not edit by hand.",
             "\\begin{tabular}{llllll}", "\\toprule",
             "Control & Effect & Arm & Without & With & Difference \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join([r["control"].replace("_", " "), r["effect"].replace("_", "\\_"),
                                 r["arm"].replace("_", "\\_"), cell(r["without"]),
                                 cell(r["with"]), cell(r["difference"])]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"
