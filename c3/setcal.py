"""Set-level calibration: the probability that every token committed together is right.

A decoder commits a *set* of positions per step.  Per-token calibration (C3a)
recalibrates each token's own confidence and so addresses only the estimation
term of Mid Eq. (1); whether the tokens committed together are right *jointly*
also depends on how their errors co-vary, which a product of per-token
probabilities ignores.  This module estimates that joint probability directly.

In a state, positions are ordered by a token calibrator's c_hat (the order a
confidence decoder commits them in; ties go to the lower position) and S_j is
the top j.  Each prefix gets features computable from one forward pass:

* ``u``          logit of the independence product prod_{i in S_j} c_hat_i;
* ``min_ell``    the weakest member's log-odds;
* ``eb_gap``     sum of the members' entropies minus the largest -- the bound on
                 their total correlation the entropy-bounded sampler uses;
* ``log_size``   log j;
* ``adj_frac``   the share of the j-1 possible adjacent pairs inside S_j;
* the state's field features (median ell, mask rate, log |M|), optionally;

and the label is y_j = 1 if every token in S_j is right.  ``SetCalibrator`` is a
logistic model in these features that nests independence (f(u) = u, every other
coefficient 0).  ``build_set_table`` builds the training/evaluation rows from
teacher-forced decoder traces in one vectorised pass; ``prefix_features`` does the
same for a single state at decode time, and the two agree exactly (tested).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import DEFAULT_KNOTS
from .logit import _sigmoid, fit_fe_logit
from .splines import Spline

SET_GRID = tuple(range(1, 33)) + (40, 48, 56, 64, 80, 96, 112, 128)
"""Prefix sizes the set table records (capped at |M| per state)."""
BASE_FEATURES = ("log_size", "min_ell", "eb_gap", "adj_frac")
FIELD_SET_FEATURES = ("median_ell", "mask_rate", "log_n_masked")
_TINY = 1e-12


def logit_from_log(log_p: np.ndarray) -> np.ndarray:
    """logit(P) from log P, stable for P near 0 and near 1."""
    log_p = np.minimum(np.asarray(log_p, dtype=np.float64), -_TINY)
    return log_p - np.log(-np.expm1(log_p))


def order_by_confidence(c_hat: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Indices sorted by c_hat descending, ties to the lower position."""
    return np.lexsort((np.asarray(position), -np.asarray(c_hat, dtype=np.float64)))


def prefix_features(ell, entropy, position, c_hat, median_ell: float, mask_rate: float,
                    log_n_masked: float, n_prefix: int | None = None) -> tuple[np.ndarray, dict]:
    """Features of every prefix S_1..S_n of one state (n = ``n_prefix`` or |M|).

    Returns ``(order, features)``: ``order`` indexes the inputs in commit order
    and ``features[name][j-1]`` describes S_j.
    """
    ell = np.asarray(ell, dtype=np.float64)
    order = order_by_confidence(c_hat, position)
    n = ell.size if n_prefix is None else min(int(n_prefix), ell.size)
    c = np.clip(np.asarray(c_hat, dtype=np.float64)[order], _TINY, 1.0 - _TINY)
    e = np.asarray(ell, dtype=np.float64)[order]
    h = np.asarray(entropy, dtype=np.float64)[order]
    log_p = np.cumsum(np.log(c))
    rank = np.empty(ell.size, dtype=np.int64)
    rank[order] = np.arange(ell.size)
    pos = np.asarray(position, dtype=np.int64)
    by_pos = np.argsort(pos, kind="stable")
    sp = pos[by_pos]
    adjacent = np.flatnonzero(np.diff(sp) == 1)
    entry = np.maximum(rank[by_pos[adjacent]], rank[by_pos[adjacent + 1]])
    adj_cum = np.cumsum(np.bincount(entry, minlength=ell.size))
    j = np.arange(1, ell.size + 1, dtype=np.float64)
    feats = {
        "u": logit_from_log(log_p),
        "log_p_ind": log_p,
        "min_ell": np.minimum.accumulate(e),
        "eb_gap": np.cumsum(h) - np.maximum.accumulate(h),
        "log_size": np.log(j),
        "adj_frac": adj_cum / np.maximum(j - 1.0, 1.0),
        "median_ell": np.full(ell.size, float(median_ell)),
        "mask_rate": np.full(ell.size, float(mask_rate)),
        "log_n_masked": np.full(ell.size, float(log_n_masked)),
    }
    return order, {k: v[:n] for k, v in feats.items()}


def build_set_table(df, c_hat: np.ndarray, js=SET_GRID):
    """Prefix-set rows for every decoder state in ``df`` (one arm of a trace).

    ``df`` needs doc_id, state_id, position, ell, entropy, agree, n_masked,
    mask_rate; ``c_hat`` is the token calibrator's probability for each row.
    Row weight is 1 / (prefixes in the state x states in the document), so every
    document carries total weight 1, as ``calibrators.pool_weights`` does.
    """
    import pandas as pd

    n = len(df)
    c = np.clip(np.asarray(c_hat, dtype=np.float64), _TINY, 1.0 - _TINY)
    state = df["state_id"].to_numpy(np.int64)
    position = df["position"].to_numpy(np.int64)
    order = np.lexsort((position, -c, state))
    st = state[order]
    starts = np.r_[0, np.flatnonzero(np.diff(st)) + 1]
    sizes = np.diff(np.r_[starts, n])
    rank = np.arange(n) - np.repeat(starts, sizes)

    t = pd.DataFrame({
        "state_id": st,
        "doc_id": df["doc_id"].to_numpy(np.int64)[order],
        "rank": rank,
        "logc": np.log(c[order]),
        "ell": df["ell"].to_numpy(np.float64)[order],
        "h": df["entropy"].to_numpy(np.float64)[order],
        "agree": df["agree"].to_numpy(np.float64)[order],
        "logc_raw": -np.logaddexp(0.0, -df["ell"].to_numpy(np.float64)[order]),
    })
    g = t.groupby("state_id", sort=False)
    log_p = g["logc"].cumsum().to_numpy()
    log_p_raw = g["logc_raw"].cumsum().to_numpy()
    min_ell = g["ell"].cummin().to_numpy()
    eb_gap = (g["h"].cumsum() - g["h"].cummax()).to_numpy()
    y = g["agree"].cumprod().to_numpy()

    # Adjacent pairs: consecutive positions both masked in the same state enter
    # the prefix once both members have; count them at their entry rank.
    rank_of_row = np.empty(n, dtype=np.int64)
    rank_of_row[order] = rank
    by_pos = np.lexsort((position, state))
    s2, p2 = state[by_pos], position[by_pos]
    adjacent = np.flatnonzero((np.diff(p2) == 1) & (s2[1:] == s2[:-1]))
    entry = np.maximum(rank_of_row[by_pos[adjacent]], rank_of_row[by_pos[adjacent + 1]])
    # st[starts] holds each state once, ascending (``order`` sorts by state first).
    slot = starts[np.searchsorted(st[starts], s2[adjacent])] + entry
    pair_at = np.bincount(slot, minlength=n).astype(np.float64)
    adj_cum = pd.Series(pair_at).groupby(st).cumsum().to_numpy()

    # Field features of each state (over all its masked positions).
    med = t.groupby("state_id", sort=False)["ell"].transform("median").to_numpy()
    mask_rate = df["mask_rate"].to_numpy(np.float64)[order]
    n_masked = df["n_masked"].to_numpy(np.float64)[order]

    j = rank + 1.0
    keep = np.isin(rank + 1, np.asarray(js))
    out = pd.DataFrame({
        "doc_id": t["doc_id"].to_numpy()[keep], "state_id": st[keep],
        "j": (rank + 1)[keep].astype(np.int64), "y": y[keep],
        "u": logit_from_log(log_p[keep]), "log_p_ind": log_p[keep],
        "log_p_raw": log_p_raw[keep], "min_ell": min_ell[keep], "eb_gap": eb_gap[keep],
        "log_size": np.log(j[keep]), "adj_frac": (adj_cum / np.maximum(j - 1.0, 1.0))[keep],
        "median_ell": med[keep], "mask_rate": mask_rate[keep],
        "log_n_masked": np.log(np.maximum(n_masked[keep], 1.0)),
    })
    per_state = out.groupby("state_id")["j"].transform("size").to_numpy(np.float64)
    states_per_doc = out.groupby("doc_id")["state_id"].transform("nunique").to_numpy(np.float64)
    out["weight"] = 1.0 / (per_state * states_per_doc)
    return out


@dataclass
class SetCalibrator:
    """logit P(all of S right) = a + spline(u) + features @ b + u * log_size * c."""

    use_field: bool = True
    n_knots: int = 5
    ridge: float = 1e-4
    spline: Spline | None = None
    names: list = field(default_factory=list)
    intercept: float = 0.0
    beta: np.ndarray | None = None
    converged: bool = False

    @property
    def features(self) -> tuple:
        return BASE_FEATURES + (FIELD_SET_FEATURES if self.use_field else ())

    def _design(self, feats) -> np.ndarray:
        u = np.asarray(feats["u"], dtype=np.float64)
        cols = [self.spline.transform(u)]
        cols += [np.asarray(feats[name], dtype=np.float64)[:, None] for name in self.features]
        cols.append((u * np.asarray(feats["log_size"], dtype=np.float64))[:, None])
        return np.column_stack(cols)

    def fit(self, table, weights: np.ndarray | None = None) -> "SetCalibrator":
        self.spline = Spline.fit(table["u"].to_numpy(np.float64), n_knots=self.n_knots)
        X = self._design(table)
        self.names = self.spline.names("f_u") + list(self.features) + ["u*log_size"]
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights * (weights.size / weights.sum())
        res = fit_fe_logit(table["y"].to_numpy(np.float64), X, np.zeros(len(table), np.int64),
                           n_groups=1, names=self.names, ridge_fe=0.0, ridge_beta=self.ridge,
                           weights=weights)
        self.intercept, self.beta = float(res.alpha[0]), res.beta
        self.converged = bool(res.converged)
        return self

    def predict(self, feats) -> np.ndarray:
        if self.beta is None:
            raise RuntimeError("SetCalibrator is not fitted")
        return _sigmoid(self.intercept + self._design(feats) @ self.beta)

    def coef(self) -> dict:
        return dict(zip(self.names, np.asarray(self.beta, dtype=float).tolist()))

    def to_dict(self) -> dict:
        return {"kind": "set_calibrator", "use_field": self.use_field, "n_knots": self.n_knots,
                "ridge": self.ridge, "knots": self.spline.knots.tolist(), "names": self.names,
                "intercept": self.intercept, "beta": np.asarray(self.beta).tolist(),
                "converged": self.converged}

    @classmethod
    def from_dict(cls, d: dict) -> "SetCalibrator":
        if d.get("kind") != "set_calibrator":
            raise ValueError(f"not a set calibrator: kind={d.get('kind')!r}")
        return cls(use_field=d["use_field"], n_knots=d["n_knots"], ridge=d["ridge"],
                   spline=Spline(np.asarray(d["knots"])), names=list(d["names"]),
                   intercept=float(d["intercept"]), beta=np.asarray(d["beta"], dtype=np.float64),
                   converged=bool(d["converged"]))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path) -> "SetCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_token_calibrator(df, features=FIELD_SET_FEATURES, n_knots: int = DEFAULT_KNOTS,
                         ridge: float = 1e-4, interactions: bool = True):
    """The decode-time token calibrator pi_hat(ell, s), fitted with pool weights."""
    from .calibrators import ContextLogistic, pool_weights
    from .frame import PositionFrame

    pos = PositionFrame.from_frame(df)
    return ContextLogistic(features=tuple(features), n_knots=n_knots, ridge=ridge,
                           interactions=interactions).fit(pos, pool_weights(pos))


def crossfit_token_probs(df, fit: np.ndarray, **kw):
    """c_hat for every row of ``df``: out of fold (two folds of the documents, by
    ``hash_halves(doc_id, 2)``) on the ``fit`` rows, and from a calibrator fitted on all
    ``fit`` rows elsewhere.  The set table must be built from out-of-fold c_hat, or the
    set model learns the token calibrator's in-sample optimism.  Returns (c_hat, full)."""
    from .frame import PositionFrame, hash_halves

    fit = np.asarray(fit, dtype=bool)
    c_hat = np.full(len(df), np.nan)
    fold = hash_halves(df["doc_id"].to_numpy(), 2)
    for f in (0, 1):
        tr, te = fit & (fold != f), fit & (fold == f)
        if not te.any():
            continue
        if not tr.any():
            raise ValueError("cross-fitting needs fit documents in both folds")
        c_hat[te] = fit_token_calibrator(df[tr], **kw).predict(PositionFrame.from_frame(df[te]))
    full = fit_token_calibrator(df[fit], **kw)
    if (~fit).any():
        c_hat[~fit] = full.predict(PositionFrame.from_frame(df[~fit]))
    return c_hat, full
