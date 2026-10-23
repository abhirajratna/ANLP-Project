"""The two calibrators behind E2 (plan v2 T6) and C3a (T11).

``IsotonicOwn`` estimates pi(l) = Pr(a=1 | l), the ordinary calibration curve,
from a token's own confidence only.  ``ContextLogistic`` estimates pi(l, s) by
adding per-state field features (the median confidence of the state's masked
positions, the mask rate, |M| and the step).  Prop. 2 splits the committed-token
gap into E_m[sigma(l) - pi(l)] plus the field term E_m[pi(l) - pi(l, s)], and
these two fits are the plug-in estimates of pi(l) and pi(l, s).

Both are fitted on one set of documents and applied to another.  Fitting and
reporting on the same documents would let the isotonic curve chase the very
winners it is meant to predict and pull the field term towards zero.

Both are fitted with *pool weights* by default: each position counts
1 / (|state| x states in its document).  That is the population the endpoints
average over -- pools are drawn per state and documents are weighted equally --
and the decomposition is only exact at m=1 (field term = 0) when pi_hat(l) and
pi_hat(l, s) are calibrated for that same population.  Weighting every position
equally instead lets large states (high mask rates) dominate the fit; when the
field features track state size, as mask rate does, the m=1 field term then
comes out nonzero by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import DEFAULT_KNOTS
from .frame import FIELD_FEATURES, PositionFrame
from .logit import _sigmoid, fit_fe_logit
from .metrics import IsotonicFit, isotonic_fit
from .splines import Spline


def pool_weights(pos: PositionFrame) -> np.ndarray:
    """1 / (|state| x number of states in the document), per position.

    Every document then carries total weight 1 and every state within a document
    the same share -- the weighting the pooled endpoints use.
    """
    s_idx, n_states = pos.state_index
    n_in_state = np.bincount(s_idx, minlength=n_states)[s_idx].astype(np.float64)
    first = np.zeros(n_states, dtype=np.int64)
    first[s_idx[::-1]] = np.arange(len(pos))[::-1]
    doc_of_state = pos.doc_id[first]
    docs, inv = np.unique(doc_of_state, return_inverse=True)
    states_per_doc = np.bincount(inv)[inv]            # per state
    return 1.0 / (n_in_state * states_per_doc[s_idx])


@dataclass
class IsotonicOwn:
    """pi_hat(l): non-decreasing in the token's own confidence log-odds."""

    fit_: IsotonicFit | None = None

    def fit(self, ell: np.ndarray, agree: np.ndarray,
            weights: np.ndarray | None = None) -> "IsotonicOwn":
        self.fit_ = isotonic_fit(ell, agree, weights)
        return self

    def predict(self, ell: np.ndarray) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("IsotonicOwn is not fitted")
        return self.fit_.predict(ell)

    def to_dict(self) -> dict:
        return {"kind": "isotonic_own", **self.fit_.to_dict()}


@dataclass
class ContextLogistic:
    """pi_hat(l, s) = sigma(alpha + f(l) + g(s) @ gamma [+ l * g(s) @ delta]).

    ``f`` is the same restricted cubic spline as Eq. 2, with knots frozen on the
    fit documents; ``g(s)`` are the per-state field features.  Features that are
    constant in the fit data are dropped (their names are recorded), and a small
    ridge keeps the fit finite when mask rate, |M| and step are nearly collinear,
    as they are on fixed-length text.

    ``interactions=True`` (the default) adds l x feature terms, so the slope of
    the calibration curve can differ between states, not just its level.  Plan v2
    T6(2) writes the additive model, kept as ``interactions=False``.  The
    interacted form is the default because noise whose scale varies by state --
    the mechanism behind the field term -- changes the *slope* of pi(l, s), and an
    additive fit then misstates pi(l, s) exactly where large-m winners sit.  On
    ``c3.sim.positions`` 'context' the additive fit inflates the m=32 field term
    about fourfold, and the Prop. 1 residual E_m[pi_hat(l, s) - a] exposes it
    (-0.028 additive vs -0.007 interacted).  Which form is pre-registered is a
    team decision; report the residual either way, as a specification check.
    """

    features: tuple = FIELD_FEATURES
    n_knots: int = DEFAULT_KNOTS
    ridge: float = 1e-4
    interactions: bool = True
    spline: Spline | None = None
    used: list = field(default_factory=list)
    intercept: float = 0.0
    beta: np.ndarray | None = None
    converged: bool = False
    feature_ranges: dict | None = None    # name -> {"min", "max"} over the fit states, if recorded

    def _design(self, ell: np.ndarray, X_field: np.ndarray) -> np.ndarray:
        cols = [self.spline.transform(ell)]
        if X_field.size:
            cols.append(X_field)
            if self.interactions:
                cols.append(ell[:, None] * X_field)
        return np.column_stack(cols)

    def fit(self, pos: PositionFrame, weights: np.ndarray | None = None) -> "ContextLogistic":
        X_field, self.used = pos.field_features(self.features)
        self.spline = Spline.fit(pos.ell, n_knots=self.n_knots)
        X = self._design(pos.ell, X_field)
        if weights is not None:
            # Rescale to mean 1 so the ridge keeps the same strength.
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights * (weights.size / weights.sum())
        names = self.spline.names() + list(self.used)
        if self.interactions and self.used:
            names += [f"ell*{u}" for u in self.used]
        res = fit_fe_logit(pos.agree, X, np.zeros(len(pos), dtype=np.int64), n_groups=1,
                           names=names, ridge_fe=0.0, ridge_beta=self.ridge, weights=weights)
        self.intercept = float(res.alpha[0])
        self.beta = res.beta
        self.converged = bool(res.converged)
        return self

    def predict(self, pos: PositionFrame) -> np.ndarray:
        if self.beta is None:
            raise RuntimeError("ContextLogistic is not fitted")
        X_field, used = pos.field_features(self.used)
        if used != list(self.used):
            # A feature that varied in the fit documents is constant here; it
            # still has a coefficient, so evaluate it rather than dropping it.
            X_field = _features_exact(pos, self.used)
        eta = self.intercept + self._design(pos.ell, X_field) @ self.beta
        return _sigmoid(eta)

    def predict_arrays(self, ell: np.ndarray, field: dict | None = None) -> np.ndarray:
        """pi_hat from raw arrays: ``field`` maps each used feature name to a value
        or an array broadcastable to ``ell`` (what a decoder has at a step)."""
        if self.beta is None:
            raise RuntimeError("ContextLogistic is not fitted")
        ell = np.asarray(ell, dtype=np.float64)
        field = field or {}
        missing = [n for n in self.used if n not in field]
        if missing:
            raise KeyError(f"field features {missing} are needed by this calibrator")
        X = (np.column_stack([np.broadcast_to(np.asarray(field[n], dtype=np.float64), ell.shape)
                              for n in self.used]) if self.used else np.empty((ell.size, 0)))
        return _sigmoid(self.intercept + self._design(ell, X) @ self.beta)

    def to_dict(self) -> dict:
        d = {"kind": "context_logistic", "features": list(self.used),
             "interactions": self.interactions,
             "knots": self.spline.knots.tolist(), "intercept": self.intercept,
             "beta": self.beta.tolist(), "converged": self.converged}
        if self.feature_ranges is not None:
            d["feature_ranges"] = self.feature_ranges
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ContextLogistic":
        if d.get("kind") != "context_logistic":
            raise ValueError(f"not a context logistic: kind={d.get('kind')!r}")
        knots = np.asarray(d["knots"], dtype=np.float64)
        return cls(features=tuple(d["features"]), n_knots=knots.size,
                   interactions=bool(d["interactions"]), spline=Spline(knots),
                   used=list(d["features"]), intercept=float(d["intercept"]),
                   beta=np.asarray(d["beta"], dtype=np.float64), converged=bool(d["converged"]),
                   feature_ranges=d.get("feature_ranges"))

    @classmethod
    def load(cls, path) -> "ContextLogistic":
        """A calibrator saved by ``scripts/fit_c3a_calibrator.py`` (its ``calibrator`` entry)
        or a bare ``to_dict()`` JSON."""
        import json
        from pathlib import Path

        d = json.loads(Path(path).read_text())
        return cls.from_dict(d.get("calibrator", d))


DECODE_FEATURES = ("median_ell", "mask_rate", "log_n_masked")
"""The field features a decoder can compute at a step, defined as in the traces.  ``step``
is left out: a trace's step counts teacher-forced commits of k tokens, a free-running
decoder's does not."""


def state_field_features(ell_masked: np.ndarray, n_masked: int, region_size: int) -> dict:
    """The decode-time field features of one state, as the traces define them
    (``c3.traces.masks.MaskedState.rate``, ``PositionFrame``): the median ell over every
    masked position, the masked share of the maskable region, and log |M|."""
    return {"median_ell": float(np.median(np.asarray(ell_masked, dtype=np.float64))),
            "mask_rate": float(n_masked) / float(region_size),
            "log_n_masked": float(np.log(max(int(n_masked), 1)))}


def _features_exact(pos: PositionFrame, names) -> np.ndarray:
    """Field features by name with no constant-column dropping."""
    cols = []
    for name in names:
        if name == "median_ell":
            cols.append(pos.state_median_ell())
        elif name == "mask_rate":
            cols.append(pos.mask_rate.astype(np.float64))
        elif name == "log_n_masked":
            cols.append(np.log(np.maximum(pos.n_masked, 1).astype(np.float64)))
        elif name == "step":
            cols.append(pos.step.astype(np.float64))
        else:
            raise KeyError(name)
    return np.column_stack(cols) if cols else np.empty((len(pos), 0))


def crossfit_predictions(pos: PositionFrame, fold_of_doc: dict, n_knots: int = DEFAULT_KNOTS,
                         features=FIELD_FEATURES, interactions: bool = True,
                         weighting: str = "pool") -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Out-of-fold pi_hat(l) and pi_hat(l, s) for every position.

    ``fold_of_doc`` maps each document to a fold label.  For every fold, both
    calibrators are fitted on the other folds' positions and applied to this
    fold's positions, so no position is predicted by a model that saw it.  With
    one fold per document this is leave-one-document-out (the pilot); with
    folds {"fit", "report"} and only the report half predicted it is the plan's
    half-1 -> half-2 rule (see :func:`fit_on_half`).
    """
    folds = np.array([fold_of_doc[int(d)] for d in pos.doc_id], dtype=object)
    pi_own = np.full(len(pos), np.nan)
    pi_ctx = np.full(len(pos), np.nan)
    fits = []
    for fold in sorted(set(folds.tolist()), key=str):
        test = folds == fold
        train = ~test
        if not train.any():
            raise ValueError("cross-fitting needs at least two folds")
        tr, te = pos.select(train), pos.select(test)
        w = _weights(tr, weighting)
        own = IsotonicOwn().fit(tr.ell, tr.agree, w)
        ctx = ContextLogistic(features=features, n_knots=n_knots,
                              interactions=interactions).fit(tr, w)
        pi_own[test] = own.predict(te.ell)
        pi_ctx[test] = ctx.predict(te)
        fits.append({"fold": str(fold), "n_train": int(train.sum()), "n_test": int(test.sum()),
                     "context_features": list(ctx.used), "context_converged": ctx.converged})
    return pi_own, pi_ctx, fits


def _weights(pos: PositionFrame, weighting: str) -> np.ndarray | None:
    if weighting == "pool":
        return pool_weights(pos)
    if weighting == "position":
        return None
    raise ValueError(f"weighting must be 'pool' or 'position', got {weighting!r}")


def fit_on_half(pos: PositionFrame, fit_docs, n_knots: int = DEFAULT_KNOTS,
                features=FIELD_FEATURES, interactions: bool = True,
                weighting: str = "pool") -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit both calibrators on ``fit_docs`` and predict every position.

    Predictions on fit-half positions are in-sample and must not be reported;
    callers restrict reporting to the other half.
    """
    in_fit = np.isin(pos.doc_id, np.asarray(list(fit_docs)))
    tr = pos.select(in_fit)
    w = _weights(tr, weighting)
    own = IsotonicOwn().fit(tr.ell, tr.agree, w)
    ctx = ContextLogistic(features=features, n_knots=n_knots,
                          interactions=interactions).fit(tr, w)
    return own.predict(pos.ell), ctx.predict(pos), {"own": own.to_dict(), "context": ctx.to_dict()}
