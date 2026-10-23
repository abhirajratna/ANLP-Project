"""Recalibration methods behind one interface (plan v2 T13 ablation 2).

"Which recalibration method is enough?"  Each method maps a masked position's
top-1 log-odds ell (and, for some, its state) to pi_hat = Pr(top-1 is right), is
fitted on fit-half positions with pool weights (``calibrators.pool_weights``),
and can then be read two ways:

* on traces, through ``predict(pos)`` (``scripts/compare_recalibration.py``):
  held-out ECE, log-loss, Brier, and the committed-token gap left after
  recalibration, E_m[pi_hat - a] over pool winners;
* inside a decoder, through ``predict_arrays(ell, field)`` -- the interface the
  calibrated policies (``c3.decode.policies.C3a``, ``C3Indep``, ``C3Set``) call --
  so C3 can run on any of them (``scripts/eval_infill.py --calibrator``).

Methods, from least to most informed:

    none              pi_hat = sigma(ell), the model's own confidence
    temperature       sigma(ell / T), one T for every state (Guo et al., 2017)
    temperature_rate  sigma(ell / T_b), one T per mask-rate bin -- the T8 per-step
                      temperature (``c3.controls.TEMPERATURE_EDGES``)
    histogram         equal-mass binning of ell, 15 bins (Zadrozny & Elkan, 2001)
    isotonic          isotonic regression on ell (``calibrators.IsotonicOwn``)
    context           logistic in a spline of ell plus field features
                      (``calibrators.ContextLogistic``, the C3 token calibrator)

The first five see only the token's own confidence, except temperature_rate,
which also sees the state's mask rate.  The temperatures are fitted by the NLL of
correctness on the top-1 log-odds (exact from the trace; T8's primary row).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibrators import ContextLogistic, IsotonicOwn
from .endpoints.fieldtest import DECODE_FEATURES
from .frame import PositionFrame
from .logit import _sigmoid
from .metrics import IsotonicFit

METHODS = ("none", "temperature", "temperature_rate", "histogram", "isotonic", "context")


def _norm(weights, n: int) -> np.ndarray:
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    return w / w.sum()


@dataclass
class Identity:
    """pi_hat = sigma(ell): no recalibration."""

    kind: str = "none"

    def fit(self, pos: PositionFrame, weights=None) -> "Identity":
        return self

    def predict(self, pos: PositionFrame) -> np.ndarray:
        return _sigmoid(np.asarray(pos.ell, np.float64))

    def predict_arrays(self, ell, field: dict | None = None) -> np.ndarray:
        return _sigmoid(np.asarray(ell, np.float64))

    def to_dict(self) -> dict:
        return {"kind": self.kind}


@dataclass
class GlobalTemperature:
    """pi_hat = sigma(ell / T), T fitted by the weighted NLL of correctness."""

    T: float = 1.0
    kind: str = "temperature"

    def fit(self, pos: PositionFrame, weights=None) -> "GlobalTemperature":
        from .controls import _minimise_log_t

        e, y = np.asarray(pos.ell, np.float64), np.asarray(pos.agree, np.float64)
        w = _norm(weights, e.size)

        def nll(lt):
            z = e / np.exp(lt)
            return float(np.sum(w * (np.logaddexp(0, z) - y * z)))

        self.T = _minimise_log_t(nll)
        return self

    def predict(self, pos: PositionFrame) -> np.ndarray:
        return self.predict_arrays(pos.ell)

    def predict_arrays(self, ell, field: dict | None = None) -> np.ndarray:
        return _sigmoid(np.asarray(ell, np.float64) / self.T)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "T": self.T}


@dataclass
class RateTemperature:
    """pi_hat = sigma(ell / T_b), b the mask-rate bin (``controls.TEMPERATURE_EDGES``)."""

    T: np.ndarray | None = None
    kind: str = "temperature_rate"
    used: tuple = ("mask_rate",)

    def fit(self, pos: PositionFrame, weights=None) -> "RateTemperature":
        from .controls import fit_temperatures

        w = np.ones(len(pos)) if weights is None else np.asarray(weights, np.float64)
        self.T = fit_temperatures("logodds", pos.mask_rate, w, ell=pos.ell, agree=pos.agree)
        return self

    def predict(self, pos: PositionFrame) -> np.ndarray:
        return self.predict_arrays(pos.ell, {"mask_rate": pos.mask_rate})

    def predict_arrays(self, ell, field: dict | None = None) -> np.ndarray:
        from .controls import temperature_bin

        if not field or "mask_rate" not in field:
            raise KeyError("temperature_rate needs the state's mask_rate")
        ell = np.asarray(ell, np.float64)
        rate = np.broadcast_to(np.asarray(field["mask_rate"], np.float64), ell.shape)
        return _sigmoid(ell / self.T[temperature_bin(rate)])

    def to_dict(self) -> dict:
        return {"kind": self.kind, "T": np.asarray(self.T).tolist()}


@dataclass
class HistogramBinning:
    """pi_hat = the weighted accuracy of ell's equal-mass bin on the fit positions.

    Bin edges are weighted quantiles of ell, so each bin holds about 1/n_bins of
    the pool weight; a run of tied ell never straddles an edge, so with heavy ties
    there may be fewer bins.  Values beyond the fitted range fall in the end bins.
    """

    n_bins: int = 15
    edges: np.ndarray | None = None     # interior edges, increasing
    values: np.ndarray | None = None    # accuracy per bin
    kind: str = "histogram"

    def fit(self, pos: PositionFrame, weights=None) -> "HistogramBinning":
        e, y = np.asarray(pos.ell, np.float64), np.asarray(pos.agree, np.float64)
        w = _norm(weights, e.size)
        o = np.argsort(e, kind="stable")
        cw = np.cumsum(w[o])
        cut = np.searchsorted(cw, np.arange(1, self.n_bins) / self.n_bins, side="left")
        cand = e[o][np.minimum(cut, e.size - 1)]
        # an edge at value v sends v to the upper bin; drop edges that would leave
        # a bin empty (ties) or sit at the minimum
        self.edges = np.unique(cand[cand > e[o][0]])
        b = np.searchsorted(self.edges, e, side="right")
        k = self.edges.size + 1
        sw = np.bincount(b, weights=w, minlength=k)
        sy = np.bincount(b, weights=w * y, minlength=k)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.values = np.where(sw > 0, sy / sw, np.nan)
        if np.isnan(self.values).any():           # cannot happen with data-driven edges
            raise ValueError("histogram binning left an empty bin")
        return self

    def predict(self, pos: PositionFrame) -> np.ndarray:
        return self.predict_arrays(pos.ell)

    def predict_arrays(self, ell, field: dict | None = None) -> np.ndarray:
        return self.values[np.searchsorted(self.edges, np.asarray(ell, np.float64), side="right")]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "n_bins": self.n_bins, "edges": self.edges.tolist(),
                "values": self.values.tolist()}


@dataclass
class Isotonic:
    """pi_hat = isotonic regression of correctness on ell (``calibrators.IsotonicOwn``)."""

    own: IsotonicOwn = field(default_factory=IsotonicOwn)
    kind: str = "isotonic"

    def fit(self, pos: PositionFrame, weights=None) -> "Isotonic":
        self.own.fit(pos.ell, pos.agree, weights)
        return self

    def predict(self, pos: PositionFrame) -> np.ndarray:
        return self.own.predict(pos.ell)

    def predict_arrays(self, ell, field: dict | None = None) -> np.ndarray:
        return self.own.predict(np.asarray(ell, np.float64))

    def to_dict(self) -> dict:
        return {"kind": self.kind, **self.own.fit_.to_dict()}


def make(method: str, features=DECODE_FEATURES, **kw):
    """An unfitted recalibrator.  ``context`` uses the decode-time field features by
    default, the ones a decoder can compute at a step."""
    if method == "none":
        return Identity()
    if method == "temperature":
        return GlobalTemperature()
    if method == "temperature_rate":
        return RateTemperature()
    if method == "histogram":
        return HistogramBinning(**kw)
    if method == "isotonic":
        return Isotonic()
    if method == "context":
        return ContextLogistic(features=tuple(features), **kw)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def from_dict(d: dict):
    kind = d.get("kind")
    if kind == "context_logistic":
        return ContextLogistic.from_dict(d)
    if kind == "none":
        return Identity()
    if kind == "temperature":
        return GlobalTemperature(T=float(d["T"]))
    if kind == "temperature_rate":
        return RateTemperature(T=np.asarray(d["T"], np.float64))
    if kind == "histogram":
        return HistogramBinning(n_bins=int(d["n_bins"]), edges=np.asarray(d["edges"], np.float64),
                                values=np.asarray(d["values"], np.float64))
    if kind == "isotonic":
        return Isotonic(own=IsotonicOwn(fit_=IsotonicFit.from_dict(d)))
    raise ValueError(f"unknown calibrator kind {kind!r}")


def save(cal, path) -> None:
    Path(path).write_text(json.dumps(cal.to_dict(), indent=1))


def load(path):
    """Any calibrator JSON this module or ``ContextLogistic.save`` writes."""
    return from_dict(json.loads(Path(path).read_text()))
