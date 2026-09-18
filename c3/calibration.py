"""The commitment gap: a selection endpoint that survives conditioning on ell.

Eq. 2 asks whether accuracy varies with selection intensity *at matched reported
confidence*.  That is the right question for the sufficiency claim, but it is
deliberately blind to the thing a decoder designer tunes, because a threshold is
applied to the winner's confidence without matching on it.

The commitment gap is the complementary, unconditional quantity:

    Delta(m) = E[ agree | r = 1, m ]  -  E[ sigma(ell) | r = 1, m ]

the difference between how often the committed token is right and how confident
the model said it was, among winners of a field of size ``m``.  It is a pure
calibration statement about the set the decoder actually commits, and its drift
in ``m`` is the winner's curse as it is experienced at inference time.

This is reported alongside ``beta_z``, not instead of it: ``beta_z`` carries the
sufficiency claim, ``Delta(m)`` carries the operational one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame import AnalysisFrame
from .rng import generator


@dataclass
class GapEstimate:
    m: int
    gap: float
    ci_low: float
    ci_high: float
    n: int
    mean_agree: float
    mean_conf: float

    def __str__(self) -> str:
        return (f"m={self.m:3d}  Delta={self.gap:+.4f}  95% CI [{self.ci_low:+.4f}, "
                f"{self.ci_high:+.4f}]  (acc={self.mean_agree:.3f} vs "
                f"conf={self.mean_conf:.3f}, n={self.n})")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def commitment_gap(frame: AnalysisFrame, n_boot: int = 2000, seed: int = 0,
                   alpha: float = 0.05) -> list[GapEstimate]:
    """``Delta(m)`` for every pool size, with a document-clustered bootstrap."""
    w = frame.winners()
    doc_idx, n_docs = w.doc_index
    conf = _sigmoid(w.ell)
    out: list[GapEstimate] = []

    for m in sorted(set(w.m.tolist())):
        sel = w.m == m
        a, c, d = w.agree[sel], conf[sel], doc_idx[sel]
        point = float(a.mean() - c.mean())
        # Cluster bootstrap by document, implemented with multiplicity weights.
        draws = np.empty(n_boot)
        rng = generator(seed, "gap", str(m))
        for b in range(n_boot):
            cnt = rng.multinomial(n_docs, np.full(n_docs, 1.0 / n_docs))
            ww = cnt[d].astype(np.float64)
            tot = ww.sum()
            draws[b] = np.nan if tot == 0 else (ww @ a - ww @ c) / tot
        draws = draws[np.isfinite(draws)]
        lo, hi = (np.quantile(draws, [alpha / 2, 1 - alpha / 2]) if draws.size else (np.nan, np.nan))
        out.append(GapEstimate(int(m), point, float(lo), float(hi), int(sel.sum()),
                               float(a.mean()), float(c.mean())))
    return out


def gap_slope(gaps: list[GapEstimate]) -> float:
    """OLS slope of ``Delta`` on ``log2(m)``: the drift the winner's curse predicts."""
    if len(gaps) < 2:
        return float("nan")
    x = np.log2([g.m for g in gaps])
    y = np.array([g.gap for g in gaps])
    return float(np.polyfit(x, y, 1)[0])
