"""Document-level inference shared by every endpoint (plan v2 Sec. 5).

Three rules drive the design:

* **The document is the unit.**  Pool draws and positions within a document are
  not independent, so estimates average within a document first and intervals
  resample whole documents.
* **One resample matrix per run.**  Every estimand in a run (Delta at each m, the
  field term, ECE, ...) is evaluated on the *same* document resamples.  A
  contrast such as Delta(32) - Delta(1) then gets a valid interval by
  differencing draws; independent resamples per estimand would not give one.
* **Declaring an effect needs two things:** the 95% interval excludes zero and
  the effect exceeds twice the non-determinism floor.

A resample is represented by multinomial document counts, which is the same
reweighting the regression bootstrap uses (``regression._boot_worker``), so a
statistic computed from per-document sums becomes one matrix product per
estimand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import N_BOOT
from .rng import generator


def doc_boot_counts(n_docs: int, n_boot: int = N_BOOT, seed: int = 0,
                    label: str = "doc-bootstrap") -> np.ndarray:
    """``(n_boot, n_docs)`` multinomial resample counts, drawn once per run."""
    if n_docs <= 0:
        raise ValueError("need at least one document")
    rng = generator(seed, label)
    dtype = np.uint16 if n_docs < 65_535 else np.uint32
    return rng.multinomial(n_docs, np.full(n_docs, 1.0 / n_docs), size=n_boot).astype(dtype)


def percentile_ci(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def boot_pvalue(draws: np.ndarray, null: float = 0.0) -> float:
    """Two-sided bootstrap p-value for H0: estimand == ``null``.

    Twice the smaller tail mass beyond the null, with the (1 + count)/(B + 1)
    correction so it is never exactly zero; at B = 10,000 the smallest value is
    about 2e-4, fine for a Holm family of a few dozen tests.
    """
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    lo = (1 + np.sum(d <= null)) / (d.size + 1)
    hi = (1 + np.sum(d >= null)) / (d.size + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(pvalues, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down.  Returns (adjusted p-values, reject flags)."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, (n - np.arange(n)) * p[order]))
    adj = np.empty(n)
    adj[order] = adj_sorted
    return adj, adj <= alpha


def declare(value: float, ci: tuple[float, float], floor: float) -> bool:
    """Plan v2 Sec. 5: the interval excludes zero AND |effect| > 2 x floor."""
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(value)):
        return False
    return bool((lo > 0 or hi < 0) and abs(value) > 2.0 * abs(floor))


@dataclass
class GroupedMean:
    """A document-first mean for each level of a grouping key (e.g. pool size m).

    ``per_doc[j, g]`` is document j's mean at level g (NaN where the document has
    no rows at that level); the estimate at level g is the unweighted mean over
    documents present.  ``draws[b, g]`` is the same statistic on resample b.
    """

    levels: np.ndarray
    point: np.ndarray
    draws: np.ndarray
    per_doc: np.ndarray
    n_docs: np.ndarray
    n_rows: np.ndarray
    alpha: float = 0.05
    _ci: np.ndarray | None = field(default=None, repr=False)

    def ci(self) -> np.ndarray:
        if self._ci is None:
            self._ci = np.array([percentile_ci(self.draws[:, g], self.alpha)
                                 for g in range(self.levels.size)])
        return self._ci

    def index(self, level) -> int:
        hits = np.flatnonzero(self.levels == level)
        if hits.size == 0:
            raise KeyError(f"level {level!r} not present; have {self.levels.tolist()}")
        return int(hits[0])

    def contrast(self, hi, lo) -> tuple[float, np.ndarray]:
        """Point estimate and joint draws of estimate(hi) - estimate(lo)."""
        i, j = self.index(hi), self.index(lo)
        return float(self.point[i] - self.point[j]), self.draws[:, i] - self.draws[:, j]

    def per_doc_contrast(self, hi, lo) -> np.ndarray:
        """Document-level differences, for documents present at both levels."""
        i, j = self.index(hi), self.index(lo)
        d = self.per_doc[:, i] - self.per_doc[:, j]
        return d[np.isfinite(d)]

    def to_records(self, name: str = "value") -> list[dict]:
        ci = self.ci()
        return [{"level": _py(self.levels[g]), name: float(self.point[g]),
                 "ci_low": float(ci[g, 0]), "ci_high": float(ci[g, 1]),
                 "n_docs": int(self.n_docs[g]), "n_rows": int(self.n_rows[g])}
                for g in range(self.levels.size)]


def _py(x):
    return x.item() if hasattr(x, "item") else x


def grouped_doc_mean(values: np.ndarray, doc_idx: np.ndarray, n_docs: int,
                     group: np.ndarray, counts: np.ndarray,
                     alpha: float = 0.05, weights: np.ndarray | None = None) -> GroupedMean:
    """Document-first mean of ``values`` at every level of ``group``.

    ``doc_idx`` is a dense 0-based document index into the columns of
    ``counts``, so that every estimand of a run shares the same resamples.
    ``weights`` (optional, per row) makes each document's mean a weighted mean
    within that document; documents are still weighted equally.
    """
    values = np.asarray(values, dtype=np.float64)
    doc_idx = np.asarray(doc_idx, dtype=np.int64)
    group = np.asarray(group)
    w = np.ones(values.size) if weights is None else np.asarray(weights, dtype=np.float64)
    if not (values.size == doc_idx.size == group.size == w.size):
        raise ValueError("values, doc_idx, group and weights must have the same length")
    if counts.shape[1] != n_docs:
        raise ValueError(f"counts has {counts.shape[1]} documents, expected {n_docs}")

    levels, g_idx = np.unique(group, return_inverse=True)
    G = levels.size
    flat = doc_idx * G + g_idx
    sums = np.bincount(flat, weights=w * values, minlength=n_docs * G).reshape(n_docs, G)
    wsum = np.bincount(flat, weights=w, minlength=n_docs * G).reshape(n_docs, G)
    cnts = np.bincount(flat, minlength=n_docs * G).reshape(n_docs, G).astype(np.float64)
    present = wsum > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        per_doc = np.where(present, sums / np.where(present, wsum, 1.0), np.nan)

    means0 = np.where(present, per_doc, 0.0)
    pres = present.astype(np.float64)
    point = means0.sum(axis=0) / np.maximum(pres.sum(axis=0), 1)
    C = counts.astype(np.float64)
    num = C @ means0
    den = C @ pres
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return GroupedMean(levels=levels, point=point, draws=draws, per_doc=per_doc,
                       n_docs=present.sum(axis=0), n_rows=cnts.sum(axis=0).astype(np.int64),
                       alpha=alpha)


def doc_mean(values: np.ndarray, doc_idx: np.ndarray, n_docs: int,
             counts: np.ndarray, alpha: float = 0.05) -> GroupedMean:
    """Document-first mean with no grouping (a single level)."""
    return grouped_doc_mean(values, doc_idx, n_docs, np.zeros(np.asarray(values).size, np.int64),
                            counts, alpha)
