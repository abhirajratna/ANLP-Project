"""Calibration metrics (plan v2 Sec. 5): equal-mass ECE, smooth ECE, reliability
bins, and isotonic regression.

Everything is plain numpy so it runs in the existing cluster environment and
locally without scikit-learn.  ECE bins are assigned once, on the point sample,
and the document bootstrap only reweights documents -- re-binning inside every
resample would make the bin edges part of the noise being measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# numpy >= 2 renamed trapz; keep working on the 1.26 floor in pyproject.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# Isotonic regression (pool adjacent violators)
# --------------------------------------------------------------------------- #

@dataclass
class IsotonicFit:
    """A non-decreasing fit of y on x, evaluated by linear interpolation between
    the fitted points and held flat beyond them (sklearn's ``clip`` behaviour)."""

    x: np.ndarray       # distinct fitted abscissae, increasing
    y: np.ndarray       # fitted values at x, non-decreasing

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(x_new, dtype=np.float64), self.x, self.y)

    def to_dict(self) -> dict:
        return {"x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicFit":
        return cls(np.asarray(d["x"], dtype=np.float64), np.asarray(d["y"], dtype=np.float64))


def isotonic_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> IsotonicFit:
    """Weighted least-squares non-decreasing regression of ``y`` on ``x``.

    Ties in ``x`` are merged first (their weighted mean is the only value a
    function of x can take), then adjacent violators are pooled with a stack.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=np.float64).ravel()
    if not (x.size == y.size == w.size):
        raise ValueError("x, y and w must have the same length")
    keep = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[keep], y[keep], w[keep]
    if x.size == 0:
        raise ValueError("no finite observations to fit")

    ux, inv = np.unique(x, return_inverse=True)
    sw = np.bincount(inv, weights=w)
    sy = np.bincount(inv, weights=w * y)

    # Stack of blocks: (sum_w, sum_wy, n_points)
    bw: list[float] = []
    by: list[float] = []
    bn: list[int] = []
    for k in range(ux.size):
        cw, cy, cn = sw[k], sy[k], 1
        while bw and by[-1] / bw[-1] > cy / cw:
            cw += bw.pop()
            cy += by.pop()
            cn += bn.pop()
        bw.append(cw); by.append(cy); bn.append(cn)
    fitted = np.repeat(np.array(by) / np.array(bw), bn)
    return IsotonicFit(ux, fitted)


# --------------------------------------------------------------------------- #
# Equal-mass ECE
# --------------------------------------------------------------------------- #

def equal_mass_bins(conf: np.ndarray, n_bins: int = 15) -> np.ndarray:
    """Bin index for every row so that bins hold (as near as possible) equal counts.

    Assignment is by rank, not by quantile edges: saturated confidence produces
    long runs of identical values, and quantile edges would then collapse into
    empty and overfull bins.  Ties are ordered by position in the array, which is
    arbitrary but fixed, so the assignment is deterministic.
    """
    conf = np.asarray(conf, dtype=np.float64)
    n = conf.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(conf, kind="stable")
    b = np.empty(n, dtype=np.int64)
    b[order] = np.minimum((np.arange(n) * n_bins) // n, n_bins - 1)
    return b


@dataclass
class ECEResult:
    ece: float
    ci: tuple[float, float]
    draws: np.ndarray
    bin_conf: np.ndarray      # mean confidence per bin (point sample)
    bin_acc: np.ndarray       # accuracy per bin (point sample)
    bin_n: np.ndarray         # rows per bin

    def to_dict(self) -> dict:
        return {"ece": self.ece, "ci_low": self.ci[0], "ci_high": self.ci[1],
                "bins": [{"conf": float(c), "acc": float(a), "n": int(n)}
                         for c, a, n in zip(self.bin_conf, self.bin_acc, self.bin_n)]}


def ece_equal_mass(conf: np.ndarray, agree: np.ndarray, n_bins: int = 15,
                   doc_idx: np.ndarray | None = None, counts: np.ndarray | None = None,
                   alpha: float = 0.05) -> ECEResult:
    """Expected calibration error with equal-mass bins, and a document-bootstrap CI.

    ECE = sum_b (n_b / N) |mean conf_b - accuracy_b|.  With ``doc_idx`` and
    ``counts`` (from :func:`c3.stats.doc_boot_counts`) the interval resamples
    documents with the bin assignment held fixed.
    """
    conf = np.asarray(conf, dtype=np.float64)
    agree = np.asarray(agree, dtype=np.float64)
    b = equal_mass_bins(conf, n_bins)
    n_b = np.bincount(b, minlength=n_bins).astype(np.float64)
    sc = np.bincount(b, weights=conf, minlength=n_bins)
    sa = np.bincount(b, weights=agree, minlength=n_bins)
    ece = float(np.abs(sc - sa).sum() / max(conf.size, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_conf, bin_acc = sc / n_b, sa / n_b

    draws = np.empty(0)
    ci = (float("nan"), float("nan"))
    if doc_idx is not None and counts is not None:
        from .stats import percentile_ci
        n_docs = counts.shape[1]
        flat = np.asarray(doc_idx, dtype=np.int64) * n_bins + b
        size = n_docs * n_bins
        Sc = np.bincount(flat, weights=conf, minlength=size).reshape(n_docs, n_bins)
        Sa = np.bincount(flat, weights=agree, minlength=size).reshape(n_docs, n_bins)
        Nn = np.bincount(flat, minlength=size).reshape(n_docs, n_bins).astype(np.float64)
        C = counts.astype(np.float64)
        diff = np.abs(C @ (Sc - Sa))                 # (B, n_bins)
        tot = C @ Nn.sum(axis=1)                     # (B,)
        draws = diff.sum(axis=1) / np.maximum(tot, 1.0)
        ci = percentile_ci(draws, alpha)
    return ECEResult(ece, ci, draws, bin_conf, bin_acc, n_b.astype(np.int64))


# --------------------------------------------------------------------------- #
# Smooth ECE (Blasiok & Nakkiran, 2023)
# --------------------------------------------------------------------------- #

def _reflected_kernel(t: np.ndarray, f: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian kernel on [0, 1] with reflection at both ends, shape (len(t), len(f))."""
    d = t[:, None] - f[None, :]
    s = t[:, None] + f[None, :]
    k = np.zeros_like(d)
    for shift in (-2.0, 0.0, 2.0):
        k += np.exp(-0.5 * ((d - shift) / sigma) ** 2)
        k += np.exp(-0.5 * ((s - shift) / sigma) ** 2)
    return k / (sigma * np.sqrt(2 * np.pi))


def smooth_ece_at(conf: np.ndarray, agree: np.ndarray, sigma: float,
                  n_grid: int = 1001, n_hist: int = 2000) -> float:
    """smECE at a fixed bandwidth: (1/n) * integral over t of |sum_i K(t, f_i)(y_i - f_i)|.

    Predictions are pre-binned onto a fine histogram so the cost does not grow
    with the number of positions.
    """
    conf = np.clip(np.asarray(conf, dtype=np.float64), 0.0, 1.0)
    resid = np.asarray(agree, dtype=np.float64) - conf
    h = np.minimum((conf * n_hist).astype(np.int64), n_hist - 1)
    centres = (np.arange(n_hist) + 0.5) / n_hist
    r_bin = np.bincount(h, weights=resid, minlength=n_hist)
    used = r_bin != 0
    t = np.linspace(0.0, 1.0, n_grid)
    K = _reflected_kernel(t, centres[used], sigma)
    curve = np.abs(K @ r_bin[used])
    return float(_trapezoid(curve, t) / max(conf.size, 1))


def smooth_ece(conf: np.ndarray, agree: np.ndarray, tol: float = 1e-4) -> tuple[float, float]:
    """smECE at its fixed-point bandwidth sigma* where smECE(sigma*) = sigma*.

    smECE(sigma) decreases in sigma, so the fixed point is found by bisection.
    Returns ``(smECE, sigma*)``.
    """
    lo, hi = 1e-4, 1.0
    if smooth_ece_at(conf, agree, lo) <= lo:
        return smooth_ece_at(conf, agree, lo), lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if smooth_ece_at(conf, agree, mid) > mid:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    s = 0.5 * (lo + hi)
    return smooth_ece_at(conf, agree, s), s


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_reliability(ax, results: dict, title: str = "") -> None:
    """Reliability diagram from ``{label: ECEResult}`` on a matplotlib axis."""
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", zorder=0)
    for label, r in results.items():
        ax.plot(r.bin_conf, r.bin_acc, marker="o", ms=3, lw=1.2,
                label=f"{label} (ECE {r.ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
