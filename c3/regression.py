"""Eqs. 2 and 3, with cross-fitting and the document-clustered bootstrap.

Eq. 2   Pr(agree) = sigma( alpha [+ alpha_state] + f(ell_i) + beta_z * z_{r,m} )
Eq. 3   Pr(agree) = sigma( alpha [+ alpha_state] + f(ell_i) + sum_{k>=2} gamma_k 1{r=k} )

Plan v2 fits both equations twice, with and without the state intercepts
(``fe=True`` / ``fe=False``).  Under Prop. 1 the state fixed effects absorb the
field term, so ``beta_z`` with FE is ~0 even when a winner's curse is present;
``beta_z < 0`` without FE and ~0 with FE is the field-effect signature, and
``beta_z != 0`` even with FE points to errors shared within a state.  Eq. 2 is
fitted on winning tokens (``frame.winners()``), Eq. 3 on the top ranks.

Cross-fitting.  The proposal fits coefficients on the first document half and
reports from the second.  Here that means the confidence-accuracy curve
``f(ell)`` is estimated on the fit half and carried into the eval half as a
fixed offset, so ``f`` -- which is deliberately flexible -- cannot be bent on the
same rows that produce the reported coefficient.  State fixed effects are
re-estimated on the eval half because the document split makes the two halves'
states disjoint by construction.

Inference is a document-clustered bootstrap.  Clusters are documents rather than
states or positions because a position appears in many pools and a document
contributes many states, so rows are massively dependent within a document.  The
resample is implemented by reweighting rather than by rebuilding the design,
which is algebraically the same estimating equation and much cheaper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frame import AnalysisFrame
from .logit import LogitFit, fit_fe_logit
from .order_stats import z_vector
from .rng import generator
from .splines import Spline


@dataclass
class Estimate:
    name: str
    value: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    boot: np.ndarray | None = field(default=None, repr=False)
    converged: bool | None = None     # did the point fit converge
    n_iter: int | None = None
    # Bootstrap convergence record (None when not recorded).
    n_nonconverged_draws: int | None = None   # draws that hit max_iter without converging
    n_iter_median: float | None = None
    n_iter_max: int | None = None
    warm_start: bool | None = None

    @property
    def excludes_zero(self) -> bool:
        return np.isfinite(self.ci_low) and np.isfinite(self.ci_high) and (
            (self.ci_low > 0) or (self.ci_high < 0)
        )

    @property
    def se(self) -> float:
        return float(np.std(self.boot, ddof=1)) if self.boot is not None and self.boot.size > 1 else float("nan")

    def __str__(self) -> str:
        return f"{self.name} = {self.value:+.4f}  95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]"


# The bootstrap worker is module-level, not a closure, because a closure cannot be
# pickled to a worker process.
#
# The pool uses "spawn", not "fork".  Forking a process that has already run BLAS
# deadlocks: the child inherits the parent's OpenBLAS thread pool in an
# inconsistent state and blocks on a lock no live thread will release.  It is a
# silent hang -- the parent sleeps at ~0% CPU with no children and no error --
# and it triggers reliably here because the point fit runs before the bootstrap.
# Spawn starts clean interpreters and takes the state through an initializer.
_BOOT_STATE: dict = {}


def _boot_init(state: dict) -> None:
    global _BOOT_STATE
    _BOOT_STATE = state


def _boot_worker(b: int) -> tuple[np.ndarray, int, bool]:
    """One resample: (coefficients, n_iter, converged).  n_iter=0 for a failed draw."""
    st = _BOOT_STATE
    rng = generator(st["seed"], st["label"], str(b))
    n_docs = st["n_docs"]
    counts = rng.multinomial(n_docs, np.full(n_docs, 1.0 / n_docs))
    w = counts[st["doc_idx"]].astype(np.float64)

    # Drop zero-weight rows and re-index the groups.  A resample leaves out ~37%
    # of documents, and every state belonging to one of them contributes no data
    # while still carrying a free intercept.  Left in, those intercepts are pinned
    # only by the ridge and IRLS oscillates: with 20 pilot documents every draw
    # hit the iteration cap without converging, which is both slow and wrong.
    keep = w > 0
    if not keep.any():
        return np.full(len(st["names"]), np.nan), 0, False
    uniq, g_dense = np.unique(st["groups"][keep], return_inverse=True)
    # Warm start from the full-sample fit.  Groups present in the resample take
    # their full-sample intercept (``uniq`` indexes the full-sample groups);
    # groups absent were dropped above and need no value.
    init_beta = init_alpha = None
    warm = st.get("warm")
    if warm is not None:
        init_beta, init_alpha = warm["beta"], warm["alpha"][uniq]
    try:
        f = fit_fe_logit(st["y"][keep], st["X"][keep], g_dense,
                         n_groups=int(g_dense.max()) + 1, names=st["names"],
                         offset=st["off"][keep], weights=w[keep],
                         ridge_beta=st.get("ridge_beta", 0.0),
                         max_iter=st.get("max_iter", 100),
                         init_beta=init_beta, init_alpha=init_alpha)
        return f.beta, f.n_iter, f.converged
    except (FloatingPointError, np.linalg.LinAlgError):
        return np.full(len(st["names"]), np.nan), 0, False


def _stack(res: list, state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not res:
        return np.empty((0, len(state["names"]))), np.zeros(0, int), np.zeros(0, bool)
    return (np.array([r[0] for r in res]), np.array([r[1] for r in res], dtype=int),
            np.array([r[2] for r in res], dtype=bool))


def _run_bootstrap(n_boot: int, n_jobs: int, state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (draws, n_iter, converged), one row/entry per resample."""
    global _BOOT_STATE
    _BOOT_STATE = state
    if n_jobs == 1:
        return _stack([_boot_worker(b) for b in range(n_boot)], state)
    import multiprocessing as mp
    import os

    # Pin BLAS to one thread per worker.  Each bootstrap fit solves a 9x9 system;
    # threaded BLAS contributes nothing there and the workers spend their time
    # contending instead of fitting.  Spawned children inherit this environment
    # and read it when they import numpy, which is why it is set here in the
    # parent rather than in the initializer -- by then numpy is already imported
    # and its thread pool already sized.
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    prev = {v: os.environ.get(v) for v in keys}
    os.environ.update({v: "1" for v in keys})
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_jobs, initializer=_boot_init, initargs=(state,)) as pool:
            return _stack(pool.map(_boot_worker, range(n_boot), chunksize=32), state)
    finally:
        for v, old_val in prev.items():
            if old_val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old_val


def _warm_state(point: LogitFit, warm_start: bool) -> dict | None:
    return {"beta": point.beta, "alpha": point.alpha} if warm_start else None


def _annotate(out: dict[str, Estimate], point: LogitFit, n_it: np.ndarray,
              conv: np.ndarray, warm_start: bool) -> None:
    ran = n_it > 0                      # exceptions / empty resamples are n_failed_draws
    for e in out.values():
        e.converged, e.n_iter = point.converged, point.n_iter
        e.n_nonconverged_draws = int(np.sum(ran & ~conv))
        e.n_iter_median = float(np.median(n_it[ran])) if ran.any() else float("nan")
        e.n_iter_max = int(n_it.max()) if n_it.size else 0
        e.warm_start = warm_start


def _percentiles(draws: np.ndarray, names: list[str], point: np.ndarray,
                 wanted: list[str], alpha: float) -> dict[str, Estimate]:
    out: dict[str, Estimate] = {}
    for name in wanted:
        j = names.index(name)
        col = draws[:, j]
        col = col[np.isfinite(col)]
        lo, hi = (np.quantile(col, [alpha / 2, 1 - alpha / 2]) if col.size else (np.nan, np.nan))
        out[name] = Estimate(name, float(point[j]), float(lo), float(hi), col)
    return out


def _design_eq2(frame: AnalysisFrame, spline: Spline) -> tuple[np.ndarray, list[str]]:
    z = z_vector(frame.r, frame.m)
    F = spline.transform(frame.ell)
    return np.column_stack([F, z]), spline.names() + ["beta_z"]


def _design_eq3(frame: AnalysisFrame, spline: Spline, max_rank: int) -> tuple[np.ndarray, list[str]]:
    F = spline.transform(frame.ell)
    ks = [k for k in range(2, max_rank + 1) if np.any(frame.r == k)]
    R = np.column_stack([(frame.r == k).astype(np.float64) for k in ks]) if ks else np.empty((len(frame), 0))
    return np.column_stack([F, R]), spline.names() + [f"gamma_{k}" for k in ks]


def build_design(frame: AnalysisFrame, spline: Spline, equation: str = "eq2",
                 max_rank: int = 32) -> tuple[np.ndarray, list[str]]:
    if equation == "eq2":
        return _design_eq2(frame, spline)
    if equation == "eq3":
        return _design_eq3(frame, spline, max_rank)
    raise ValueError(f"equation must be 'eq2' or 'eq3', got {equation!r}")


def _groups(frame: AnalysisFrame, fe: bool) -> tuple[np.ndarray, int]:
    """State intercepts (``fe=True``) or a single common intercept (``fe=False``)."""
    if fe:
        return frame.state_index
    return np.zeros(len(frame), dtype=np.int64), 1


def fit_once(frame: AnalysisFrame, spline: Spline, equation: str = "eq2",
             offset: np.ndarray | None = None, weights: np.ndarray | None = None,
             max_rank: int = 32, fe: bool = True, **kw) -> LogitFit:
    X, names = build_design(frame, spline, equation, max_rank)
    groups, n_groups = _groups(frame, fe)
    return fit_fe_logit(frame.agree, X, groups, n_groups=n_groups, names=names,
                        offset=offset, weights=weights, **kw)


def _spline_offset(frame: AnalysisFrame, spline: Spline, fit: LogitFit) -> np.ndarray:
    """f_hat(ell) evaluated on a new frame, using fit-half spline coefficients."""
    n_f = spline.n_columns
    return spline.transform(frame.ell) @ fit.beta[:n_f]


def cluster_bootstrap(
    frame: AnalysisFrame,
    spline: Spline,
    equation: str = "eq2",
    n_boot: int = 10_000,
    seed: int = 0,
    offset: np.ndarray | None = None,
    alpha: float = 0.05,
    max_rank: int = 32,
    n_jobs: int = 1,
    report: list[str] | None = None,
    fe: bool = True,
    ridge_beta: float = 0.0,
    warm_start: bool = True,
    max_iter: int = 100,
) -> dict[str, Estimate]:
    """Percentile CIs from a document-clustered bootstrap.

    Documents are resampled with replacement; a row's weight is the multiplicity
    of its document.  Under-represented states fall back on the fixed-effect
    ridge, which keeps their intercepts finite without touching the reported
    block.

    ``warm_start`` starts every resample's IRLS from the full-sample fit instead
    of zero.  Same estimator and tolerance (the draws agree with a cold start to
    ~1e-9 where the cold fit converges); only the number of iterations changes.
    ``warm_start=False`` is the frozen behaviour.  The number of draws that hit
    ``max_iter`` is recorded on each Estimate (``n_nonconverged_draws``).
    """
    X, names = build_design(frame, spline, equation, max_rank)
    groups, n_groups = _groups(frame, fe)
    doc_idx, n_docs = frame.doc_index
    y = frame.agree
    off = np.zeros(len(frame)) if offset is None else offset

    point = fit_fe_logit(y, X, groups, n_groups=n_groups, names=names, offset=off,
                         ridge_beta=ridge_beta, max_iter=max_iter)
    wanted = report or [n for n in names if not n.startswith("f_ell")]

    draws, n_it, conv = _run_bootstrap(n_boot, n_jobs, {
        "seed": seed, "label": "bootstrap", "n_docs": n_docs, "doc_idx": doc_idx,
        "y": y, "X": X, "groups": groups, "n_groups": n_groups, "names": names,
        "off": off, "ridge_beta": ridge_beta, "max_iter": max_iter,
        "warm": _warm_state(point, warm_start),
    })
    out = _percentiles(draws, names, point.beta, wanted, alpha)
    _annotate(out, point, n_it, conv, warm_start)
    return out


def fit_and_report(
    frame: AnalysisFrame,
    equation: str = "eq2",
    n_knots: int = 5,
    n_boot: int = 10_000,
    seed: int = 0,
    split_seed: int = 0,
    n_jobs: int = 1,
    max_rank: int = 32,
    cross_fit: bool = True,
    fe: bool = True,
    fit_docs=None,
    alpha: float = 0.05,
    ridge_beta: float = 0.0,
    warm_start: bool = True,
    max_iter: int = 100,
) -> dict[str, Estimate]:
    """The pre-registered path: fit f(ell) on half 1, report from half 2.

    ``cross_fit=False`` pools both halves into a single fit.  That is for the
    pilot only, where 20 documents will not support a split; it is not the
    reporting path.  ``fit_docs`` fixes the fit half explicitly (the committed
    split); otherwise the proposal-era seeded permutation is used.  With
    ``fe=False`` the nuisance fit and the eval fit each carry one common
    intercept, so the offset carries only the shape of f.
    """
    if not cross_fit:
        spline = Spline.fit(frame.ell, n_knots=n_knots)
        return cluster_bootstrap(frame, spline, equation, n_boot, seed, alpha=alpha,
                                 max_rank=max_rank, n_jobs=n_jobs, fe=fe,
                                 ridge_beta=ridge_beta, warm_start=warm_start,
                                 max_iter=max_iter)

    if fit_docs is not None:
        fit_half, eval_half = frame.split_by_documents(fit_docs)
    else:
        fit_half, eval_half = frame.split_by_document(0.5, seed=split_seed)
    if len(fit_half) == 0 or len(eval_half) == 0:
        raise ValueError("document split produced an empty half; too few documents")

    spline = Spline.fit(fit_half.ell, n_knots=n_knots)   # knots frozen on half 1
    nuisance = fit_once(fit_half, spline, equation, max_rank=max_rank, fe=fe,
                        ridge_beta=ridge_beta)
    offset = _spline_offset(eval_half, spline, nuisance)

    # On the eval half only the selection terms are free; f(ell) is an offset.
    z = z_vector(eval_half.r, eval_half.m)
    if equation == "eq2":
        X, names = z[:, None], ["beta_z"]
    else:
        ks = [k for k in range(2, max_rank + 1) if np.any(eval_half.r == k)]
        X = np.column_stack([(eval_half.r == k).astype(np.float64) for k in ks])
        names = [f"gamma_{k}" for k in ks]

    groups, n_groups = _groups(eval_half, fe)
    doc_idx, n_docs = eval_half.doc_index
    y = eval_half.agree
    point = fit_fe_logit(y, X, groups, n_groups=n_groups, names=names, offset=offset,
                         ridge_beta=ridge_beta, max_iter=max_iter)

    draws, n_it, conv = _run_bootstrap(n_boot, n_jobs, {
        "seed": seed, "label": "bootstrap-xfit", "n_docs": n_docs, "doc_idx": doc_idx,
        "y": y, "X": X, "groups": groups, "n_groups": n_groups, "names": names,
        "off": offset, "ridge_beta": ridge_beta, "max_iter": max_iter,
        "warm": _warm_state(point, warm_start),
    })
    out = _percentiles(draws, names, point.beta, names, alpha)
    _annotate(out, point, n_it, conv, warm_start)
    return out
