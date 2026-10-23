"""The analysis tables.

``AnalysisFrame`` has one row per (pool, retained member) and is what the pooled
endpoints (Delta(m), Eqs. 2-3) are computed from.  ``PositionFrame`` has one row
per (masked state, masked position) -- the trace itself -- and is what the
calibrators, ECE and the field features are computed from.  ``src_row`` links a
pool row back to the position row it was drawn from, so anything fitted on
positions (a calibrator prediction, a stored confidence) can be looked up for
pool winners without re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from .rng import derive_seed


@dataclass
class AnalysisFrame:
    doc_id: np.ndarray      # clustering unit for the bootstrap
    state_id: np.ndarray    # masked state == fixed-effect group
    position: np.ndarray    # token index within the document
    ell: np.ndarray         # reported confidence, log-odds of the top-1 token
    agree: np.ndarray       # 1{v_i == x0_i}
    m: np.ndarray           # pool size
    r: np.ndarray           # rank within pool, 1 = winner
    src_row: np.ndarray | None = None   # row of the PositionFrame this was drawn from

    def __post_init__(self) -> None:
        n = self.doc_id.size
        for name in ("state_id", "position", "ell", "agree", "m", "r", "src_row"):
            col = getattr(self, name)
            if col is None:
                continue
            if col.size != n:
                raise ValueError(f"column {name!r} has {col.size} rows, expected {n}")

    def __len__(self) -> int:
        return int(self.doc_id.size)

    def select(self, mask: np.ndarray) -> "AnalysisFrame":
        mask = np.asarray(mask)
        return replace(self, **{
            f.name: getattr(self, f.name)[mask]
            for f in fields(self) if getattr(self, f.name) is not None
        })

    def winners(self) -> "AnalysisFrame":
        return self.select(self.r == 1)

    @property
    def state_index(self) -> tuple[np.ndarray, int]:
        """Dense 0-based fixed-effect group index and the number of groups."""
        codes, inv = np.unique(self.state_id, return_inverse=True)
        return inv.astype(np.int64), int(codes.size)

    @property
    def doc_index(self) -> tuple[np.ndarray, int]:
        codes, inv = np.unique(self.doc_id, return_inverse=True)
        return inv.astype(np.int64), int(codes.size)

    def split_by_document(self, frac: float = 0.5, seed: int = 0) -> tuple["AnalysisFrame", "AnalysisFrame"]:
        """Document-level split by a seeded permutation of the documents present.

        The halves depend on which documents are in this frame, so two frames
        with different document sets get different splits.  The reported
        analyses use :meth:`split_by_documents` with a fixed assignment instead
        (see :func:`hash_halves`); this remains for the simulator and the
        proposal-era scripts.
        """
        from .rng import generator
        docs = np.unique(self.doc_id)
        rng = generator(seed, "split")
        perm = rng.permutation(docs.size)
        n_fit = int(round(frac * docs.size))
        return self.split_by_documents(docs[perm[:n_fit]])

    def split_by_documents(self, fit_docs) -> tuple["AnalysisFrame", "AnalysisFrame"]:
        """Split into (fit, eval) by an explicit set of fit-half documents."""
        in_fit = np.isin(self.doc_id, np.asarray(list(fit_docs)))
        return self.select(in_fit), self.select(~in_fit)


def hash_halves(doc_ids, seed: int = 0) -> np.ndarray:
    """A fixed 0/1 half for every document, a function of (seed, doc_id) only.

    Unlike a permutation of the documents present, this cannot change when a
    document is dropped (a contamination flag, a model that failed on it), so
    fit-half and report-half documents never trade places across analyses.  The
    halves are balanced in expectation, not exactly; the committed split manifest
    replaces this once the corpus manifest exists.
    """
    return np.array([derive_seed(seed, "half", str(int(d))) & 1 for d in np.asarray(doc_ids)],
                    dtype=np.int64)


def nested_doc_subsets(doc_ids, sizes, seed: int = 0) -> dict:
    """{size: sorted ids} with every smaller subset inside every larger one (T13 ablation 3).

    Documents are ordered by a hash of (seed, doc_id) and each subset is a prefix of
    that order, so a curve over sizes adds documents rather than swapping them, and a
    document's place does not depend on which other documents are present.  A size
    above the number of documents is refused.
    """
    ids = np.unique(np.asarray(doc_ids, np.int64))
    order = ids[np.argsort([derive_seed(seed, "nested", str(int(d))) for d in ids], kind="stable")]
    out = {}
    for n in sorted(int(s) for s in sizes):
        if n > ids.size:
            raise ValueError(f"asked for {n} documents, only {ids.size} available")
        out[n] = np.sort(order[:n])
    return out


# Per-state covariates the context-aware calibrator may use (plan v2 T6(2)).
FIELD_FEATURES = ("median_ell", "mask_rate", "log_n_masked", "step")


@dataclass
class PositionFrame:
    """One row per (masked state, masked position): the trace itself."""

    doc_id: np.ndarray
    state_id: np.ndarray
    position: np.ndarray
    ell: np.ndarray
    conf: np.ndarray
    agree: np.ndarray
    mask_kind: np.ndarray
    mask_rate: np.ndarray
    n_masked: np.ndarray
    step: np.ndarray

    def __post_init__(self) -> None:
        n = self.doc_id.size
        for f in fields(self):
            if getattr(self, f.name).size != n:
                raise ValueError(f"column {f.name!r} has {getattr(self, f.name).size} rows, expected {n}")

    def __len__(self) -> int:
        return int(self.doc_id.size)

    @classmethod
    def from_frame(cls, df) -> "PositionFrame":
        """Build from a trace DataFrame (the columns ``collect`` writes).

        ``conf`` falls back to sigma(ell) for traces that do not store it; the
        v1 pilot traces do store it, and Delta is defined on the stored value.
        """
        ell = df["ell"].to_numpy(np.float64)
        conf = (df["conf"].to_numpy(np.float64) if "conf" in df.columns
                else 1.0 / (1.0 + np.exp(-ell)))
        n = len(df)

        def col(name, default, dtype):
            return df[name].to_numpy(dtype) if name in df.columns else np.full(n, default, dtype=dtype)

        return cls(
            doc_id=df["doc_id"].to_numpy(np.int64),
            state_id=df["state_id"].to_numpy(np.int64),
            position=df["position"].to_numpy(np.int64),
            ell=ell, conf=conf,
            agree=df["agree"].to_numpy(np.float64),
            mask_kind=col("mask_kind", "na", object),
            mask_rate=col("mask_rate", np.nan, np.float64),
            n_masked=col("n_masked", -1, np.int64),
            step=col("step", 0, np.int64),
        )

    def select(self, mask: np.ndarray) -> "PositionFrame":
        mask = np.asarray(mask)
        return replace(self, **{f.name: getattr(self, f.name)[mask] for f in fields(self)})

    @property
    def doc_index(self) -> tuple[np.ndarray, int]:
        codes, inv = np.unique(self.doc_id, return_inverse=True)
        return inv.astype(np.int64), int(codes.size)

    @property
    def state_index(self) -> tuple[np.ndarray, int]:
        codes, inv = np.unique(self.state_id, return_inverse=True)
        return inv.astype(np.int64), int(codes.size)

    def state_median_ell(self) -> np.ndarray:
        """Per row: the median ell over every masked position of the row's state."""
        idx, n_states = self.state_index
        order = np.argsort(idx, kind="stable")
        bounds = np.flatnonzero(np.diff(idx[order])) + 1
        med = np.empty(n_states)
        for s, rows in enumerate(np.split(order, bounds)):
            med[s] = np.median(self.ell[rows])
        return med[idx]

    def field_features(self, names=FIELD_FEATURES) -> tuple[np.ndarray, list[str]]:
        """Per-state covariates as a dense matrix, constant columns dropped.

        A feature that does not vary in this frame (step is always 0 for uniform
        masks) carries no information and would only make the fit singular.
        """
        cols, used = [], []
        for name in names:
            if name == "median_ell":
                v = self.state_median_ell()
            elif name == "mask_rate":
                v = self.mask_rate.astype(np.float64)
            elif name == "log_n_masked":
                v = np.log(np.maximum(self.n_masked, 1).astype(np.float64))
            elif name == "step":
                v = self.step.astype(np.float64)
            else:
                raise KeyError(f"unknown field feature {name!r}; known: {FIELD_FEATURES}")
            if np.all(np.isfinite(v)) and np.ptp(v) > 0:
                cols.append(v)
                used.append(name)
        X = np.column_stack(cols) if cols else np.empty((len(self), 0))
        return X, used
