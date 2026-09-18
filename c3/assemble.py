"""Turn a stored position-level trace into an :class:`AnalysisFrame`.

Pools are drawn here rather than at collection time, so the pool design is a
free parameter of the analysis and costs no forward passes to change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .design import draw_pools
from .frame import AnalysisFrame


def frame_from_arrays(doc_id, state_id, position, ell, agree,
                      pool_sizes=(2, 4, 8, 16, 32), n_pools_per_size: int = 16,
                      seed: int = 0, retain: int | str = "all") -> AnalysisFrame:
    doc_id = np.asarray(doc_id, dtype=np.int64)
    state_id = np.asarray(state_id, dtype=np.int64)
    position = np.asarray(position, dtype=np.int64)
    ell = np.asarray(ell, dtype=np.float64)
    agree = np.asarray(agree, dtype=np.float64)

    order = np.argsort(state_id, kind="stable")
    doc_id, state_id, position, ell, agree = (a[order] for a in (doc_id, state_id, position, ell, agree))
    bounds = np.flatnonzero(np.diff(state_id)) + 1
    slices = np.split(np.arange(state_id.size), bounds)

    D, S, P, E, A, M, R = [], [], [], [], [], [], []
    for idx in slices:
        if idx.size == 0:
            continue
        pd = draw_pools(ell[idx], pool_sizes, n_pools_per_size, seed=seed,
                        state_key=str(int(state_id[idx[0]])), retain=retain)
        if len(pd) == 0:
            continue
        rows = idx[pd.state_row]
        D.append(doc_id[rows]); S.append(state_id[rows]); P.append(position[rows])
        E.append(ell[rows]); A.append(agree[rows]); M.append(pd.m); R.append(pd.r)

    if not D:
        raise ValueError("no state had enough masked positions to form a pool")
    return AnalysisFrame(
        doc_id=np.concatenate(D), state_id=np.concatenate(S),
        position=np.concatenate(P), ell=np.concatenate(E),
        agree=np.concatenate(A), m=np.concatenate(M), r=np.concatenate(R))


def load_traces(path: str | Path, mask_kind: str | None = None,
                model: str | None = None, precision: str | None = None):
    """Read a parquet trace, optionally filtered to one arm of the design."""
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    df = t.to_pandas()
    if mask_kind is not None:
        df = df[df["mask_kind"] == mask_kind]
    if model is not None:
        df = df[df["model"] == model]
    if precision is not None:
        df = df[df["precision"] == precision]
    return df


def frame_from_traces(path_or_df, mask_kind: str | None = None, model: str | None = None,
                      precision: str | None = None, **pool_kw) -> AnalysisFrame:
    df = path_or_df
    if isinstance(df, (str, Path)):
        df = load_traces(df, mask_kind=mask_kind, model=model, precision=precision)
    if len(df) == 0:
        raise ValueError("trace selection is empty")
    return frame_from_arrays(df["doc_id"].to_numpy(), df["state_id"].to_numpy(),
                             df["position"].to_numpy(), df["ell"].to_numpy(),
                             df["agree"].to_numpy(), **pool_kw)
