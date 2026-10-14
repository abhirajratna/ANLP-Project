"""Turn a stored position-level trace into analysis tables.

Pools are drawn here rather than at collection time, so the pool design is a
free parameter of the analysis and costs no forward passes to change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import N_POOLS, POOL_SIZES
from .design import draw_pools
from .frame import AnalysisFrame, PositionFrame


def frame_from_arrays(doc_id, state_id, position, ell, agree,
                      pool_sizes=POOL_SIZES, n_pools_per_size: int = N_POOLS,
                      seed: int = 0, retain: int | str = "all",
                      stream: str = "per_m", min_state_size: int = 1) -> AnalysisFrame:
    """Draw pools in every state and return the pool-row frame.

    ``src_row`` on the result indexes the input arrays, i.e. the row of the
    trace a pool member came from.  ``min_state_size`` drops states with fewer
    masked positions, so that every pool size averages over the same states;
    otherwise large-m estimates come only from large states and the trend in m
    is confounded with state composition.
    """
    doc_id = np.asarray(doc_id, dtype=np.int64)
    state_id = np.asarray(state_id, dtype=np.int64)
    position = np.asarray(position, dtype=np.int64)
    ell = np.asarray(ell, dtype=np.float64)
    agree = np.asarray(agree, dtype=np.float64)

    order = np.argsort(state_id, kind="stable")
    sorted_state = state_id[order]
    bounds = np.flatnonzero(np.diff(sorted_state)) + 1
    slices = np.split(order, bounds)

    D, S, P, E, A, M, R, SRC = [], [], [], [], [], [], [], []
    for idx in slices:
        if idx.size == 0 or idx.size < min_state_size:
            continue
        pd = draw_pools(ell[idx], pool_sizes, n_pools_per_size, seed=seed,
                        state_key=str(int(state_id[idx[0]])), retain=retain, stream=stream)
        if len(pd) == 0:
            continue
        rows = idx[pd.state_row]
        D.append(doc_id[rows]); S.append(state_id[rows]); P.append(position[rows])
        E.append(ell[rows]); A.append(agree[rows]); M.append(pd.m); R.append(pd.r)
        SRC.append(rows)

    if not D:
        raise ValueError("no state had enough masked positions to form a pool")
    return AnalysisFrame(
        doc_id=np.concatenate(D), state_id=np.concatenate(S),
        position=np.concatenate(P), ell=np.concatenate(E),
        agree=np.concatenate(A), m=np.concatenate(M), r=np.concatenate(R),
        src_row=np.concatenate(SRC).astype(np.int64))


SCALAR_COLUMNS = (
    "model", "variant", "precision", "dataset", "mask_kind", "policy", "k", "target_rate",
    "mask_rate", "step", "n_masked", "doc_id", "state_id", "state_idx", "doc_len",
    "position", "top1", "conf", "ell", "p_true", "entropy", "margin", "delta_ell",
    "agree", "committed", "commit_rank",
)
"""Everything but the top-k lists: what the endpoint analyses read."""


def trace_meta(path: str | Path) -> dict:
    """The ``.meta.json`` written next to a trace (a dataset manifest is its own
    meta), or {} if there is none."""
    import json

    from .traces.shards import is_dataset

    if is_dataset(path):
        return json.loads(Path(path).read_text())
    p = Path(path).with_suffix(".meta.json")
    return json.loads(p.read_text()) if p.exists() else {}


def read_trace_table(path: str | Path, columns=None, filters=None):
    """A pyarrow Table from a parquet trace or a ``*.dataset.json`` manifest of parts.

    ``columns`` names a file lacks are skipped; ``filters`` are pyarrow DNF filters
    (e.g. ``[("mask_kind", "in", ["decoder"])]``), applied while reading.
    """
    import pyarrow.parquet as pq

    from .traces.shards import dataset_files, is_dataset

    sources = [str(f) for f in dataset_files(path)] if is_dataset(path) else str(path)
    first = sources[0] if isinstance(sources, list) else sources
    if columns is not None:
        names = set(pq.read_schema(first).names)
        columns = [c for c in columns if c in names]
    return pq.read_table(sources, columns=columns, filters=filters)


def upgrade_v1(df, meta: dict | None = None):
    """Give a schema-v1 trace the v2 design columns, where v1 implies them.

    v1 decoder states came from one teacher-forced confidence top-k trajectory
    whose k the meta records as ``tokens_per_step``.  Measurements v1 never took
    (p_true, entropy, committed, ...) are left absent rather than invented;
    ``endpoints.e1.native_gap`` rebuilds the v1 commit sets itself.
    """
    meta = meta or {}
    if "policy" not in df.columns and "mask_kind" in df.columns:
        dec = (df["mask_kind"] == "decoder").to_numpy()
        k = int(meta.get("config", {}).get("tokens_per_step", 8))
        df["policy"] = np.where(dec, "topk_conf", "na")
        df["k"] = np.where(dec, k, 0).astype(np.int16)
    return df


def load_traces(path: str | Path, mask_kind: str | None = None,
                model: str | None = None, precision: str | None = None,
                columns=None):
    """Read a parquet trace (or a ``*.dataset.json`` manifest of its parts),
    optionally filtered to one arm of the design.

    ``columns`` limits what is read (names a file lacks are skipped), which
    matters for v2 files whose top-k lists dominate their size.  v1 files are
    upgraded with :func:`upgrade_v1`.
    """
    df = read_trace_table(path, columns=columns).to_pandas()
    df = upgrade_v1(df, trace_meta(path))
    if mask_kind is not None:
        df = df[df["mask_kind"] == mask_kind]
    if model is not None:
        df = df[df["model"] == model]
    if precision is not None:
        df = df[df["precision"] == precision]
    return df.reset_index(drop=True)


def arm_labels(df) -> np.ndarray:
    """The design arm of every row: the mask kind, split by (policy, k) where needed.

    A v2 trace records decoder trajectories for several k (and later several
    decoders) under one ``mask_kind``; they are different arms and must not be
    pooled.  A kind with a single (policy, k) keeps its bare name, so v1 traces
    and single-arm runs are labelled exactly as before.
    """
    kind = df["mask_kind"].astype(str).to_numpy()
    if "policy" not in df.columns or "k" not in df.columns:
        return kind.astype(object)
    policy = df["policy"].astype(str).to_numpy()
    k = df["k"].to_numpy(np.int64)
    out = kind.astype(object)
    for kd in np.unique(kind):
        sel = kind == kd
        combos = set(zip(policy[sel], k[sel]))
        if len(combos) > 1:
            out[sel] = [f"{kd}:{p}:k{kk}" for p, kk in zip(policy[sel], k[sel])]
    return out


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


def position_and_pool_frames(df, **pool_kw) -> tuple[PositionFrame, AnalysisFrame]:
    """The position table and the pools drawn from it, linked by ``src_row``.

    ``df`` must be one arm of the design (one model, precision and mask kind);
    ``state_id`` is only unique within an arm.
    """
    if len(df) == 0:
        raise ValueError("trace selection is empty")
    df = df.reset_index(drop=True)
    pos = PositionFrame.from_frame(df)
    pools = frame_from_arrays(pos.doc_id, pos.state_id, pos.position, pos.ell, pos.agree,
                              **pool_kw)
    return pos, pools
