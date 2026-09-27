"""E1: the committed-token gap Delta(m) (plan v2 T5).

    Delta(m) = E_m[ c_i - a_i ]   over tokens that won a pool of size m

Positive means the committed token is overconfident.  (The proposal-era
``c3.calibration.commitment_gap`` used the opposite sign, a - c; it is kept
unchanged only so the committed Stage 0 artifacts still reproduce.)

Averaging is document-first: within a document over its winners, then equally
across documents.  All pool sizes share one set of document resamples, so the
headline contrast Delta(32) - Delta(1) has a valid interval.
"""

from __future__ import annotations

import numpy as np

from .. import POOL_SIZES
from ..frame import AnalysisFrame, PositionFrame
from ..pools import long_weights
from ..stats import GroupedMean, grouped_doc_mean


def doc_codes_of(pos: PositionFrame) -> np.ndarray:
    """The sorted document ids that index the columns of a run's resample matrix."""
    return np.unique(pos.doc_id)


def _doc_idx(doc_ids: np.ndarray, codes: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(codes, doc_ids)
    if np.any(idx >= codes.size) or np.any(codes[np.minimum(idx, codes.size - 1)] != doc_ids):
        raise KeyError("a document is missing from the resample index")
    return idx


def winner_values(pools: AnalysisFrame, per_position: np.ndarray,
                  pos_mask: np.ndarray | None = None) -> tuple[np.ndarray, AnalysisFrame]:
    """Per-winner values of a per-position quantity, via ``src_row``.

    ``pos_mask`` (per position) keeps only winners drawn from selected positions,
    e.g. the report half.
    """
    if pools.src_row is None:
        raise ValueError("pool frame has no src_row; build it with assemble.frame_from_arrays")
    w = pools.winners()
    if pos_mask is not None:
        w = w.select(np.asarray(pos_mask)[w.src_row])
    return np.asarray(per_position)[w.src_row], w


def winner_gap(pools: AnalysisFrame, pos: PositionFrame, q: np.ndarray, target: np.ndarray,
               codes: np.ndarray, counts: np.ndarray, pos_mask: np.ndarray | None = None,
               exact_m1: bool = True, alpha: float = 0.05) -> GroupedMean:
    """Document-first E_m[q - target] over pool winners, for every pool size.

    ``q`` and ``target`` are per-position arrays (e.g. conf and agree for Delta;
    conf and pi_hat(l) for the own-confidence prediction).  With ``exact_m1`` the
    m=1 value is computed over every masked position of every state (each state
    weighted equally within its document, as the pools weight them) rather than
    from 20 random singletons: a pool of one is a uniformly random position, so
    this is its exact expectation.
    """
    diff = np.asarray(q, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    vals, w = winner_values(pools, diff, pos_mask)
    m = w.m.astype(np.int64)
    weights = np.ones(vals.size)
    doc_ids = w.doc_id

    if exact_m1:
        keep = m != 1
        vals, m, weights, doc_ids = vals[keep], m[keep], weights[keep], doc_ids[keep]
        sel = np.ones(len(pos), dtype=bool) if pos_mask is None else np.asarray(pos_mask, bool)
        # Only states that contributed pools contribute to m=1, so every m
        # averages over the same states.
        states_in_pools = np.unique(pools.state_id)
        sel &= np.isin(pos.state_id, states_in_pools)
        s_idx, _ = pos.select(sel).state_index
        n_in_state = np.bincount(s_idx)[s_idx].astype(np.float64)
        vals = np.concatenate([vals, diff[sel]])
        m = np.concatenate([m, np.ones(int(sel.sum()), dtype=np.int64)])
        weights = np.concatenate([weights, 1.0 / n_in_state])
        doc_ids = np.concatenate([doc_ids, pos.doc_id[sel]])

    return grouped_doc_mean(vals, _doc_idx(doc_ids, codes), codes.size, m, counts,
                            alpha=alpha, weights=weights)


def winner_gap_exact(pos: PositionFrame, q: np.ndarray, target: np.ndarray, codes: np.ndarray,
                     counts: np.ndarray, pos_mask: np.ndarray | None = None,
                     pool_sizes=POOL_SIZES, min_state_size: int = 1,
                     alpha: float = 0.05) -> GroupedMean:
    """Like :func:`winner_gap`, with the exact expectation over all pools.

    Each (position, m) row carries its probability of winning a random pool of
    size m at its state (``c3.pools``); within a state these sum to 1, so states
    are weighted equally within a document, as in the Monte Carlo design, and
    documents equally overall.
    """
    diff = np.asarray(q, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    sel = np.ones(len(pos), bool) if pos_mask is None else np.asarray(pos_mask, bool)
    idx = np.flatnonzero(sel)
    row, m, w = long_weights(pos.state_id[idx], pos.ell[idx], pool_sizes, min_state_size)
    rows = idx[row]
    return grouped_doc_mean(diff[rows], _doc_idx(pos.doc_id[rows], codes), codes.size, m,
                            counts, alpha=alpha, weights=w)


def delta_m(pools: AnalysisFrame | None, pos: PositionFrame, codes: np.ndarray, counts: np.ndarray,
            pos_mask: np.ndarray | None = None, exact_m1: bool = True, method: str = "mc",
            pool_sizes=POOL_SIZES, min_state_size: int = 1) -> GroupedMean:
    """Delta(m) = E_m[c - a], with the stored confidence c.

    ``method="mc"`` averages over the drawn pools (the plan's 20 per size);
    ``method="exact"`` uses the closed-form expectation over all pools.
    """
    if method == "exact":
        return winner_gap_exact(pos, pos.conf, pos.agree, codes, counts, pos_mask,
                                pool_sizes, min_state_size)
    return winner_gap(pools, pos, pos.conf, pos.agree, codes, counts, pos_mask, exact_m1)


def gap_slope_log2(g: GroupedMean) -> tuple[float, np.ndarray]:
    """OLS slope of the estimate on log2(m), with joint bootstrap draws."""
    x = np.log2(g.levels.astype(np.float64))
    if x.size < 2:
        return float("nan"), np.full(g.draws.shape[0], np.nan)
    xc = x - x.mean()
    denom = float(xc @ xc)
    point = float(xc @ (g.point - g.point.mean()) / denom)
    draws = ((g.draws - g.draws.mean(axis=1, keepdims=True)) @ xc) / denom
    return point, draws


def native_gap(pos: PositionFrame, codes: np.ndarray, counts: np.ndarray, k: int,
               min_masked: int = 32, pos_mask: np.ndarray | None = None) -> GroupedMean:
    """Delta over the tokens a teacher-forced top-k decoder actually committed.

    The v1 trace harness does not store which positions were committed, but its
    decoder is deterministic: at every recorded state it commits the
    ``min(k, |M| - min_masked + 1)`` most confident positions, ties to the lower
    position (``masks.decoder_trajectory_masks``).  Rows are stored in position
    order with the same confidences, so the commit set can be rebuilt exactly.
    Results are grouped by floor(log2 |M_t|).
    """
    sel = pos.mask_kind == "decoder"
    if pos_mask is not None:
        sel &= np.asarray(pos_mask, bool)
    p = pos.select(sel)
    if len(p) == 0:
        raise ValueError("no decoder-induced states in this frame")
    order = np.lexsort((p.position, p.state_id))
    s_sorted = p.state_id[order]
    bounds = np.flatnonzero(np.diff(s_sorted)) + 1
    committed = np.zeros(len(p), dtype=bool)
    for rows in np.split(order, bounds):
        n = rows.size
        kk = min(k, n - min_masked + 1)
        if kk <= 0:
            continue
        top = np.argsort(-p.conf[rows], kind="stable")[:kk]
        committed[rows[top]] = True
    c = p.select(committed)
    grp = np.floor(np.log2(np.maximum(c.n_masked, 1))).astype(np.int64)
    return grouped_doc_mean(c.conf - c.agree, _doc_idx(c.doc_id, codes), codes.size,
                            2 ** grp, counts)
