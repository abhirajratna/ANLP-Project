"""The trace harness.

What is written to disk is the *position-level* table -- one row per (masked
state, masked position) carrying ``ell``, ``agree`` and the top-1 token.  Pools
are drawn later, at analysis time, from that table.

Keeping pools out of the stored artefact matters for two reasons.  It shrinks
the release by two orders of magnitude (2,000 docs x ~8 states x 256 positions
is ~4M rows, against hundreds of millions if every pool membership were
materialised), and it lets the pool design be re-drawn -- different sizes,
different counts, different seeds -- without re-running a single forward pass.
That is what makes the proposal's promise to release the traces, so the audit
can be repeated without repeating the compute, actually cheap to keep.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..rng import generator
from .masks import MaskedState, decoder_trajectory_masks, uniform_masks

TRACE_COLUMNS = (
    "model", "precision", "mask_kind", "mask_rate", "step",
    "doc_id", "state_id", "position", "ell", "conf", "agree", "top1",
    "n_masked", "doc_len",
)


@dataclass
class CollectConfig:
    mask_rates: tuple = (0.2, 0.4, 0.6, 0.8)
    decoder_states: int = 4
    tokens_per_step: int = 8
    min_masked: int = 32
    include_uniform: bool = True
    include_decoder: bool = True
    eps: float = 1e-6
    seed: int = 20260917


def _masked_input(x0: np.ndarray, mask: np.ndarray, mask_token_id: int):
    import torch

    ids = x0.copy()
    ids[mask] = mask_token_id
    return torch.as_tensor(ids[None, :], dtype=torch.long)


def score_state(model, x0: np.ndarray, state: MaskedState, eps: float):
    import torch

    ids = _masked_input(x0, state.mask, model.mask_token_id)
    dev = getattr(getattr(model, "model", None), "device", None)
    if dev is not None:
        ids = ids.to(dev)
    out = model.score_masked(ids, state.mask, eps=eps)
    truth = x0[state.mask]
    agree = (out.top1 == truth).astype(np.int8)
    return out, agree


def collect_document(model, doc_id: int, x0: np.ndarray, cfg: CollectConfig) -> list[dict]:
    """Every masked state for one document, as position-level rows."""
    L = int(x0.size)
    rng = generator(cfg.seed, "masks", str(doc_id))
    states: list[MaskedState] = []
    if cfg.include_uniform:
        states += uniform_masks(L, cfg.mask_rates, rng, cfg.min_masked)
    if cfg.include_decoder:
        def confidences_fn(mask):
            out, _ = score_state(model, x0, MaskedState(mask, "decoder", 0, float(mask.mean())), cfg.eps)
            return out.conf
        states += decoder_trajectory_masks(
            confidences_fn, L, tokens_per_step=cfg.tokens_per_step,
            max_states=cfg.decoder_states, min_masked=cfg.min_masked)

    rows: list[dict] = []
    for si, st in enumerate(states):
        out, agree = score_state(model, x0, st, cfg.eps)
        pos = np.flatnonzero(st.mask)
        state_id = doc_id * 1000 + si
        for j in range(pos.size):
            rows.append({
                "model": model.name,
                "precision": getattr(model, "precision", "na"),
                "mask_kind": st.kind,
                "mask_rate": st.rate,
                "step": st.step,
                "doc_id": doc_id,
                "state_id": state_id,
                "position": int(pos[j]),
                "ell": float(out.ell[j]),
                "conf": float(out.conf[j]),
                "agree": int(agree[j]),
                "top1": int(out.top1[j]),
                "n_masked": st.n_masked,
                "doc_len": L,
            })
    return rows


def collect(model, documents, cfg: CollectConfig, out_path: str | Path,
            progress_every: int = 25) -> Path:
    """Run the harness over a corpus and write a parquet trace file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n_rows = n_sat = 0
    try:
        for i, (doc_id, x0) in enumerate(documents):
            rows = collect_document(model, int(doc_id), np.asarray(x0), cfg)
            if not rows:
                continue
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table)
            n_rows += len(rows)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"[collect] {i+1} docs, {n_rows} rows", flush=True)
    finally:
        if writer is not None:
            writer.close()

    meta = {"config": asdict(cfg), "model": model.name,
            "precision": getattr(model, "precision", "na"), "n_rows": n_rows}
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[collect] wrote {n_rows} rows -> {out_path}")
    return out_path
