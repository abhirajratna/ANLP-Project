"""The analysis frame: one row per (pool, retained member).

This is the single table every endpoint is computed from, assembled either from
collected model traces or from the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class AnalysisFrame:
    doc_id: np.ndarray      # clustering unit for the bootstrap
    state_id: np.ndarray    # masked state == fixed-effect group
    position: np.ndarray    # token index within the document
    ell: np.ndarray         # reported confidence, log-odds of the top-1 token
    agree: np.ndarray       # 1{v_i == x0_i}
    m: np.ndarray           # pool size
    r: np.ndarray           # rank within pool, 1 = winner

    def __post_init__(self) -> None:
        n = self.doc_id.size
        for name in ("state_id", "position", "ell", "agree", "m", "r"):
            got = getattr(self, name).size
            if got != n:
                raise ValueError(f"column {name!r} has {got} rows, expected {n}")

    def __len__(self) -> int:
        return int(self.doc_id.size)

    def select(self, mask: np.ndarray) -> "AnalysisFrame":
        mask = np.asarray(mask)
        return replace(self, **{
            k: getattr(self, k)[mask]
            for k in ("doc_id", "state_id", "position", "ell", "agree", "m", "r")
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
        """Document-level 50/50 split, fixed by seed (proposal Sec. 6).

        The split is on documents, not rows: a document's states must not
        straddle the halves or the fixed effects leak between them.
        """
        from .rng import generator
        docs = np.unique(self.doc_id)
        rng = generator(seed, "split")
        perm = rng.permutation(docs.size)
        n_fit = int(round(frac * docs.size))
        fit_docs = set(docs[perm[:n_fit]].tolist())
        in_fit = np.fromiter((d in fit_docs for d in self.doc_id), bool, len(self))
        return self.select(in_fit), self.select(~in_fit)
