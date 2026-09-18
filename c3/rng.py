"""Seed discipline.

Every stochastic step in the audit derives its generator from the run seed and a
string label, so that re-running one stage does not perturb any other stage and a
run is reproducible from (seed, label) alone.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(seed: int, *labels: str) -> int:
    """Derive a stable 63-bit seed from a run seed and a sequence of labels."""
    h = hashlib.blake2b(digest_size=8)
    h.update(int(seed).to_bytes(8, "little", signed=False))
    for label in labels:
        h.update(b"\x00")
        h.update(str(label).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") & ((1 << 63) - 1)


def generator(seed: int, *labels: str) -> np.random.Generator:
    """A PCG64 generator for the substream identified by ``labels``."""
    return np.random.default_rng(derive_seed(seed, *labels))
