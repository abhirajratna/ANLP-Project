"""Masked-state construction.

Two mask distributions are audited (proposal Sec. 5.2):

*Uniform* masks match the training corruption process -- each position is masked
independently -- so they ask what calibration looks like at states the model was
actually trained on.

*Decoder-induced* masks are the mask patterns a real confidence decoder visits on
its way through a document.  They are not uniform: the decoder commits confident
positions first, so later states are enriched for hard positions.  Comparing the
two measures the selection-induced covariate shift.

Decoder-induced trajectories here are **teacher-forced**: at every step the
decoder chooses which positions to commit by its own confidence, but the
committed slots are filled with the *true* tokens.  Letting the decoder commit
its own tokens would drift the context away from the reference document, and
``agree_i`` is only defined against that reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MaskedState:
    mask: np.ndarray        # bool, True where masked
    kind: str               # "uniform" | "decoder"
    step: int               # decoding step index (0 for uniform)
    rate: float             # realised fraction masked

    @property
    def n_masked(self) -> int:
        return int(self.mask.sum())


def uniform_masks(
    length: int,
    rates=(0.2, 0.4, 0.6, 0.8),
    rng: np.random.Generator | None = None,
    min_masked: int = 32,
) -> list[MaskedState]:
    """One independent-Bernoulli mask per rate, matching the training process."""
    rng = rng or np.random.default_rng(0)
    states = []
    for t in rates:
        mask = rng.random(length) < t
        if mask.sum() < min_masked:
            # A pool of size m needs at least m masked positions to exist.
            idx = rng.choice(length, size=min_masked, replace=False)
            mask = np.zeros(length, dtype=bool)
            mask[idx] = True
        states.append(MaskedState(mask=mask, kind="uniform", step=0,
                                  rate=float(mask.mean())))
    return states


def decoder_trajectory_masks(
    confidences_fn,
    length: int,
    tokens_per_step: int = 8,
    max_states: int = 8,
    min_masked: int = 32,
) -> list[MaskedState]:
    """Mask patterns visited by a teacher-forced confidence top-k decoder.

    ``confidences_fn(mask) -> array`` returns the model's confidence at each
    currently masked position, given that every unmasked slot holds the true
    token.  The decoder commits the ``tokens_per_step`` most confident positions
    and repeats.

    The trajectory is run **to completion** and ``max_states`` states are then
    sampled evenly across it, rather than taking the first ``max_states`` steps.
    Truncating at the start would return only near-fully-masked states: with 256
    positions and 8 tokens committed per step, the first four states sit at mask
    rates 1.00, 0.97, 0.94 and 0.91, where the model is predicting almost
    everything from almost nothing.  Measured on LLaDA-8B that arm had an
    agreement rate of 0.038 and confidence log-odds compressed into [-3.4, -2.6],
    which carries no information about how calibration behaves along a real
    decode.  Sampling across the trajectory instead spans mask rates from 1.0 down
    to ``min_masked / length``.

    Only states with at least ``min_masked`` masked positions are visited, since
    smaller ones cannot support the largest pool.
    """
    mask = np.ones(length, dtype=bool)
    trajectory: list[MaskedState] = []
    step = 0
    while mask.sum() >= min_masked:
        trajectory.append(MaskedState(mask=mask.copy(), kind="decoder", step=step,
                                      rate=float(mask.mean())))
        conf = np.asarray(confidences_fn(mask), dtype=np.float64)
        masked_idx = np.flatnonzero(mask)
        if conf.size != masked_idx.size:
            raise ValueError(
                f"confidences_fn returned {conf.size} values for "
                f"{masked_idx.size} masked positions")
        k = min(tokens_per_step, masked_idx.size - min_masked + 1)
        if k <= 0:
            break
        commit = masked_idx[np.argsort(-conf, kind="stable")[:k]]
        mask[commit] = False
        step += 1

    if len(trajectory) > max_states:
        idx = np.linspace(0, len(trajectory) - 1, max_states).round().astype(int)
        trajectory = [trajectory[i] for i in np.unique(idx)]
    return trajectory
