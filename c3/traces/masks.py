"""Masked-state construction.

Two mask distributions are audited (plan v2 Sec. 3.3):

*Uniform* masks match the training corruption process -- each position is masked
independently -- so they ask what calibration looks like at states the model was
actually trained on.  The plan's grid is t in {0.1, ..., 0.9} with 2 masks per
rate per document.

*Decoder-induced* masks are the mask patterns a real confidence decoder visits on
its way through a document.  They are not uniform: the decoder commits confident
positions first, so later states are enriched for hard positions.  Comparing the
two measures the selection-induced covariate shift.

Decoder-induced trajectories here are **teacher-forced**: at every step the
decoder chooses which positions to commit by its own confidence, but the
committed slots are filled with the *true* tokens.  Letting the decoder commit
its own tokens would drift the context away from the reference document, and
``agree_i`` is only defined against that reference.  Plan v2 records **every**
state of the trajectory, for k in {2, 4, 8} tokens per step.

*Uniform-matched* masks (T7) pair every decoder state with a uniform mask of the
same size.  Mask rate alone does not separate "decoder states are harder" from
"decoder states have fewer masked positions"; holding |M| fixed state by state
leaves only the *pattern* of the mask to differ.

Every builder takes an optional ``maskable`` region.  For raw text it is the
whole document; for GSM8K/MBPP it is the answer span, so the question is never
masked (T1 step 2) and mask rates are fractions of the span.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UNIFORM_RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
"""Plan v2 Sec. 3.3: the uniform mask-rate grid."""

MASKS_PER_RATE = 2
"""Plan v2 Sec. 3.3: random masks per rate per document."""

DECODER_K = (2, 4, 8)
"""Plan v2 Sec. 3.3: tokens committed per step for the decoder trajectories."""


@dataclass
class MaskedState:
    mask: np.ndarray        # bool, True where masked
    kind: str               # "uniform" | "decoder" | "uniform_matched"
    step: int               # decoding step index (0 for uniform)
    rate: float             # realised fraction of the maskable region masked
    policy: str = "na"      # decoder that produced (or was matched to) the state
    k: int = 0              # its tokens per step; 0 for uniform states
    target_rate: float = float("nan")   # the nominal uniform rate t

    @property
    def n_masked(self) -> int:
        return int(self.mask.sum())


def maskable_mask(length: int, maskable=None) -> np.ndarray:
    """Boolean region that may be masked: everything, a (start, end) span, or a mask."""
    if maskable is None:
        return np.ones(length, dtype=bool)
    if isinstance(maskable, tuple):
        start, end = maskable
        out = np.zeros(length, dtype=bool)
        out[int(start):int(end)] = True
        return out
    out = np.asarray(maskable, dtype=bool)
    if out.shape != (length,):
        raise ValueError(f"maskable has shape {out.shape}, expected ({length},)")
    return out


def uniform_masks(
    length: int,
    rates=(0.2, 0.4, 0.6, 0.8),
    rng: np.random.Generator | None = None,
    min_masked: int = 32,
    per_rate: int = 1,
    maskable=None,
) -> list[MaskedState]:
    """``per_rate`` independent-Bernoulli masks per rate, matching the training process.

    Draws run rate by rate, then repeat by repeat, so ``per_rate=1`` reproduces
    the proposal-era masks exactly.
    """
    rng = rng or np.random.default_rng(0)
    region = maskable_mask(length, maskable)
    idx = np.flatnonzero(region)
    floor = min(min_masked, idx.size)
    states = []
    for t in rates:
        for _ in range(per_rate):
            hit = rng.random(idx.size) < t
            if hit.sum() < floor:
                # A pool of size m needs at least m masked positions to exist.
                hit = np.zeros(idx.size, dtype=bool)
                hit[rng.choice(idx.size, size=floor, replace=False)] = True
            mask = np.zeros(length, dtype=bool)
            mask[idx[hit]] = True
            states.append(MaskedState(mask=mask, kind="uniform", step=0,
                                      rate=float(hit.mean()), target_rate=float(t)))
    return states


def matched_uniform_masks(decoder_states, rng: np.random.Generator,
                          maskable=None) -> list[MaskedState]:
    """For each decoder state, a uniform mask with the same number of masked positions.

    The positions are a uniform draw from the maskable region and never look at
    the decoder's confidences, so the pair differs only in which positions the
    decoder left masked.  ``step``, ``policy`` and ``k`` are copied from the
    decoder state so the two can be paired in analysis.
    """
    out = []
    for st in decoder_states:
        region = maskable_mask(st.mask.size, maskable)
        idx = np.flatnonzero(region)
        mask = np.zeros(st.mask.size, dtype=bool)
        mask[rng.choice(idx, size=st.n_masked, replace=False)] = True
        out.append(MaskedState(mask=mask, kind="uniform_matched", step=st.step,
                               rate=float(st.n_masked / idx.size), policy=st.policy, k=st.k))
    return out


# --------------------------------------------------------------------------- #
# Teacher-forced trajectories
# --------------------------------------------------------------------------- #

def topk_confidence(stats: dict, k: int, min_masked: int = 1) -> tuple[np.ndarray, int]:
    """Confidence top-k: rank masked positions by confidence, commit the first k.

    Ranks by the log-odds ``ell``, which orders exactly as confidence does but is
    computed without saturating: in float32 many confidences round to 1.0 and
    would tie.  Ties that remain go to the lower position (stable sort on rows
    stored in position order).  Returns (order, n_commit): indices into the
    state's masked positions, best first, and how many of them are committed.
    """
    ell = np.asarray(stats["ell"], dtype=np.float64)
    order = np.argsort(-ell, kind="stable")
    return order, max(0, min(int(k), ell.size - min_masked + 1))


POLICIES = {"topk_conf": topk_confidence}
"""Commit rules usable in a teacher-forced trajectory.  The T12 decoders
(threshold, entropy-bounded, margin, KLASS, PC-Sampler) join in c3.decode."""


@dataclass
class TrajectoryStep:
    """One recorded state of a trajectory, with what the decoder did there."""

    state: MaskedState
    stats: dict               # per-masked-position statistics, position order
    rank: np.ndarray          # commit rank of each masked position (1 = chosen first)
    n_commit: int
    delta_ell: np.ndarray     # ell change since the previous step (NaN at step 0)

    @property
    def committed(self) -> np.ndarray:
        return self.rank <= self.n_commit


@dataclass
class TeacherForcedTrajectory:
    """A teacher-forced decode, advanced one forward pass at a time.

    The caller scores :attr:`mask` (the true tokens everywhere else) and hands
    the statistics to :meth:`advance`, which records the state and commits the
    policy's choice.  Keeping the forward pass outside lets several trajectories
    (one per k and policy) share batches, and means the state that is recorded
    is scored by the same forward pass that chose its commits.
    """

    length: int
    k: int
    policy: str = "topk_conf"
    min_masked: int = 1
    maskable: object = None
    mask: np.ndarray = field(init=False)
    steps: list = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise KeyError(f"unknown policy {self.policy!r}; known: {sorted(POLICIES)}")
        self.mask = maskable_mask(self.length, self.maskable).copy()
        self._prev_ell = np.full(self.length, np.nan)
        self._region = int(self.mask.sum())

    @property
    def done(self) -> bool:
        return int(self.mask.sum()) < max(self.min_masked, 1)

    def advance(self, stats: dict) -> TrajectoryStep:
        pos = np.flatnonzero(self.mask)
        ell = np.asarray(stats["ell"], dtype=np.float64)
        if ell.size != pos.size:
            raise ValueError(f"stats cover {ell.size} positions, the state has {pos.size} masked")
        order, n_commit = POLICIES[self.policy](stats, self.k, self.min_masked)
        rank = np.empty(pos.size, dtype=np.int64)
        rank[order] = np.arange(1, pos.size + 1)
        state = MaskedState(mask=self.mask.copy(), kind="decoder", step=len(self.steps),
                            rate=float(pos.size / self._region), policy=self.policy, k=self.k)
        step = TrajectoryStep(state=state, stats=stats, rank=rank, n_commit=int(n_commit),
                              delta_ell=ell - self._prev_ell[pos])
        self._prev_ell[pos] = ell
        self.steps.append(step)
        if n_commit <= 0:
            self.mask[:] = False       # nothing more to commit: end the trajectory
        else:
            self.mask[pos[order[:n_commit]]] = False
        return step


def run_trajectory(score_fn, length: int, k: int, policy: str = "topk_conf",
                   min_masked: int = 1, maskable=None) -> list[TrajectoryStep]:
    """Run one teacher-forced trajectory to completion.

    ``score_fn(mask) -> stats`` returns per-position statistics (at least
    ``ell``) for the masked positions in position order.
    """
    traj = TeacherForcedTrajectory(length, k, policy, min_masked, maskable)
    while not traj.done:
        traj.advance(score_fn(traj.mask.copy()))
    return traj.steps


def subsample_evenly(items: list, n: int | None) -> list:
    """``n`` items spread evenly over the list (all of them when n is None)."""
    if n is None or len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in np.unique(idx)]


def decoder_trajectory_masks(
    confidences_fn,
    length: int,
    tokens_per_step: int = 8,
    max_states: int = 8,
    min_masked: int = 32,
) -> list[MaskedState]:
    """Mask patterns visited by a teacher-forced confidence top-k decoder (v1).

    The proposal-era construction, kept because ``endpoints.e1.native_gap``
    rebuilds the commit sets of v1 traces from it.  New traces use
    :class:`TeacherForcedTrajectory`, which records every state and its commits.

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

    return subsample_evenly(trajectory, max_states)
