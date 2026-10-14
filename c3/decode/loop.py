"""The shared decode loop (plan v2 T8, T10, T11, T12; the contamination screen).

One step = one forward pass over the current sequence, per-position statistics
for the masked positions the decoder may commit now, the policy's choice
(``c3.decode.policies``), and the commit.  Three commit modes:

* free-running greedy (default): the committed slot takes the model's argmax --
  real generation, as T12 and the contamination screen need;
* ``sample=True``: a draw from the model's distribution at ``temperature``;
* teacher-forced (``teacher=`` the reference): the slot takes the *true* token.
  The decoder still chooses *which* slots by its own confidence, so this is the
  trajectory ``c3.traces.masks.TeacherForcedTrajectory`` records, and a test
  holds the two to the same commit sets.

Block-wise decoding (``block_length``, 32 is the LLaDA default): the generated
region is split into consecutive blocks and a block is decoded only once every
earlier block is fully committed; within a block the policy chooses among its
masked positions.  Without it the whole region is one block.

Each step records the committed positions, their confidence, ell, entropy and
margin, and |M_t| (masked positions in the whole sequence and in the active
block), which is what T5's native gap and T10's disagreement rates read.  KLASS
needs the full-vocabulary KL between consecutive steps' predictions; the loop
computes it exactly from the logits for the active block's positions, keeping
one previous log-distribution per position on the logits' device.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .policies import StepContext
from .scoring import position_stats


@dataclass
class StepRecord:
    step: int
    block: int
    n_masked: int            # |M_t| over the whole sequence, before this step's commits
    n_eligible: int          # masked positions in the active block
    positions: np.ndarray    # committed, in the policy's order
    tokens: np.ndarray
    conf: np.ndarray
    ell: np.ndarray
    entropy: np.ndarray
    margin: np.ndarray
    p_true: np.ndarray       # NaN without a reference


@dataclass
class DecodeResult:
    tokens: np.ndarray
    steps: list = field(default_factory=list)   # empty unless record=True
    n_forward: int = 0
    n_committed: int = 0

    @property
    def tokens_per_step(self) -> float:
        """Tokens committed per forward pass: the speed axis T12 matches decoders on."""
        return self.n_committed / max(self.n_forward, 1)


def _blocks(region: np.ndarray, block_length: int | None) -> list[np.ndarray]:
    idx = np.flatnonzero(region)
    if block_length is None or idx.size == 0:
        return [idx]
    start = idx[0]
    return [b for b in (idx[(idx - start) // block_length == j]
                        for j in range(int((idx[-1] - start) // block_length) + 1)) if b.size]


def decode(model, x_init, policy, block_length: int | None = None, max_steps: int | None = None,
           teacher=None, sample: bool = False, temperature: float = 1.0, seed: int = 0,
           record: bool = True, topk: int = 20, on_step=None) -> DecodeResult:
    """Decode every masked slot of ``x_init`` (one sequence) with ``policy``.

    ``on_step(step, stats, ctx, chosen)``, if given, sees every step's full
    statistics (all eligible rows) before the commit -- on-policy logging.
    """
    import torch

    from ..traces.models import check_finite_logits

    x = np.array(x_init, dtype=np.int64, copy=True)
    mask_id = int(model.mask_token_id)
    region = x == mask_id
    ref = None if teacher is None else np.asarray(teacher, dtype=np.int64)
    blocks = _blocks(region, block_length)
    gen_start = int(np.flatnonzero(region)[0]) if region.any() else 0
    dev = getattr(model, "input_device", None)
    gen = torch.Generator().manual_seed(int(seed))
    needs_kl = bool(getattr(policy, "needs_kl", False))
    prev_logp: dict[int, object] = {}
    kl_history: dict[int, list] = {}
    res = DecodeResult(tokens=x)
    cap = max_steps if max_steps is not None else int(region.sum()) + 1

    for step in range(cap):
        masked = x == mask_id
        if not masked.any():
            break
        b = next(j for j, blk in enumerate(blocks) if masked[blk].any())
        eligible = np.zeros_like(masked)
        eligible[blocks[b][masked[blocks[b]]]] = True
        positions = np.flatnonzero(eligible)

        t = torch.as_tensor(x[None, :], dtype=torch.long)
        if dev is not None:
            t = t.to(dev)
        with torch.no_grad():
            lg = model.logits(t)
            check_finite_logits(lg, getattr(model, "name", "model"),
                                getattr(model, "precision", "na"), mask_token_id=mask_id)
            stats = position_stats(lg, eligible[None, :], mask_id,
                                   true_ids=None if ref is None else ref[None, :], topk=topk,
                                   chunk=256)
            if needs_kl or sample:
                z = lg[0, torch.as_tensor(positions, device=lg.device)].float()
                if 0 <= mask_id < z.shape[-1]:
                    z[:, mask_id] = float("-inf")
            if needs_kl:
                logp = torch.log_softmax(z, dim=-1)
                for j, pos in enumerate(positions.tolist()):
                    if pos in prev_logp:
                        old = prev_logp[pos]
                        # KL(p_prev || p_now), full vocabulary (KLASS Eq. 6).  The mask
                        # token is -inf in both, and 0 * (-inf - -inf) would make it NaN.
                        p_old = old.exp()
                        terms = torch.where(p_old > 0, p_old * (old - logp[j]),
                                            torch.zeros_like(p_old))
                        kl = float(terms.sum().clamp_min(0.0))
                        kl_history.setdefault(pos, []).append(kl)
                    prev_logp[pos] = logp[j].clone()
        res.n_forward += 1

        ctx = StepContext(step=step, positions=positions, gen_start=gen_start,
                          kl_history=kl_history, n_masked_total=int(masked.sum()),
                          seq_len=int(x.size))
        order, n_commit = policy.select(stats, ctx)
        n_commit = max(1, min(int(n_commit), positions.size))
        chosen = np.asarray(order[:n_commit], dtype=np.int64)
        if on_step is not None:
            on_step(step, stats, ctx, chosen)
        commit = positions[chosen]
        if ref is not None:
            tokens = ref[commit]
        elif sample:
            probs = torch.softmax(z[torch.as_tensor(chosen, device=z.device)] / temperature, dim=-1)
            tokens = torch.multinomial(probs.cpu(), 1, generator=gen)[:, 0].numpy()
        else:
            tokens = np.asarray(stats["top1"])[chosen]
        if np.any(x[commit] != mask_id):
            raise AssertionError("a policy tried to commit an unmasked slot")
        x[commit] = tokens
        res.n_committed += int(commit.size)
        for pos in commit.tolist():
            prev_logp.pop(pos, None)
        if record:
            res.steps.append(StepRecord(
                step=step, block=b, n_masked=int(masked.sum()), n_eligible=int(positions.size),
                positions=commit, tokens=np.asarray(tokens, dtype=np.int64),
                conf=np.asarray(stats["conf"])[chosen], ell=np.asarray(stats["ell"])[chosen],
                entropy=np.asarray(stats["entropy"])[chosen],
                margin=np.asarray(stats["margin"])[chosen],
                p_true=np.asarray(stats["p_true"])[chosen]))
    res.tokens = x
    return res
