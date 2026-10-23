"""Commit policies: which masked positions a decoder fills at each step (plan v2 T12).

Every policy sees the per-position statistics of one forward pass over the
positions it may commit (``c3.decode.scoring.position_stats``, rows in position
order) and returns ``(order, n_commit)``: indices into those rows, best first,
and how many of the first are committed now.  Returning the full order, not just
the chosen set, is what lets the trace record each position's commit rank.

Tokens-per-step convention.  Fixed-k policies commit exactly k per step, except
where fewer than k positions remain (the end of a block).  Threshold-type
policies (Fast-dLLM, EB-Sampler, KLASS) commit a data-dependent number, always
at least one.  T12 compares decoders at *matched average tokens per step* --
tokens committed / forward passes over the whole generation, as reported by
``DecodeResult.tokens_per_step`` -- sweeping each policy's own speed knob (k,
tau, gamma, ...) to hit the same averages.

Definitions follow the papers (checked against the arXiv versions, Sep 2026):

* ``TopKConfidence``  confidence top-k (LLaDA's low-confidence remasking,
  Nie et al. 2025, arXiv:2502.09992).  Ranked by the exact log-odds ell, which
  orders exactly as confidence but does not tie when float32 confidence
  saturates; remaining ties go to the lower position.
* ``Margin``  top-k by p_top1 - p_top2 (J. Kim et al. 2025, arXiv:2502.06768).
* ``Threshold``  Fast-dLLM parallel decoding (Wu et al. 2025, arXiv:2505.22618,
  Alg. 1): commit every position with c >= tau, and always the most confident one;
  tau = 0.9 in the paper.
* ``EntropyBounded``  EB-Sampler (Ben-Hamu et al. 2025, arXiv:2505.24857, Eq. 2):
  sort by entropy (ascending) and commit the largest prefix U with
  sum_{U} H - max_{U} H <= gamma; at least one.
* ``KLASS``  (S. H. Kim et al. 2025, arXiv:2511.05664, Eqs. 6-8): a position is
  stable if its full-vocabulary KL between consecutive steps' predictions stayed
  below eps_kl for the last n steps (n = 2 for text) and its confidence exceeds
  tau; all stable positions are committed, else the top-u by confidence.  u is
  not given in the paper; 1 here.  Per-task (tau, eps_kl) are in its Table 7.
* ``PCSampler``  (Huang et al. 2025, arXiv:2508.13021, Eq. 5): score =
  exp(-lambda * i) * min(-log p_D(x_hat), alpha) * confidence, with i the index in
  the generated region, p_D a background token-frequency distribution (a C4
  subset in the paper), lambda = 0.25 and alpha = 10; top-k by score.  Without
  ``log_freq`` the frequency factor is 1 and only the positional weight remains,
  which is recorded as ``variant = "position-only"``.

C3a (plan v2 T11, this project): commit every eligible position with
c_hat >= 1 - alpha, and at least one, where c_hat = pi_hat(ell, s) is a context-aware
calibrator (``c3.calibrators.ContextLogistic``) fitted on fit-half traces.  The
state s is described by the decode-time field features (``DECODE_FEATURES``),
computed over every masked position of the generated region as the traces do,
not only over the active block (``needs_state``; ``c3.decode.loop.decode``).  All
positions of a state share s, so within a step C3a is a confidence threshold
whose cut moves with the state: it changes how many tokens are committed, not
which.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepContext:
    """What a policy may know besides the current step's statistics."""

    step: int
    positions: np.ndarray                 # the rows' sequence positions, ascending
    gen_start: int = 0                    # first position of the generated region
    kl_history: dict = field(default_factory=dict)   # position -> recent KLs, newest last
    state: dict | None = None             # decode-time field features (policies with needs_state)


def _stable_order(key: np.ndarray) -> np.ndarray:
    """Indices sorting ``key`` descending; ties keep position order."""
    return np.argsort(-np.asarray(key, dtype=np.float64), kind="stable")


@dataclass
class TopKConfidence:
    k: int
    name: str = "topk_conf"

    def select(self, stats, ctx):
        order = _stable_order(stats["ell"])
        return order, min(self.k, order.size)


@dataclass
class Margin:
    k: int
    name: str = "margin"

    def select(self, stats, ctx):
        order = _stable_order(stats["margin"])
        return order, min(self.k, order.size)


@dataclass
class Threshold:
    tau: float = 0.9
    name: str = "threshold"

    def select(self, stats, ctx):
        order = _stable_order(stats["ell"])
        n = int(np.sum(np.asarray(stats["conf"]) >= self.tau))
        return order, max(1, n)


@dataclass
class EntropyBounded:
    gamma: float
    name: str = "entropy_bounded"

    def select(self, stats, ctx):
        h = np.asarray(stats["entropy"], dtype=np.float64)
        order = np.argsort(h, kind="stable")
        # For the sorted prefix of length j the max is its last element, so the
        # bound is on the sum of the first j-1 entropies; it only grows with j.
        before = np.concatenate([[0.0], np.cumsum(h[order])[:-1]])
        return order, max(1, int(np.sum(before <= self.gamma + 1e-12)))


@dataclass
class KLASS:
    tau: float = 0.9
    eps_kl: float = 0.001
    history: int = 2
    fallback: int = 1
    name: str = "klass"
    needs_kl: bool = True

    def select(self, stats, ctx):
        conf = np.asarray(stats["conf"], dtype=np.float64)
        stable = np.zeros(conf.size, dtype=bool)
        for j, pos in enumerate(ctx.positions):
            kl = ctx.kl_history.get(int(pos), [])
            stable[j] = (len(kl) >= self.history and all(v < self.eps_kl for v in kl[-self.history:])
                         and conf[j] > self.tau)
        by_conf = _stable_order(stats["ell"])
        if stable.any():
            order = np.concatenate([by_conf[stable[by_conf]], by_conf[~stable[by_conf]]])
            return order, int(stable.sum())
        return by_conf, min(self.fallback, by_conf.size)


@dataclass
class PCSampler:
    k: int
    lam: float = 0.25
    alpha: float = 10.0
    log_freq: np.ndarray | None = None    # log p_D(token), background unigram frequencies
    name: str = "pc_sampler"

    @property
    def variant(self) -> str:
        return "full" if self.log_freq is not None else "position-only"

    def weights(self, stats, ctx) -> np.ndarray:
        i = np.asarray(ctx.positions) - ctx.gen_start
        w = np.exp(-self.lam * i)
        if self.log_freq is not None:
            w = w * np.minimum(-np.asarray(self.log_freq)[np.asarray(stats["top1"])], self.alpha)
        return w

    def select(self, stats, ctx):
        score = self.weights(stats, ctx) * np.asarray(stats["conf"], dtype=np.float64)
        order = _stable_order(score)
        return order, min(self.k, order.size)


@dataclass
class C3a:
    alpha: float
    calibrator: object = field(default=None, repr=False)
    name: str = "c3a"
    needs_state: bool = True

    def calibrated(self, stats, ctx) -> np.ndarray:
        """c_hat for every row, from its ell and the state's field features."""
        if self.calibrator is None:
            raise ValueError("C3a needs a fitted calibrator")
        if ctx.state is None:
            raise ValueError("C3a needs the state's field features (StepContext.state)")
        return self.calibrator.predict_arrays(np.asarray(stats["ell"], dtype=np.float64),
                                              ctx.state)

    def select(self, stats, ctx):
        c = self.calibrated(stats, ctx)
        order = _stable_order(c)
        return order, max(1, int(np.sum(c >= 1.0 - self.alpha)))


POLICIES = {"topk_conf": TopKConfidence, "margin": Margin, "threshold": Threshold,
            "entropy_bounded": EntropyBounded, "klass": KLASS, "pc_sampler": PCSampler,
            "c3a": C3a}
CALIBRATED = ("c3a",)


def unigram_log_freq(token_arrays, vocab_size: int, smoothing: float = 1.0) -> np.ndarray:
    """log p_D over a vocabulary from token-id arrays, add-``smoothing``.

    PC-Sampler's p_D comes from a background corpus, which must not be the
    evaluation documents; build it from text the audit never scores.
    """
    counts = np.full(vocab_size, float(smoothing))
    for ids in token_arrays:
        counts += np.bincount(np.asarray(ids, dtype=np.int64), minlength=vocab_size)[:vocab_size]
    return np.log(counts / counts.sum())
