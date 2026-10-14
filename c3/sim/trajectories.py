"""Synthetic traces with all three mask families, for E3 (T7) and the T8 controls.

``c3.sim.positions`` emits uniform states only, which is enough for E1/E2 but
not for comparing decoder-visited states with random ones.  This generator runs
the same design as ``c3.traces.collect`` on a toy model whose calibration is
known exactly: uniform masks on the plan's rate grid, a teacher-forced
confidence top-k trajectory recording every state, and a uniform mask matched
in size to each decoder state.  Columns follow the trace schema, plus
``truth_p`` (the true probability that the top-1 token is right) and
``truth_dist`` (distance to the nearest unmasked token).

Each position has a latent ease ``ell*``; in a state, visible neighbours raise
it (``context_gain``), so a decoder that commits easy positions first leaves
hard ones behind -- the selection-induced covariate shift E3 is about.  The
model's reported log-odds depart from the truth by one mechanism per scenario:

``calibrated``   reported = true log-odds.  pi(l) = sigma(l) in every state type.
``decoder_shift``  decoder states report ``shift`` more than the truth, so at
                 equal confidence they are right less often (E3 type effect > 0).
``temperature``  reported = true x (1 + ``temp_slope`` x mask rate): overconfident,
                 more so at high mask rates; a per-rate-bin temperature undoes it.
``position``     positions at least 3 away from any visible token report
                 ``shift`` more than the truth: positional miscalibration (T8).
``noisy``        reported = true + N(0, ``noise``): regression to the mean, so
                 the calibration curve depends on the state's prior (shift).

``err_corr`` (any scenario) couples the *errors* of position pairs (2t, 2t+1)
through a Gaussian copula with latent correlation ``err_corr`` (negative:
anticorrelated), leaving every position's marginal probability of being right
unchanged.  Per-token calibration is then exactly as before, but the chance that a
committed set is right jointly departs from the product of its members -- the case
set-level calibration (``c3.setcal``) is for; :func:`set_truth` gives it exactly.
"""

from __future__ import annotations

import numpy as np

from ..rng import generator

SCENARIOS = ("calibrated", "decoder_shift", "temperature", "position", "noisy")
STATE_STRIDE = 100_000          # as c3.traces.collect


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def distance_to_visible(mask: np.ndarray) -> np.ndarray:
    """For a boolean mask, the distance from every masked slot to the nearest unmasked one.

    A slot with no visible token on either side gets ``len(mask)``.
    """
    L = mask.size
    left = np.full(L, np.inf)
    right = np.full(L, np.inf)
    last = -np.inf
    for i in range(L):
        if not mask[i]:
            last = i
        left[i] = i - last
    last = np.inf
    for i in range(L - 1, -1, -1):
        if not mask[i]:
            last = i
        right[i] = last - i
    d = np.minimum(left, right)
    d[~np.isfinite(d)] = L
    return d[mask].astype(np.int64)


def simulate_design(scenario: str = "calibrated", n_docs: int = 40, doc_len: int = 64,
                    ks=(4,), rates=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    masks_per_rate: int = 2, matched: bool = True, seed: int = 0,
                    mu: float = 0.5, sd: float = 2.0, context_gain: float = 2.0,
                    shift: float = 1.0, temp_slope: float = 0.8, noise: float = 1.0,
                    vocab: int = 40, n_trivial: int = 10, err_corr: float = 0.0):
    """Return a trace-schema DataFrame for ``n_docs`` documents of ``doc_len`` tokens.

    Token ids below ``n_trivial`` stand for punctuation / stop-words (T8 token
    type); they are made easier (ell* + 1), as such tokens are in real text.
    """
    import pandas as pd

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; known: {SCENARIOS}")
    cols = {k: [] for k in (
        "doc_id", "state_id", "state_idx", "position", "ell", "conf", "agree", "top1",
        "mask_kind", "policy", "k", "target_rate", "mask_rate", "step", "n_masked",
        "doc_len", "span_start", "committed", "commit_rank", "delta_ell", "truth_p",
        "truth_dist")}

    for d in range(n_docs):
        rng = generator(seed, "sim-design", str(d))
        x0 = rng.integers(0, vocab, doc_len)
        ell_star = mu + sd * rng.normal(size=doc_len) + (x0 < n_trivial) * 1.0
        states = []          # (mask, kind, policy, k, target_rate, step)

        for t in rates:
            for _ in range(masks_per_rate):
                m = rng.random(doc_len) < t
                if not m.any():
                    m[rng.integers(doc_len)] = True
                states.append((m, "uniform", "na", 0, float(t), 0))

        def score(mask, kind, key):
            """Reported ell, true probability and agreement at the masked slots."""
            pos = np.flatnonzero(mask)
            vis = ~mask
            # visible neighbours within +-2, as a fraction of the 4 slots
            nb = np.zeros(doc_len)
            for off in (-2, -1, 1, 2):
                src = np.arange(doc_len) + off
                ok = (src >= 0) & (src < doc_len)
                nb[ok] += vis[src[ok]]
            true = ell_star[pos] + context_gain * (nb[pos] / 4.0 - 0.5)
            r = generator(seed, "sim-design-state", str(d), key)
            if err_corr:
                agree = _copula_agree(pos, _sigmoid(true), err_corr,
                                      generator(seed, "sim-design-copula", str(d), key))
            else:
                agree = (r.random(pos.size) < _sigmoid(true)).astype(np.int64)
            rate = pos.size / doc_len
            rep = true.copy()
            if scenario == "decoder_shift" and kind == "decoder":
                rep = rep + shift
            elif scenario == "temperature":
                rep = rep * (1.0 + temp_slope * rate)
            elif scenario == "position":
                rep = rep + shift * (distance_to_visible(mask) >= 3)
            elif scenario == "noisy":
                rep = rep + noise * r.normal(size=pos.size)
            return pos, rep, _sigmoid(true), agree

        scored = [score(s[0], s[1], f"u{i}") for i, s in enumerate(states)]
        extra = [None] * len(states)

        dec_states = []
        for k in ks:
            mask = np.ones(doc_len, dtype=bool)
            step = 0
            prev = np.full(doc_len, np.nan)
            while mask.any():
                pos, rep, tp, ag = score(mask, "decoder", f"d{k}-{step}")
                order = np.argsort(-rep, kind="stable")
                n_commit = min(k, pos.size)
                rank = np.empty(pos.size, dtype=np.int64)
                rank[order] = np.arange(1, pos.size + 1)
                states.append((mask.copy(), "decoder", "topk_conf", int(k), float("nan"), step))
                scored.append((pos, rep, tp, ag))
                extra.append((rank <= n_commit, rank, rep - prev[pos]))
                dec_states.append((mask.copy(), int(k), step))
                prev[pos] = rep
                mask[pos[order[:n_commit]]] = False
                step += 1

        if matched:
            for i, (dm, k, step) in enumerate(dec_states):
                m = np.zeros(doc_len, dtype=bool)
                m[rng.choice(doc_len, size=int(dm.sum()), replace=False)] = True
                states.append((m, "uniform_matched", "topk_conf", k, float("nan"), step))
                scored.append(score(m, "uniform_matched", f"m{i}"))
                extra.append(None)

        for si, ((mask, kind, policy, k, t, step), (pos, rep, tp, ag), ex) in enumerate(
                zip(states, scored, extra)):
            n = pos.size
            cols["doc_id"].append(np.full(n, d, np.int64))
            cols["state_id"].append(np.full(n, d * STATE_STRIDE + si, np.int64))
            cols["state_idx"].append(np.full(n, si, np.int64))
            cols["position"].append(pos.astype(np.int64))
            cols["ell"].append(rep)
            cols["conf"].append(_sigmoid(rep))
            cols["agree"].append(ag)
            cols["top1"].append(np.where(ag == 1, x0[pos], (x0[pos] + 1) % vocab))
            cols["mask_kind"].append(np.full(n, kind, object))
            cols["policy"].append(np.full(n, policy, object))
            cols["k"].append(np.full(n, k, np.int16))
            cols["target_rate"].append(np.full(n, t, np.float32))
            cols["mask_rate"].append(np.full(n, n / doc_len))
            cols["step"].append(np.full(n, step, np.int64))
            cols["n_masked"].append(np.full(n, n, np.int64))
            cols["doc_len"].append(np.full(n, doc_len, np.int64))
            cols["span_start"].append(np.zeros(n, np.int64))
            cols["committed"].append(np.zeros(n, bool) if ex is None else ex[0])
            cols["commit_rank"].append(np.full(n, -1, np.int32) if ex is None
                                       else ex[1].astype(np.int32))
            cols["delta_ell"].append(np.full(n, np.nan, np.float32) if ex is None
                                     else ex[2].astype(np.float32))
            cols["truth_p"].append(tp)
            cols["truth_dist"].append(distance_to_visible(mask))

    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df["model"] = f"sim-design:{scenario}"
    df["precision"] = "na"
    df["dataset"] = "sim"
    return df


def _copula_agree(pos: np.ndarray, p: np.ndarray, r: float, rng) -> np.ndarray:
    """Correct/incorrect draws with marginals ``p`` and latent correlation ``r`` within
    each pair block (2t, 2t+1); blocks are independent."""
    from scipy.special import ndtr

    block = pos // 2
    _, b_idx = np.unique(block, return_inverse=True)
    w = rng.normal(size=b_idx.max() + 1)[b_idx]
    sign = np.where((pos % 2 == 1) & (r < 0), -1.0, 1.0)
    z = np.sqrt(abs(r)) * sign * w + np.sqrt(1.0 - abs(r)) * rng.normal(size=pos.size)
    return (ndtr(z) < p).astype(np.int64)


def set_truth(positions: np.ndarray, truth_p: np.ndarray, err_corr: float = 0.0) -> float:
    """Exact probability that every position of a set is right, under ``err_corr``.

    A pair block whose two members are both in the set contributes their bivariate
    normal orthant probability; every other member contributes its own ``truth_p``.
    """
    from scipy.special import ndtri
    from scipy.stats import multivariate_normal

    positions = np.asarray(positions, dtype=np.int64)
    p = np.asarray(truth_p, dtype=np.float64)
    if err_corr == 0 or positions.size < 2:
        return float(np.prod(p))
    total = 1.0
    blocks = positions // 2
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        if idx.size == 1:
            total *= p[idx[0]]
            continue
        cov = [[1.0, err_corr], [err_corr, 1.0]]
        total *= float(multivariate_normal(mean=[0, 0], cov=cov).cdf(ndtri(p[idx])))
    return total


def true_token_ids(n_docs: int, doc_len: int = 64, seed: int = 0, vocab: int = 40) -> dict:
    """The x0 of every simulated document (what the corpus is for real traces)."""
    return {d: generator(seed, "sim-design", str(d)).integers(0, vocab, doc_len)
            for d in range(n_docs)}
