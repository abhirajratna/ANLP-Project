"""Position-level traces with known truth, in the trace schema.

This generator exists to test the v2 estimators end to end: it emits exactly the
columns ``c3.traces.collect`` writes, so the real-data pipeline
(``scripts/analyze_traces.py``) runs on it unchanged.  It is also the T0
simulation of record: ``scripts/t0_scenarios.py`` builds the plan's four-scenario
table and Figure 1 from it, and ``c3.sim.truth`` gives its exact pi(l, s) and
pi(l).  (The plan cites a ``sim_field.py``, which is not in this repository.)

Each masked position has a true log-odds ell* and correctness a ~ Bernoulli(
sigma(ell*)); the model reports ell = ell* + noise.  The scenarios switch on the
terms of Prop. 2 one at a time:

``calibrated``   no noise, identical states.  pi(l, s) = sigma(l): every term is 0.
``uniform``      noise of one scale in identical states.  pi(l, s) = pi(l) != sigma(l):
                 Delta(m) grows with m through ordinary miscalibration; field term 0.
``context``      state difficulty d_s lowers the field mean and raises the noise
                 scale, so pi(l, s) depends on s and a lone high score in a weak,
                 noisy field is more likely noise: the field term is positive and
                 grows with m (plan v2 Fig. 1).
``misreading``   a hidden per-state event shifts the scores of a subset of
                 positions together, leaving correctness alone.  Positions are no
                 longer independent given the state, which breaks Prop. 1.
"""

from __future__ import annotations

import numpy as np

from ..rng import generator

SCENARIOS = ("calibrated", "uniform", "context", "misreading")

_RATES = (0.2, 0.4, 0.6, 0.8)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_positions(scenario: str = "context", n_docs: int = 200, states_per_doc: int = 4,
                       n_positions: int = 128, seed: int = 0, mu: float = 0.5,
                       sd: float = 1.5, sigma: float = 1.0, difficulty_shift: float = 1.0,
                       difficulty_noise: float = 0.6, misread_prob: float = 0.3,
                       misread_frac: float = 0.25, misread_shift: float = 2.0):
    """Return a pandas DataFrame in the trace schema, plus two columns of truth.

    ``p_true`` is sigma(ell*), the position's true probability of being right, and
    ``difficulty`` is the state's latent d_s (it moves the field only in
    'context'), from which ``c3.sim.truth`` computes the exact pi(l, s).

    Random numbers are keyed by (seed, doc, state) and not by scenario, so the
    scenarios share their draws (common random numbers) and differ only by the
    mechanism under study.
    """
    import pandas as pd

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; known: {SCENARIOS}")

    cols = {k: [] for k in ("doc_id", "state_id", "position", "ell", "conf", "agree",
                            "mask_kind", "mask_rate", "n_masked", "step", "doc_len", "p_true",
                            "difficulty")}
    doc_len = int(round(n_positions / min(_RATES)))
    for d in range(n_docs):
        for s in range(states_per_doc):
            rng = generator(seed, "sim-positions", str(d), str(s))
            d_s = rng.normal()
            z = rng.normal(size=n_positions)
            eps = rng.normal(size=n_positions)
            u = rng.random(n_positions)
            misread = rng.random() < misread_prob
            subset = rng.random(n_positions) < misread_frac

            mean_s, sig_s = mu, sigma
            if scenario == "context":
                mean_s = mu - difficulty_shift * d_s
                sig_s = sigma * np.exp(difficulty_noise * d_s)
            if scenario == "calibrated":
                sig_s = 0.0

            ell_star = mean_s + sd * z
            ell = ell_star + sig_s * eps
            if scenario == "misreading" and misread:
                ell = ell + misread_shift * subset
            p_true = _sigmoid(ell_star)
            agree = (u < p_true).astype(np.int8)

            rate = _RATES[s % len(_RATES)]
            cols["doc_id"].append(np.full(n_positions, d, np.int64))
            cols["state_id"].append(np.full(n_positions, d * 1000 + s, np.int64))
            cols["position"].append(np.arange(n_positions, dtype=np.int64))
            cols["ell"].append(ell)
            cols["conf"].append(_sigmoid(ell))
            cols["agree"].append(agree)
            cols["mask_kind"].append(np.full(n_positions, "uniform", object))
            cols["mask_rate"].append(np.full(n_positions, rate))
            cols["n_masked"].append(np.full(n_positions, n_positions, np.int64))
            cols["step"].append(np.zeros(n_positions, np.int64))
            cols["doc_len"].append(np.full(n_positions, doc_len, np.int64))
            cols["p_true"].append(p_true)
            cols["difficulty"].append(np.full(n_positions, d_s))

    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df["model"] = f"sim:{scenario}"
    df["precision"] = "na"
    return df
