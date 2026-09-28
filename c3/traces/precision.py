"""Comparing two traces of the same documents at different precisions (T1 step 5, T8).

Rows are only comparable where both runs scored the *same* state.  Uniform
states share their masks by construction (same seed), but a decoder trajectory
can diverge: one flipped commit at step t changes every later state.  So states
are matched by their mask (the set of masked positions), not by state id, and
for each trajectory the first step where the two runs disagree is reported --
that is itself a measure of how much precision perturbs a decode.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _state_keys(df):
    """(doc_id, mask_kind, policy, k, step, hash of masked positions) per state."""
    policy = df["policy"] if "policy" in df.columns else "na"
    k = df["k"] if "k" in df.columns else 0
    g = df.assign(_policy=policy, _k=k).groupby("state_id", sort=False)
    keys = {}
    for sid, rows in g:
        pos = np.sort(rows["position"].to_numpy(np.int64))
        h = hashlib.blake2b(pos.tobytes(), digest_size=8).hexdigest()
        r = rows.iloc[0]
        keys[sid] = (int(r["doc_id"]), str(r["mask_kind"]), str(r["_policy"]), int(r["_k"]),
                     int(r["step"]), h)
    return keys


def compare_precisions(df_ref, df_alt) -> dict:
    """Per-position differences on the states both traces scored identically."""
    ka, kb = _state_keys(df_ref), _state_keys(df_alt)
    inv_b = {v: s for s, v in kb.items()}
    pairs = {sa: inv_b[v] for sa, v in ka.items() if v in inv_b}

    a = df_ref[df_ref["state_id"].isin(pairs)].copy()
    a["_pair"] = a["state_id"].map(pairs)
    b = df_alt[df_alt["state_id"].isin(set(pairs.values()))]
    m = a.merge(b, left_on=["_pair", "position"], right_on=["state_id", "position"],
                suffixes=("_ref", "_alt"))
    d_ell = np.abs(m["ell_ref"].to_numpy() - m["ell_alt"].to_numpy())
    d_conf = np.abs(m["conf_ref"].to_numpy() - m["conf_alt"].to_numpy())

    # First step at which each decoder trajectory's states stop matching.
    divergence: dict = {}
    for full in ka.values():
        doc, kind, pol, k, step, _h = full
        if kind != "decoder":
            continue
        key = (doc, pol, k)
        divergence.setdefault(key, None)
        if full not in inv_b:
            cur = divergence[key]
            divergence[key] = step if cur is None else min(cur, step)
    div_steps = [v for v in divergence.values() if v is not None]
    return {
        "n_states_ref": len(ka), "n_states_matched": len(pairs),
        "n_positions_compared": int(len(m)),
        "max_abs_delta_ell": float(d_ell.max()) if d_ell.size else float("nan"),
        "p99_abs_delta_ell": float(np.quantile(d_ell, 0.99)) if d_ell.size else float("nan"),
        "median_abs_delta_ell": float(np.median(d_ell)) if d_ell.size else float("nan"),
        "max_abs_delta_conf": float(d_conf.max()) if d_conf.size else float("nan"),
        "top1_agreement": float(np.mean(m["top1_ref"].to_numpy() == m["top1_alt"].to_numpy()))
        if len(m) else float("nan"),
        "trajectories": len(divergence),
        "trajectories_diverged": len(div_steps),
        "first_divergent_step": int(min(div_steps)) if div_steps else None,
    }
