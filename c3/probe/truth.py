"""Exact p*(x_S | x_obs) by enumeration, and the (A)/(B) split of the decoder's KL.

A decoding state is a vector ``obs`` over the instance's fields: the observed
value index, or -1 where the field is masked.  The decoder commits a set S of
masked fields in one step, sampling each from its own marginal p_theta,i.  Its
error against the truth is

    KL(p*_S || prod_i p_theta,i) = TC(x_S | x_obs) + sum_{i in S} KL(p*_i || p_theta,i)

with p*_S = p*(x_S | x_obs), the masked fields outside S summed out, and
TC = sum_i H(p*_i) - H(p*_S).  The identity is exact (expand the log of the
product), so the two sides computed separately must agree to rounding; that is
the unit test the plan asks for.

Groups are independent, so p*_S is a product of per-group tables, each over at
most two fields, and TC is a sum of per-group terms.  Only the direct KL needs
the full product table over S, so it is computed only where that table is small
(``max_cells``) -- it exists to check the decomposition, not to replace it.
"""

from __future__ import annotations

import numpy as np

from .instances import Group, Schema

MASKED = -1


def group_joint(schema: Schema, group: Group, rho: float) -> np.ndarray:
    """The true joint over a group's fields (axes in ``group.members`` order)."""
    if len(group.members) == 1:
        return schema.fields[group.members[0]].p.astype(np.float64).copy()
    a, b = group.members
    pa, pb = schema.fields[a].p.astype(np.float64), schema.fields[b].p.astype(np.float64)
    joint = (1.0 - rho) * pa[:, None] * pb[None, :]
    joint[np.arange(pa.size), group.fmap] += rho * pa
    return joint


def group_table(schema: Schema, group: Group, rho: float, obs, keep) -> tuple[np.ndarray, list[int]]:
    """p*(group fields in ``keep`` | the group's observed fields), other masked fields summed out.

    Returns the normalised table and the field index of each of its axes.
    """
    table = group_joint(schema, group, rho)
    axes = list(group.members)
    for ax in range(len(axes) - 1, -1, -1):          # condition on observed fields
        v = int(obs[axes[ax]])
        if v != MASKED:
            table = np.take(table, v, axis=ax)
            del axes[ax]
    for ax in range(len(axes) - 1, -1, -1):          # sum out masked fields not committed
        if axes[ax] not in keep:
            table = table.sum(axis=ax)
            del axes[ax]
    z = table.sum()
    if not z > 0:
        raise ValueError(f"observed values have zero probability in group {group.members}")
    return table / z, axes


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def kl(p: np.ndarray, logq: np.ndarray) -> float:
    """KL(p || q) in nats with q given as log-probabilities over p's support."""
    p = np.asarray(p, dtype=np.float64)
    logq = np.asarray(logq, dtype=np.float64)
    nz = p > 0
    return float((p[nz] * (np.log(p[nz]) - logq[nz])).sum())


def marginals(schema: Schema, rho: float, obs, fields) -> dict[int, np.ndarray]:
    """p*_i = p*(x_i | x_obs) for each field i in ``fields`` (all masked)."""
    groups = schema.group_of()
    out = {}
    for i in fields:
        t, _ = group_table(schema, schema.groups[groups[i]], rho, obs, {i})
        out[int(i)] = t
    return out


def decompose(schema: Schema, rho: float, obs, S, logq: dict) -> dict:
    """(A), (B) and their per-field parts for committing the masked fields S.

    ``logq[i]`` is log p_theta,i over field i's values (the model's log-probability
    of each value's token; it need not sum to one, the rest of the vocabulary
    takes the remainder).
    """
    S = [int(i) for i in S]
    if any(int(obs[i]) != MASKED for i in S):
        raise ValueError("every committed field must be masked")
    groups = schema.group_of()
    A = 0.0
    kl_i, h_i = {}, {}
    for g in sorted({int(groups[i]) for i in S}):
        table, axes = group_table(schema, schema.groups[g], rho, obs, set(S))
        h_joint = entropy(table)
        for ax, i in enumerate(axes):
            pi = table.sum(axis=tuple(j for j in range(table.ndim) if j != ax))
            h_i[i] = entropy(pi)
            kl_i[i] = kl(pi, logq[i])
        A += sum(h_i[i] for i in axes) - h_joint
    return {"A": A, "B": float(sum(kl_i.values())), "kl": kl_i, "h": h_i}


def joint_over(schema: Schema, rho: float, obs, S) -> np.ndarray:
    """The full table p*(x_S | x_obs), one axis per field of S in the given order."""
    S = [int(i) for i in S]
    groups = schema.group_of()
    full = np.ones([schema.fields[i].n for i in S], dtype=np.float64)
    for g in sorted({int(groups[i]) for i in S}):
        table, axes = group_table(schema, schema.groups[g], rho, obs, set(S))
        shape = [1] * len(S)
        order = sorted(range(len(axes)), key=lambda a: S.index(axes[a]))
        table = np.transpose(table, order)
        for a in order:
            shape[S.index(axes[a])] = schema.fields[axes[a]].n
        full = full * table.reshape(shape)
    return full


def direct_kl(schema: Schema, rho: float, obs, S, logq: dict, max_cells: int = 2_000_000
              ) -> float | None:
    """KL(p*_S || prod_i q_i) by enumerating every joint value of S (None if too large)."""
    S = [int(i) for i in S]
    if int(np.prod([schema.fields[i].n for i in S])) > max_cells:
        return None
    p = joint_over(schema, rho, obs, S)
    logq_full = np.zeros(p.shape)
    for ax, i in enumerate(S):
        shape = [1] * len(S)
        shape[ax] = schema.fields[i].n
        logq_full = logq_full + np.asarray(logq[i], dtype=np.float64).reshape(shape)
    nz = p > 0
    return float((p[nz] * (np.log(p[nz]) - logq_full[nz])).sum())
