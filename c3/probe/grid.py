"""The T9 grid: rho x tokens per step k x selection rule, on any diffusion LM.

For each evaluation instance and each (k, rule) the decoder starts with every
field value masked (the JSON structure and the rho marker are observed).  At
each step one forward pass gives p_theta,i for every masked field; the rule
picks S (k fields: uniformly at random, or the k most confident by the top-1
probability over the whole vocabulary, which is what a confidence decoder
sees); (A) and (B) for committing S are computed exactly (``c3.probe.truth``);
then S is filled with the instance's true values -- teacher forcing, as in
plan v2 Sec. 3.3, so the path is a draw from p* -- and the next step begins.

Per instance the step terms are summed: (A)+(B) is then the KL of the whole
parallel decode against the true distribution along that path.  Each cell's
mean and 95% interval come from a document bootstrap over instances.

Wherever the joint table over S is small enough, the direct KL is computed as
well and the largest |(A)+(B) - KL| is reported: the identity check on real
model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import N_BOOT
from ..rng import generator
from ..stats import doc_boot_counts, percentile_ci
from . import KS, RULES
from .instances import Instance, ProbeSpec, serialize
from .tokens import value_tokens
from .truth import MASKED, decompose, direct_kl, kl, marginals


@dataclass
class Prepared:
    inst: Instance
    ids: np.ndarray                 # token ids of the full instance
    pos: np.ndarray                 # token index of each field's value
    cand: list                      # candidate token ids per field (this tokenizer)


def prepare(spec: ProbeSpec, instances, tokenizer, tok_name: str) -> list[Prepared]:
    """Tokenise each instance and locate every field's value token.

    Fails if a value is not one token in context or its token differs from the
    id recorded in the spec: the exactness of p* depends on both.
    """
    out = []
    for inst in instances:
        sch = spec.schemas[inst.schema]
        text, spans = serialize(sch, inst.values, inst.rho)
        ids, which = value_tokens(tokenizer, text, spans)
        if any(w is None for w in which):
            raise ValueError(f"instance {inst.instance_id}: a value is not one {tok_name} token")
        cand = [np.asarray(f.tokens[tok_name], dtype=np.int64) for f in sch.fields]
        for k, (f, w) in enumerate(zip(sch.fields, which)):
            if ids[w] != cand[k][inst.values[k]]:
                raise ValueError(f"instance {inst.instance_id} field {f.path}: token {ids[w]} "
                                 f"is not the spec's {cand[k][inst.values[k]]}")
        out.append(Prepared(inst, np.asarray(ids, dtype=np.int64), np.asarray(which), cand))
    return out


def score(model, batch: list[Prepared], obs: list[np.ndarray]) -> list[dict]:
    """One forward pass for a batch of equal-length states.

    Returns, per instance, {field: (log q over its values, top-1 prob over the vocabulary)}
    for every masked field.  The [MASK] token is excluded from the prediction (SUBS).
    """
    import torch

    ids = np.stack([p.ids for p in batch])
    for b, (p, o) in enumerate(zip(batch, obs)):
        m = np.flatnonzero(o == MASKED)
        ids[b, p.pos[m]] = model.mask_token_id
    x = torch.as_tensor(ids, dtype=torch.long, device=model.input_device)
    with torch.no_grad():
        lg = model.logits(x).float()
        if 0 <= int(model.mask_token_id) < lg.shape[-1]:
            lg[..., int(model.mask_token_id)] = float("-inf")
        logp = torch.log_softmax(lg, dim=-1)
        top = logp.max(dim=-1).values.exp()
    out = []
    for b, (p, o) in enumerate(zip(batch, obs)):
        res = {}
        for i in np.flatnonzero(o == MASKED):
            row = logp[b, int(p.pos[i])]
            res[int(i)] = (row[torch.as_tensor(p.cand[i], device=row.device)].double().cpu().numpy(),
                           float(top[b, int(p.pos[i])]))
        out.append(res)
    return out


def select(rule: str, scored: dict, k: int, rng: np.random.Generator) -> list[int]:
    masked = sorted(scored)
    if rule == "random":
        return sorted(int(i) for i in rng.choice(masked, size=min(k, len(masked)), replace=False))
    if rule == "confidence":
        order = sorted(masked, key=lambda i: (-scored[i][1], i))
        return sorted(order[:k])
    raise ValueError(f"unknown rule {rule!r}")


def run_trajectories(model, spec: ProbeSpec, prepared: list[Prepared], k: int, rule: str,
                     seed: int, batch_size: int = 32, max_cells: int = 200_000) -> list[dict]:
    """Decode every instance with (k, rule); one row per (instance, step)."""
    rows = []
    by_len: dict[int, list[Prepared]] = {}
    for p in prepared:
        by_len.setdefault(p.ids.size, []).append(p)
    for L in sorted(by_len):
        group = by_len[L]
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            obs = [np.full(len(p.inst.values), MASKED, dtype=np.int64) for p in batch]
            rngs = [generator(seed, "probe-select", p.inst.instance_id, k, rule) for p in batch]
            step = 0
            while True:
                live = [b for b, o in enumerate(obs) if (o == MASKED).any()]
                if not live:
                    break
                scored = score(model, [batch[b] for b in live], [obs[b] for b in live])
                for b, sc in zip(live, scored):
                    p, o = batch[b], obs[b]
                    sch = spec.schemas[p.inst.schema]
                    S = select(rule, sc, k, rngs[b])
                    logq = {i: sc[i][0] for i in sc}
                    d = decompose(sch, p.inst.rho, o, S, logq)
                    masked = sorted(sc)
                    marg = marginals(sch, p.inst.rho, o, masked)
                    kl_all = [kl(marg[i], logq[i]) for i in masked]
                    direct = direct_kl(sch, p.inst.rho, o, S, logq, max_cells=max_cells)
                    rows.append({
                        "instance_id": p.inst.instance_id, "schema": p.inst.schema,
                        "rho": p.inst.rho, "k": k, "rule": rule, "step": step,
                        "n_fields": len(o), "n_masked": len(masked), "n_commit": len(S),
                        "A": d["A"], "B": d["B"],
                        "B_masked_mean": float(np.mean(kl_all)),
                        "B_commit_mean": d["B"] / len(S),
                        "conf_commit_mean": float(np.mean([sc[i][1] for i in S])),
                        "conf_masked_mean": float(np.mean([sc[i][1] for i in masked])),
                        "direct_kl": np.nan if direct is None else direct,
                    })
                    o[S] = p.inst.values[S]
                step += 1
    return rows


def instance_totals(rows: list[dict]) -> dict:
    """{(rule, k, rho): {instance_id: (A_total, B_total, n_fields)}}."""
    tot: dict = {}
    for r in rows:
        cell = tot.setdefault((r["rule"], r["k"], r["rho"]), {})
        a, b, n = cell.get(r["instance_id"], (0.0, 0.0, r["n_fields"]))
        cell[r["instance_id"]] = (a + r["A"], b + r["B"], n)
    return tot


def summarise(rows: list[dict], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Per cell: mean (A), (B), (A)+(B) and (A)-(B) per instance and per field, with
    document-bootstrap 95% intervals; the crossover k per (rule, rho); the identity check."""
    cells = []
    for (rule, k, rho), per in sorted(instance_totals(rows).items()):
        ids = sorted(per)
        A = np.array([per[i][0] for i in ids])
        B = np.array([per[i][1] for i in ids])
        n = np.array([per[i][2] for i in ids], dtype=np.float64)
        counts = doc_boot_counts(len(ids), n_boot=n_boot, seed=seed,
                                 label=f"probe-{rule}-{k}-{rho}").astype(np.float64)
        def est(v):
            draws = counts @ v / len(ids)
            lo, hi = percentile_ci(draws)
            return {"mean": float(v.mean()), "ci95": [lo, hi]}
        cells.append({"rule": rule, "k": k, "rho": rho, "n_instances": len(ids),
                      "A": est(A), "B": est(B), "A_plus_B": est(A + B), "A_minus_B": est(A - B),
                      "A_per_field": est(A / n), "B_per_field": est(B / n),
                      "share_A": float(A.sum() / max(1e-300, (A + B).sum()))})
    crossover = {}
    for rule in sorted({c["rule"] for c in cells}):
        for rho in sorted({c["rho"] for c in cells}):
            mine = sorted((c for c in cells if c["rule"] == rule and c["rho"] == rho),
                          key=lambda c: c["k"])
            first = next((c["k"] for c in mine if c["A"]["mean"] > c["B"]["mean"]), None)
            sure = next((c["k"] for c in mine if c["A_minus_B"]["ci95"][0] > 0), None)
            crossover[f"{rule}/rho={rho}"] = {"first_k_A_exceeds_B": first,
                                              "first_k_A_exceeds_B_ci": sure}
    checked = [r for r in rows if np.isfinite(r["direct_kl"])]
    err = max((abs(r["A"] + r["B"] - r["direct_kl"]) for r in checked), default=float("nan"))
    return {"cells": cells, "crossover": crossover,
            "identity": {"steps_checked": len(checked), "steps_total": len(rows),
                         "max_abs_err_nats": err}}


def run_grid(model, spec: ProbeSpec, prepared: list[Prepared], ks=KS, rules=RULES,
             seed: int = 0, batch_size: int = 32, log=print) -> list[dict]:
    rows = []
    for rule in rules:
        for k in ks:
            got = run_trajectories(model, spec, prepared, k, rule, seed, batch_size)
            log(f"[grid] {rule} k={k}: {len(got)} steps, mean A {np.mean([r['A'] for r in got]):.4f} "
                f"B {np.mean([r['B'] for r in got]):.4f} per step")
            rows += got
    return rows
