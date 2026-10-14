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

from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from .. import N_BOOT
from ..rng import generator
from ..stats import doc_boot_counts, percentile_ci
from . import KS, RULES
from .instances import Instance, ProbeSpec, serialize
from .tokens import value_tokens
from .truth import MASKED, decompose, direct_kl, exact_joint_correct, kl, marginals


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


ScoredField = namedtuple("ScoredField", "logq top value entropy ell")
"""One masked field after a forward pass: log q over the field's values, the top-1
probability over the whole vocabulary, which value the top-1 token is (-1: none of
them), the entropy of the vocabulary distribution (nats) and the exact top-1
log-odds.  Indexes like the (logq, top) pair it replaces."""

POLICY_RULES = ("threshold", "entropy_bounded", "c3a", "c3indep", "c3set")
"""Rules with a speed knob instead of a fixed k: the decoders of ``c3.decode.policies``
applied to the masked fields (position = field index)."""


def score(model, batch: list[Prepared], obs: list[np.ndarray]) -> list[dict]:
    """One forward pass for a batch of equal-length states.

    Returns, per instance, {field: ScoredField} for every masked field.  The [MASK]
    token is excluded from the prediction (SUBS).
    """
    import torch

    from ..decode.scoring import position_stats

    mask_id = int(model.mask_token_id)
    ids = np.stack([p.ids for p in batch])
    where = np.zeros(ids.shape, dtype=bool)
    for b, (p, o) in enumerate(zip(batch, obs)):
        m = np.flatnonzero(o == MASKED)
        ids[b, p.pos[m]] = mask_id
        where[b, p.pos[m]] = True
    x = torch.as_tensor(ids, dtype=torch.long, device=model.input_device)
    with torch.no_grad():
        lg = model.logits(x)
        # Exact log-odds, entropy and top-1 at the masked positions only (chunked; SUBS).
        st = position_stats(lg, where, mask_id, topk=1, chunk=512)
        r_t = torch.as_tensor(st["rows"], device=lg.device)
        c_t = torch.as_tensor(st["cols"], device=lg.device)
        z = lg[r_t, c_t].float()
        if 0 <= mask_id < z.shape[-1]:
            z[:, mask_id] = float("-inf")
        logp = torch.log_softmax(z, dim=-1)
    at = {(int(b), int(c)): j for j, (b, c) in enumerate(zip(st["rows"], st["cols"]))}
    out = []
    for b, (p, o) in enumerate(zip(batch, obs)):
        res = {}
        for i in np.flatnonzero(o == MASKED):
            j = at[(b, int(p.pos[i]))]
            row = logp[j]
            hit = np.flatnonzero(p.cand[i] == st["top1"][j])
            res[int(i)] = ScoredField(
                row[torch.as_tensor(p.cand[i], device=row.device)].double().cpu().numpy(),
                float(row.max().exp()), int(hit[0]) if hit.size else -1,
                float(st["entropy"][j]), float(st["ell"][j]))
        out.append(res)
    return out


def _policy_stats(scored: dict, n_fields: int):
    from ..decode.policies import StepContext

    masked = np.array(sorted(scored), dtype=np.int64)
    stats = {"ell": np.array([scored[i].ell for i in masked]),
             "conf": np.array([scored[i].top for i in masked]),
             "entropy": np.array([scored[i].entropy for i in masked])}
    ctx = StepContext(step=0, positions=masked, n_masked_total=int(masked.size),
                      seq_len=int(n_fields))
    return masked, stats, ctx


def select(rule: str, scored: dict, k, rng: np.random.Generator, policy=None,
           n_fields: int | None = None) -> list[int]:
    """The fields committed this step.  ``k`` is the tokens per step of random /
    confidence; the knob rules (``POLICY_RULES``) take a ``policy`` object."""
    masked = sorted(scored)
    if rule == "random":
        return sorted(int(i) for i in rng.choice(masked, size=min(k, len(masked)), replace=False))
    if rule == "confidence":
        order = sorted(masked, key=lambda i: (-scored[i][1], i))
        return sorted(order[:k])
    if rule in POLICY_RULES:
        if policy is None or n_fields is None:
            raise ValueError(f"rule {rule!r} needs a policy and n_fields")
        fields, stats, ctx = _policy_stats(scored, n_fields)
        order, n = policy.select(stats, ctx)
        n = max(1, min(int(n), fields.size))
        return sorted(int(fields[j]) for j in order[:n])
    raise ValueError(f"unknown rule {rule!r}")


def set_predictions(policy, scored: dict, S, n_fields: int) -> dict:
    """P_ind (product of the token calibrator's c_hat over S) and, for C3-set, P_set(S)."""
    from ..decode.policies import calibrated_confidence

    fields, stats, ctx = _policy_stats(scored, n_fields)
    c, _ = calibrated_confidence(policy.calibrator, stats, ctx)
    idx = np.searchsorted(fields, np.asarray(S))
    out = {"P_ind": float(np.prod(c[idx]))}
    if getattr(policy, "set_calibrator", None) is not None:
        _, p = policy.set_probs(stats, ctx)
        out["P_set"] = float(np.minimum.accumulate(p)[len(S) - 1])
    return out


def run_trajectories(model, spec: ProbeSpec, prepared: list[Prepared], k, rule: str,
                     seed: int, batch_size: int = 32, max_cells: int = 200_000, policy=None,
                     states: list | None = None) -> list[dict]:
    """Decode every instance with (k, rule); one row per (instance, step).

    ``k`` is the tokens per step for random / confidence and the knob value (recorded
    as ``k``) for the ``POLICY_RULES``, which need ``policy``.  Every row also has the
    exact joint correctness J of committing the decoder's top tokens on S
    (``truth.exact_joint_correct``), the expected number of them that are wrong, and
    whether they are all equal to the instance's values; calibrated rules add P_ind and
    P_set.  ``states``, if given, collects every masked field of every state (for
    fitting the probe's calibrators).
    """
    rows = []
    by_len: dict[int, list[Prepared]] = {}
    for p in prepared:
        by_len.setdefault(p.ids.size, []).append(p)
    calibrated = rule in ("c3a", "c3indep", "c3set")
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
                    S = select(rule, sc, k, rngs[b], policy=policy, n_fields=len(o))
                    logq = {i: sc[i][0] for i in sc}
                    d = decompose(sch, p.inst.rho, o, S, logq)
                    masked = sorted(sc)
                    marg = marginals(sch, p.inst.rho, o, masked)
                    kl_all = [kl(marg[i], logq[i]) for i in masked]
                    direct = direct_kl(sch, p.inst.rho, o, S, logq, max_cells=max_cells)
                    top = [sc[i].value for i in S]
                    p_right = {i: (float(marg[i][sc[i].value]) if sc[i].value >= 0 else 0.0)
                               for i in masked}
                    row = {
                        "instance_id": p.inst.instance_id, "schema": p.inst.schema,
                        "rho": p.inst.rho, "k": k, "rule": rule, "step": step,
                        "n_fields": len(o), "n_masked": len(masked), "n_commit": len(S),
                        "A": d["A"], "B": d["B"],
                        "B_masked_mean": float(np.mean(kl_all)),
                        "B_commit_mean": d["B"] / len(S),
                        "conf_commit_mean": float(np.mean([sc[i][1] for i in S])),
                        "conf_masked_mean": float(np.mean([sc[i][1] for i in masked])),
                        "direct_kl": np.nan if direct is None else direct,
                        "J": exact_joint_correct(sch, p.inst.rho, o, S, top),
                        "err_commit": float(sum(1.0 - p_right[i] for i in S)),
                        "all_right": bool(all(sc[i].value == p.inst.values[i] for i in S)),
                        "P_ind": np.nan, "P_set": np.nan,
                    }
                    if calibrated:
                        row.update(set_predictions(policy, sc, S, len(o)))
                    rows.append(row)
                    if states is not None:
                        for i in masked:
                            states.append({
                                "doc_id": p.inst.instance_id,
                                "state_id": p.inst.instance_id * 1000 + step, "step": step,
                                "position": i, "ell": sc[i].ell, "conf": sc[i].top,
                                "entropy": sc[i].entropy,
                                "agree": int(sc[i].value == p.inst.values[i]),
                                "p_right": p_right[i], "n_masked": len(masked),
                                "mask_rate": len(masked) / len(o), "rho": p.inst.rho})
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
        # Instances have 6-16 fields, so totals weight schemas by size; the per-field
        # versions weight instances equally and are what the crossover is read from.
        cells.append({"rule": rule, "k": k, "rho": rho, "n_instances": len(ids),
                      "A": est(A), "B": est(B), "A_plus_B": est(A + B), "A_minus_B": est(A - B),
                      "A_per_field": est(A / n), "B_per_field": est(B / n),
                      "A_minus_B_per_field": est((A - B) / n),
                      "share_A": float((A / n).sum() / max(1e-300, ((A + B) / n).sum()))})
    crossover = {}
    for rule in sorted({c["rule"] for c in cells}):
        for rho in sorted({c["rho"] for c in cells}):
            mine = sorted((c for c in cells if c["rule"] == rule and c["rho"] == rho),
                          key=lambda c: c["k"])
            first = next((c["k"] for c in mine
                          if c["A_per_field"]["mean"] > c["B_per_field"]["mean"]), None)
            sure = next((c["k"] for c in mine if c["A_minus_B_per_field"]["ci95"][0] > 0), None)
            crossover[f"{rule}/rho={rho}"] = {"first_k_A_exceeds_B": first,
                                              "first_k_A_exceeds_B_ci": sure}
    checked = [r for r in rows if np.isfinite(r["direct_kl"])]
    err = max((abs(r["A"] + r["B"] - r["direct_kl"]) for r in checked), default=float("nan"))
    return {"cells": cells, "crossover": crossover,
            "identity": {"steps_checked": len(checked), "steps_total": len(rows),
                         "max_abs_err_nats": err}}


MATCHED_METRICS = ("A_per_field", "B_per_field", "AB_per_field", "err_per_field", "J_step",
                   "all_right_step", "cal_err_set", "cal_err_ind")
CONTRASTS = (("c3set", "confidence"), ("c3set", "entropy_bounded"), ("c3set", "c3indep"),
             ("c3set", "threshold"), ("c3indep", "confidence"))


def summarise_matched(rows: list[dict], speeds=(2.0, 4.0), n_boot: int = N_BOOT,
                      seed: int = 0) -> dict:
    """Every rule at matched fields per step, on one instance bootstrap.

    Per configuration (rule, knob) and instance: (A), (B), (A)+(B) and the expected
    number of wrong commits, each per field; the mean exact joint correctness J and
    realised all-right rate per step; and, for calibrated rules, the mean |P - J|
    of the set model and of the independence product.  Speed is fields / steps
    (summed over instances).  Each rule's curve is interpolated at ``speeds`` in
    log2(speed) inside every draw (``c3.infill.interp_draws``); contrasts are
    differences of those draws.
    """
    import pandas as pd

    from ..infill import interp_at, interp_draws

    df = pd.DataFrame(rows)
    df["cfg"] = df["rule"].astype(str) + "@" + df["k"].astype(str)
    df["cal_set"] = (df["P_set"] - df["J"]).abs()
    df["cal_ind"] = (df["P_ind"] - df["J"]).abs()
    per = df.groupby(["cfg", "instance_id"]).agg(
        rule=("rule", "first"), k=("k", "first"), n_fields=("n_fields", "first"),
        steps=("step", "size"), A=("A", "sum"), B=("B", "sum"), err=("err_commit", "sum"),
        J=("J", "mean"), all_right=("all_right", "mean"), cal_set=("cal_set", "mean"),
        cal_ind=("cal_ind", "mean")).reset_index()
    inst = np.array(sorted(df["instance_id"].unique()))
    cfgs = sorted(per["cfg"].unique())
    n_i = per.groupby("cfg")["instance_id"].nunique()
    if (n_i != inst.size).any():
        raise ValueError(f"configurations ran different instances: {n_i[n_i != inst.size].to_dict()}")
    per["A_per_field"] = per["A"] / per["n_fields"]
    per["B_per_field"] = per["B"] / per["n_fields"]
    per["AB_per_field"] = (per["A"] + per["B"]) / per["n_fields"]
    per["err_per_field"] = per["err"] / per["n_fields"]
    per["J_step"], per["all_right_step"] = per["J"], per["all_right"]
    per["cal_err_set"], per["cal_err_ind"] = per["cal_set"], per["cal_ind"]

    def mat(col):
        return per.pivot(index="cfg", columns="instance_id", values=col).loc[cfgs, inst].to_numpy(float)

    counts = doc_boot_counts(inst.size, n_boot=n_boot, seed=seed,
                             label="probe-matched").astype(np.float64)
    NF, ST = mat("n_fields"), mat("steps")
    speed_pt, speed_dr = NF.sum(1) / ST.sum(1), (counts @ NF.T) / (counts @ ST.T)
    first = per.drop_duplicates("cfg").set_index("cfg").loc[cfgs]
    rule_of = first["rule"].to_numpy()
    pts, drs = {}, {}
    for m in MATCHED_METRICS:
        M = mat(m)
        whole = np.isfinite(M).all(axis=1)       # cal_err_* exist for calibrated rules only
        pts[m] = np.where(whole, np.nan_to_num(M).mean(axis=1), np.nan)
        drs[m] = counts @ np.nan_to_num(M).T / inst.size
        drs[m][:, ~whole] = np.nan
    configs = [{"config": c, "rule": rule_of[j], "knob": float(first["k"].iloc[j]),
                "fields_per_step": float(speed_pt[j]),
                **{m: float(pts[m][j]) for m in MATCHED_METRICS}} for j, c in enumerate(cfgs)]
    matched, mdraws = {}, {}
    for r in sorted(set(rule_of)):
        idx = np.flatnonzero(rule_of == r)
        matched[r], mdraws[r] = {}, {}
        for t in speeds:
            matched[r][str(t)], mdraws[r][t] = {}, {}
            for m in MATCHED_METRICS:
                pt = interp_at(speed_pt[idx], pts[m][idx], t)
                dr = interp_draws(speed_dr[:, idx], drs[m][:, idx], t)
                lo, hi = percentile_ci(dr)
                matched[r][str(t)][m] = {"value": pt, "ci": [lo, hi]}
                mdraws[r][t][m] = dr
    contrasts = {}
    for x, y in CONTRASTS:
        if x in mdraws and y in mdraws:
            contrasts[f"{x}-{y}"] = {
                str(t): {m: dict(zip(("value", "ci"), (
                    matched[x][str(t)][m]["value"] - matched[y][str(t)][m]["value"],
                    list(percentile_ci(mdraws[x][t][m] - mdraws[y][t][m])))))
                    for m in ("A_per_field", "AB_per_field", "err_per_field", "J_step")}
                for t in speeds}
    cal = {}
    for r in ("c3set", "c3indep", "c3a"):
        sub = df[df["rule"] == r]
        if len(sub):
            pi = sub.groupby("instance_id")[["cal_set", "cal_ind"]].mean().reindex(inst)
            d = (pi["cal_ind"] - pi["cal_set"]).to_numpy(float)
            rec = {"abs_err_set": float(pi["cal_set"].mean()), "abs_err_ind": float(pi["cal_ind"].mean())}
            if np.isfinite(d).all():
                rec["ind_minus_set"] = {"value": float(d.mean()),
                                        "ci": list(percentile_ci(counts @ d / inst.size))}
            cal[r] = rec
    zero = df[df["rho"] == 0.0]
    return {"speeds": list(speeds), "n_instances": int(inst.size), "configs": configs,
            "matched": matched, "contrasts": contrasts, "calibration": cal,
            "max_A_at_rho0": float(zero["A"].abs().max()) if len(zero) else float("nan"),
            "max_A_at_rho0_by_rule": zero.groupby("rule")["A"].apply(lambda v: float(v.abs().max())).to_dict()
            if len(zero) else {}}


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
