"""T10: C2 on real text -- parallel-sequential disagreement and the (B) proxies (plan v2 T10).

On the probe (T9, ``c3.probe``) the committed-token error splits exactly into
(A), the dependency error of committing k positions in one step, and (B), the
error in each position's own marginal.  On real text p* is unknown, so T10
reports proxies at the same states and committed positions:

* **(A): parallel-sequential disagreement.**  At a masked state, one forward
  pass ranks the masked positions by confidence; the k most confident are the
  commit set and their argmax tokens are the *parallel* commit.  The same k
  positions are then committed one at a time in that order: pass j has ranks
  1..j-1 filled with the tokens chosen before it, and its argmax at rank j is
  the *sequential* token.  The disagreement rate is the fraction of the k
  positions where the two differ.  Rank 1's sequential token comes from the
  state's own pass, so it always agrees and k = 1 is 0 by construction.  The
  commit sets nest (the top 2 lie inside the top 4 inside the top 8) and pass j
  depends only on ranks below j, so one chain of 8 passes per state serves
  k = 2, 4 and 8 at once.
* **(B): the committed-token gap and cross-entropy**, from the state's own pass
  at the same k positions: Delta = conf - 1[argmax == true token] (positive =
  overconfident) and CE = -log p(true token).

A model whose prediction at one position ignores every other position gives
zero disagreement; only dependence between the committed positions moves it.
Whether it tracks (A) is checked on the probe: the same proxy is computed on the
very states ``scripts/run_probe.py`` decodes (``probe_disagreement`` mirrors
``c3.probe.grid.run_trajectories`` under the confidence rule) and rank-correlated
with the exact (A) of each (rho, k) cell.

The document (a probe instance) is the unit: values are averaged within a
document first, and intervals resample whole documents (``c3.stats``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import N_BOOT
from .rng import generator
from .stats import doc_boot_counts, grouped_doc_mean, percentile_ci
from .traces.masks import uniform_masks

T10_RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
"""One uniform mask per rate per document: the ten states of plan v2 T10."""

T10_KS = (2, 4, 8)
"""Tokens per step for the disagreement proxy.  k = 1 is 0 by construction."""

B_KS = (1, 2, 4, 8)
"""k for the (B) proxies.  k = 1 is kept: the plan expects the estimation gap to
be present there too, where there is no dependency error at all."""

N_DOCS = 300
"""Fit-half documents in the real-text run (plan v2 T10: 300 documents x 10 states)."""

ZERO_A = 1e-12
"""|(A)| below this many nats is zero.  (A) is exactly 0 at rho = 0 and wherever no
pair is committed together; the float residue (~1e-17) must not be ranked."""

_DOC_ID = re.compile(r'\s*\{\s*"doc_id"\s*:\s*(\d+)\s*[,}]')


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #

def load_fit_documents(corpus: str | Path, manifest: str | Path, tokenizer, n_docs: int = N_DOCS,
                       max_tokens: int = 256) -> tuple[list, dict]:
    """The first ``n_docs`` fit-half documents (manifest value 0), in doc_id order.

    Tokenised as ``c3.data.fineweb.tokenize_documents`` does (first ``max_tokens``
    tokens, no special tokens) and with the same filter: a document shorter than
    ``max_tokens`` is skipped, so the run has exactly ``n_docs`` full documents.

    Nothing is computed on the report half before the pre-registration is
    frozen, so a report-half line is never parsed: the doc_id is read from the
    start of each line and only fit-half lines go on to ``json.loads``.
    """
    split = {int(k): int(v) for k, v in json.loads(Path(manifest).read_text()).items()}
    fit = sorted(d for d, h in split.items() if h == 0)
    wanted = set(fit)
    texts: dict[int, str] = {}
    with Path(corpus).open() as fh:
        for n, line in enumerate(fh):
            m = _DOC_ID.match(line)
            if m is None:
                raise ValueError(f"{corpus} line {n + 1} does not start with a doc_id")
            did = int(m.group(1))
            if did not in wanted:
                continue
            rec = json.loads(line)
            if int(rec["doc_id"]) != did:
                raise ValueError(f"{corpus} line {n + 1}: doc_id prefix {did} != {rec['doc_id']}")
            texts[did] = rec["text"]
    docs, short, missing = [], [], []
    for did in fit:
        if len(docs) == n_docs:
            break
        if did not in texts:
            missing.append(did)
            continue
        ids = tokenizer(texts[did], truncation=True, max_length=max_tokens,
                        add_special_tokens=False)["input_ids"]
        if len(ids) < max_tokens:
            short.append(did)
            continue
        docs.append((did, np.asarray(ids, dtype=np.int64)))
    if any(split[d] != 0 for d, _ in docs):
        raise RuntimeError("a report-half document was selected; refusing")
    if len(docs) < n_docs:
        raise ValueError(f"only {len(docs)} fit-half documents reach {max_tokens} tokens; "
                         f"asked for {n_docs}")
    return docs, {"n_fit_half": len(fit), "last_doc_id": int(docs[-1][0]) if docs else None,
                  "skipped_short": short, "missing": missing}


# --------------------------------------------------------------------------- #
# Forward passes
# --------------------------------------------------------------------------- #

def masked_stats(logits, rows, cols, mask_token_id: int, truth=None, chunk: int = 1024,
                 name: str = "model", precision: str = "na") -> dict:
    """top1, conf, ell and log p(true) at the (row, col) positions of ``logits``.

    The conventions of ``c3.decode.scoring.position_stats``: [MASK] excluded
    first (SUBS), float32, ell = z_top1 - logsumexp(z_others) exactly.  log p(true)
    is read from the log-softmax rather than as log(p_true), which underflows in
    float32 for a confidently wrong model.
    """
    import torch

    from .traces.models import check_finite_logits

    dev = logits.device
    r = torch.as_tensor(np.asarray(rows), dtype=torch.long, device=dev)
    c = torch.as_tensor(np.asarray(cols), dtype=torch.long, device=dev)
    t = None if truth is None else torch.as_tensor(np.asarray(truth), dtype=torch.long, device=dev)
    acc: dict[str, list] = {"top1": [], "conf": [], "ell": [], "logp_true": []}
    for s in range(0, r.numel(), chunk):
        z = logits[r[s:s + chunk], c[s:s + chunk]].float()
        check_finite_logits(z, name, precision, mask_token_id=mask_token_id)
        if 0 <= mask_token_id < z.shape[-1]:
            z[:, mask_token_id] = float("-inf")
        logp = torch.log_softmax(z, dim=-1)
        lp1, top1 = logp.max(dim=-1)
        others = z.scatter(1, top1[:, None], float("-inf"))
        acc["top1"].append(top1)
        acc["conf"].append(lp1.exp())
        acc["ell"].append((z.gather(1, top1[:, None])[:, 0] - torch.logsumexp(others, -1)).double())
        if t is not None:
            acc["logp_true"].append(logp.gather(1, t[s:s + chunk, None])[:, 0].double())
    out = {f: torch.cat(v).cpu().numpy() for f, v in acc.items() if v}
    if t is None:
        out["logp_true"] = np.full(r.numel(), np.nan)
    out["top1"] = out["top1"].astype(np.int64)
    return out


def sequential_tokens(model, ids: np.ndarray, ranked: list, first, counter: dict | None = None
                      ) -> list[np.ndarray]:
    """Commit each row's ranked positions one at a time; the argmax chosen at each.

    ``ids`` (B, L) are the states, [MASK] wherever masked.  ``ranked[b]`` is row
    b's commit set, best first; ``first[b]`` is the argmax at ``ranked[b][0]``
    from the state's own pass, which is the rank-1 sequential token by
    definition.  Pass j (j >= 2) fills ranks 1..j-1 with the tokens chosen
    before it -- the model's own, not the true ones -- and reads the argmax at
    rank j ([MASK] excluded).  Rows are batched pass by pass; a row drops out once
    its set is done.  Rank j's token depends only on passes 1..j, so a longer
    chain never changes a shorter one's tokens: the sets of k = 2, 4, 8 nest.
    """
    import torch

    from .traces.models import check_finite_logits

    counter = counter if counter is not None else {"forward_passes": 0}
    cur = np.array(ids, dtype=np.int64, copy=True)
    depth = [len(r) for r in ranked]
    seq = [np.full(d, -1, dtype=np.int64) for d in depth]
    for b, d in enumerate(depth):
        if d:
            seq[b][0] = int(first[b])
    mask_id = int(model.mask_token_id)
    for j in range(1, max(depth, default=0)):
        live = [b for b, d in enumerate(depth) if d > j]
        for b in live:
            cur[b, int(ranked[b][j - 1])] = seq[b][j - 1]
        x = torch.as_tensor(cur[live], dtype=torch.long, device=model.input_device)
        with torch.no_grad():
            lg = model.logits(x)
            at = torch.as_tensor([int(ranked[b][j]) for b in live], device=lg.device)
            z = lg[torch.arange(len(live), device=lg.device), at].float()
            del lg
            check_finite_logits(z, model.name, getattr(model, "precision", "na"),
                                mask_token_id=mask_id)
            if 0 <= mask_id < z.shape[-1]:
                z[:, mask_id] = float("-inf")
            tok = z.argmax(dim=-1).cpu().numpy()
        counter["forward_passes"] += 1
        for b, v in zip(live, tok):
            seq[b][j] = int(v)
    return seq


# --------------------------------------------------------------------------- #
# Real text
# --------------------------------------------------------------------------- #

@dataclass
class TextState:
    doc_id: int
    ids: np.ndarray          # the true tokens
    mask: np.ndarray         # True where masked
    rate_idx: int
    target_rate: float
    rate: float              # realised fraction masked


def text_states(docs, rates=T10_RATES, seed: int = 20260917, min_masked: int = max(T10_KS)
                ) -> list[TextState]:
    """One uniform mask per (document, rate), each from its own seeded stream.

    ``c3.traces.masks.uniform_masks`` draws the mask (independent Bernoulli, as in
    training).  ``min_masked`` is its floor: a state needs k masked positions to
    commit k.  At 256 tokens and rate 0.1 a floor of 8 binds in about one draw in
    140,000; the traces' floor of 32 would bind in 89% of them and turn rate 0.1
    into 0.125.  Seeding per (document, rate) keeps a state fixed whatever else
    is run.
    """
    out = []
    for doc_id, ids in docs:
        ids = np.asarray(ids, dtype=np.int64)
        for r, t in enumerate(rates):
            rng = generator(seed, "t10-uniform", str(int(doc_id)), f"{float(t):g}")
            (st,) = uniform_masks(ids.size, (float(t),), rng, min_masked=min_masked)
            out.append(TextState(int(doc_id), ids, st.mask, r, float(t), st.rate))
    return out


TEXT_COLUMNS = ("doc_id", "rate_idx", "target_rate", "mask_rate", "n_masked", "rank", "position",
                "conf", "ell", "logp_true", "true", "parallel", "sequential")
"""One row per (state, rank) of the commit chain."""


def run_text(model, docs, rates=T10_RATES, depth: int = max(T10_KS), seed: int = 20260917,
             batch_size: int = 32, min_masked: int = max(T10_KS), counter: dict | None = None,
             log=print) -> dict:
    """The proxies on real text: per state, its ``depth`` most confident positions.

    Columns of :data:`TEXT_COLUMNS`.  Positions are ranked by ell, the exact
    log-odds, which orders as confidence does without float32 ties at 1.0;
    remaining ties go to the lower position (``c3.traces.masks.topk_confidence``).
    States of equal length are batched across documents.
    """
    import torch

    counter = counter if counter is not None else {"forward_passes": 0}
    states = text_states(docs, rates, seed, min_masked)
    mask_id = int(model.mask_token_id)
    cols: dict[str, list] = {c: [] for c in TEXT_COLUMNS}
    by_len: dict[int, list[TextState]] = {}
    for s in states:
        by_len.setdefault(s.ids.size, []).append(s)
    done, n_batches = 0, 0
    for L in sorted(by_len):
        group = by_len[L]
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            n_batches += 1
            truth = np.stack([s.ids for s in batch])
            masks = np.stack([s.mask for s in batch])
            ids = truth.copy()
            ids[masks] = mask_id
            rows, pos = np.nonzero(masks)          # row-major: each state's positions in order
            x = torch.as_tensor(ids, dtype=torch.long, device=model.input_device)
            with torch.no_grad():
                lg = model.logits(x)
                # ~128 MB per float32 temporary: 667 rows at the 110M's vocabulary, 265 at
                # LLaDA's, which must fit beside 9 GiB of weights on an 11 GB card
                st = masked_stats(lg, rows, pos, mask_id, truth=truth[rows, pos],
                                  chunk=max(1, (32 << 20) // int(lg.shape[-1])),
                                  name=model.name, precision=getattr(model, "precision", "na"))
                del lg
            counter["forward_passes"] += 1
            ranked, first, picks = [], [], []
            for b in range(len(batch)):
                mine = np.flatnonzero(rows == b)
                order = mine[np.argsort(-st["ell"][mine], kind="stable")][:depth]
                picks.append(order)
                ranked.append(pos[order])
                first.append(st["top1"][order[0]])
            seq = sequential_tokens(model, ids, ranked, first, counter)
            for b, s in enumerate(batch):
                o, n = picks[b], picks[b].size
                for c, v in (("doc_id", s.doc_id), ("rate_idx", s.rate_idx),
                             ("target_rate", s.target_rate), ("mask_rate", s.rate),
                             ("n_masked", int(s.mask.sum()))):
                    cols[c].append(np.full(n, v))
                cols["rank"].append(np.arange(1, n + 1))
                cols["position"].append(pos[o])
                for c in ("conf", "ell", "logp_true"):
                    cols[c].append(st[c][o])
                cols["true"].append(truth[b, pos[o]])
                cols["parallel"].append(st["top1"][o])
                cols["sequential"].append(seq[b])
            done += len(batch)
            if n_batches % 20 == 0 or done == len(states):
                log(f"[t10] text: {done}/{len(states)} states, "
                    f"{counter['forward_passes']} forward passes")
    return {c: np.concatenate(v) if v else np.empty(0) for c, v in cols.items()}


def _metrics(par, seq, true, conf, logp_true) -> dict[str, np.ndarray]:
    agree = (par == true).astype(np.float64)
    return {"disagreement": (par != seq).astype(np.float64),
            "delta": conf - agree,
            "ce": -logp_true,
            "conf": conf,
            "acc_parallel": agree,
            "acc_sequential": (seq == true).astype(np.float64)}


def _est(gm, level) -> dict:
    g = gm.index(level)
    lo, hi = gm.ci()[g]
    return {"mean": float(gm.point[g]), "ci95": [float(lo), float(hi)]}


def summarise_text(cols: dict, ks=T10_KS, b_ks=B_KS, rates=T10_RATES, n_boot: int = N_BOOT,
                   seed: int = 0) -> dict:
    """Per k and per k x mask rate: document-first means with document-bootstrap intervals.

    A state counts towards k when it has at least k masked positions; its k
    ranks are averaged with the document's other states (each state contributes
    exactly k rows, so this is also the mean over the document's states).  Every
    estimand shares one resample matrix, so contrasts between k get intervals.
    """
    rank = np.asarray(cols["rank"])
    n_masked = np.asarray(cols["n_masked"])
    doc = np.asarray(cols["doc_id"])
    doc_ids = np.unique(doc)
    counts = doc_boot_counts(doc_ids.size, n_boot=n_boot, seed=seed, label="t10-text-docs")
    all_k = sorted(set(ks) | set(b_ks))
    pick = [np.flatnonzero((rank <= k) & (n_masked >= k)) for k in all_k]
    idx = np.concatenate(pick)
    kcol = np.concatenate([np.full(p.size, k) for p, k in zip(pick, all_k)])
    rcol = np.asarray(cols["rate_idx"])[idx]
    doc_idx = np.searchsorted(doc_ids, doc[idx])
    met = _metrics(*(np.asarray(cols[c])[idx] for c in
                     ("parallel", "sequential", "true", "conf", "logp_true")))
    by_k = {m: grouped_doc_mean(v, doc_idx, doc_ids.size, kcol, counts) for m, v in met.items()}
    by_kr = {m: grouped_doc_mean(v, doc_idx, doc_ids.size, kcol * 1000 + rcol, counts)
             for m, v in met.items()}
    # At rate 1.0 every document's input is the all-[MASK] sequence, so its parallel and
    # sequential tokens are the same for every document: one state, weighted 1/10 in per_k.
    tr = np.asarray(cols["target_rate"])[idx]
    part = tr < 1.0
    by_k_part = ({m: grouped_doc_mean(v[part], doc_idx[part], doc_ids.size, kcol[part], counts)
                  for m, v in met.items()} if part.any() and not part.all() else None)

    def wanted(m, k):
        # disagreement at k = 1 is kept as a check of the construction (exactly 0)
        return (k in ks or k == 1) if m == "disagreement" else k in b_ks

    states = np.unique(np.stack([doc, np.asarray(cols["rate_idx"])]), axis=1).shape[1]
    per_k = []
    for k in all_k:
        rec = {"k": int(k), "n_docs": int(by_k["conf"].n_docs[by_k["conf"].index(k)])}
        rec.update({m: _est(gm, k) for m, gm in by_k.items() if wanted(m, k)})
        per_k.append(rec)
    per_k_part = []
    for k in (all_k if by_k_part else []):
        rec = {"k": int(k), "n_docs": int(by_k_part["conf"].n_docs[by_k_part["conf"].index(k)])}
        rec.update({m: _est(gm, k) for m, gm in by_k_part.items() if wanted(m, k)})
        per_k_part.append(rec)
    per_k_rate = []
    for k in all_k:
        for r, t in enumerate(rates):
            code = k * 1000 + r
            if code not in by_kr["conf"].levels:
                continue
            rec = {"k": int(k), "rate": float(t),
                   "n_docs": int(by_kr["conf"].n_docs[by_kr["conf"].index(code)])}
            rec.update({m: _est(gm, code) for m, gm in by_kr.items() if wanted(m, k)})
            per_k_rate.append(rec)
    contrasts = {}
    for m, pairs in (("disagreement", [(4, 2), (8, 4), (8, 2)]), ("delta", [(8, 1)]),
                     ("ce", [(8, 1)])):
        for hi, lo in pairs:
            if hi in by_k[m].levels and lo in by_k[m].levels:
                pt, draws = by_k[m].contrast(hi, lo)
                contrasts[f"{m}_k{hi}_minus_k{lo}"] = {"mean": pt, "ci95": list(percentile_ci(draws))}
    real = np.asarray(cols["mask_rate"])
    rate_real = [{"rate": float(t), "realised_mean": float(np.mean(real[np.asarray(cols["rate_idx"]) == r]))}
                 for r, t in enumerate(rates) if np.any(np.asarray(cols["rate_idx"]) == r)]
    rank1 = rank == 1
    return {"n_docs": int(doc_ids.size), "n_states": int(states),
            "rank1_always_agrees": bool(np.all(np.asarray(cols["parallel"])[rank1]
                                               == np.asarray(cols["sequential"])[rank1])),
            "per_k": per_k, "per_k_rates_below_1": per_k_part, "per_k_rate": per_k_rate,
            "contrasts": contrasts,
            "realised_rates": rate_real}


# --------------------------------------------------------------------------- #
# Probe calibration
# --------------------------------------------------------------------------- #

def score_probe(model, batch, obs) -> tuple[np.ndarray, list[dict]]:
    """``c3.probe.grid.score`` plus each field's full-vocabulary argmax.

    The same operations in the same order, so log q and the top-1 probability
    -- and therefore the confidence rule's choices -- are the grid's bit for bit.
    Returns the masked input ids and, per instance, {field: (log q over its
    values, top-1 probability, argmax token id)} for every masked field.
    """
    import torch

    from .probe.truth import MASKED

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
        best = logp.max(dim=-1)
        top = best.values.exp()
        arg = best.indices.cpu().numpy()
    out = []
    for b, (p, o) in enumerate(zip(batch, obs)):
        res = {}
        for i in np.flatnonzero(o == MASKED):
            row = logp[b, int(p.pos[i])]
            res[int(i)] = (row[torch.as_tensor(p.cand[i], device=row.device)].double().cpu().numpy(),
                           float(top[b, int(p.pos[i])]), int(arg[b, int(p.pos[i])]))
        out.append(res)
    return ids, out


def probe_disagreement(model, spec, prepared, k: int, batch_size: int = 32,
                       counter: dict | None = None, log=print) -> list[dict]:
    """The proxy on every state of the grid's confidence-rule trajectories for this k.

    Mirrors ``c3.probe.grid.run_trajectories`` (length groups, batches, the
    teacher-forced commits) so that each state is the grid's; at each one the
    commit set S is ``grid.select("confidence", ...)``, (A) is recomputed exactly
    (``c3.probe.truth.decompose``) to check the alignment and drive the
    bootstrap, and S is committed one field at a time in decreasing confidence
    (ties to the lower field, as the rule breaks them).  Only field values are
    ever masked or filled, so the rho marker stays visible and unscored, as in
    the grid.  One row per (instance, step).
    """
    from .probe.grid import select
    from .probe.truth import MASKED, decompose

    counter = counter if counter is not None else {"forward_passes": 0}
    rows = []
    by_len: dict[int, list] = {}
    for p in prepared:
        by_len.setdefault(p.ids.size, []).append(p)
    for L in sorted(by_len):
        group = by_len[L]
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            obs = [np.full(len(p.inst.values), MASKED, dtype=np.int64) for p in batch]
            step = 0
            while True:
                live = [b for b, o in enumerate(obs) if (o == MASKED).any()]
                if not live:
                    break
                ids, scored = score_probe(model, [batch[b] for b in live], [obs[b] for b in live])
                counter["forward_passes"] += 1
                ranked, first, sets = [], [], []
                for b, sc in zip(live, scored):
                    S = select("confidence", sc, k, None)          # sorted, as the grid has it
                    order = sorted(S, key=lambda i: (-sc[i][1], i))
                    sets.append((S, order))
                    ranked.append(batch[b].pos[order])
                    first.append(sc[order[0]][2])
                seq = sequential_tokens(model, ids, ranked, first, counter)
                for j, (b, sc) in enumerate(zip(live, scored)):
                    p, o, (S, order) = batch[b], obs[b], sets[j]
                    sch = spec.schemas[p.inst.schema]
                    d = decompose(sch, p.inst.rho, o, S, {i: sc[i][0] for i in sc})
                    par = np.array([sc[i][2] for i in order])
                    true = p.ids[p.pos[order]]
                    rows.append({
                        "instance_id": p.inst.instance_id, "schema": p.inst.schema,
                        "rho": p.inst.rho, "k": k, "step": step, "n_fields": len(o),
                        "n_masked": len(sc), "n_commit": len(S), "A": d["A"], "B": d["B"],
                        "n_disagree": int((par != seq[j]).sum()),
                        "n_agree_true_parallel": int((par == true).sum()),
                        "n_agree_true_sequential": int((seq[j] == true).sum()),
                        # summed in the grid's order, so equal states give equal floats
                        "conf_commit_mean": float(np.mean([sc[i][1] for i in S])),
                    })
                    o[S] = p.inst.values[S]
                step += 1
    log(f"[t10] probe k={k}: {len(prepared)} instances, {len(rows)} states, "
        f"{counter['forward_passes']} forward passes so far")
    return rows


def spearman_draws(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Spearman correlation of two (n_draws, n_cells) arrays (average ranks for ties)."""
    from scipy.stats import rankdata

    rx = rankdata(np.atleast_2d(x), axis=1)
    ry = rankdata(np.atleast_2d(y), axis=1)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rx * ry).sum(axis=1) / np.where(den > 0, den, 1.0), np.nan)


def _zero_small(a):
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.abs(a) < ZERO_A, 0.0, a)


def summarise_probe(rows: list[dict], grid: dict, n_boot: int = N_BOOT, seed: int = 0,
                    label: str = "probe") -> dict:
    """Per (rho, k) cell: mean disagreement beside the grid's exact (A); their Spearman.

    Per instance, the disagreement is pooled over its trajectory's committed
    fields (every field is committed once, so it is the count over the number of
    fields), the counterpart of the grid's (A) per field.  The point Spearman
    uses the grid JSON's cell means; its interval resamples instances within
    each rho (the grid's 200 per rho), one resample shared by every k and by both
    quantities, with (A) per instance recomputed on the same states.  k = 1 cells
    are left out: both quantities are 0 there by construction.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    inst = (df.groupby(["k", "rho", "instance_id"], sort=True)
              .agg(n_disagree=("n_disagree", "sum"), n_commit=("n_commit", "sum"),
                   A=("A", "sum"), n_fields=("n_fields", "first"),
                   acc_par=("n_agree_true_parallel", "sum"),
                   acc_seq=("n_agree_true_sequential", "sum"))
              .reset_index())
    if (inst["n_commit"] != inst["n_fields"]).any():
        raise RuntimeError("a trajectory did not commit every field exactly once")
    inst["D"] = inst["n_disagree"] / inst["n_commit"]
    inst["A_pf"] = _zero_small(inst["A"]) / inst["n_fields"]
    grid_cells = {(c["k"], float(c["rho"])): c for c in grid.get("cells", [])
                  if c["rule"] == "confidence"}
    counts = {}
    cells, d_draws, a_draws = [], [], []
    for (k, rho), g in inst.groupby(["k", "rho"], sort=True):
        g = g.sort_values("instance_id")
        key = float(rho)
        if key not in counts:
            counts[key] = (g["instance_id"].to_numpy(),
                           doc_boot_counts(len(g), n_boot=n_boot, seed=seed,
                                           label=f"t10-{label}-rho={key:g}").astype(np.float64))
        ids, C = counts[key]
        if not np.array_equal(ids, g["instance_id"].to_numpy()):
            raise RuntimeError(f"rho={key}: the instances differ across k")
        D, A = g["D"].to_numpy(), g["A_pf"].to_numpy()
        d_draws.append(C @ D / D.size)
        a_draws.append(C @ A / A.size)
        gc = grid_cells.get((int(k), key))
        grid_a = None if gc is None else float(_zero_small(gc["A_per_field"]["mean"]))
        cells.append({
            "rho": key, "k": int(k), "n_instances": int(len(g)),
            "disagreement": {"mean": float(D.mean()), "ci95": list(percentile_ci(d_draws[-1]))},
            "A_per_field_grid": grid_a,
            "A_per_field_recomputed": float(A.mean()),
            "A_grid": None if gc is None else float(_zero_small(gc["A"]["mean"])),
            "B_per_field_grid": None if gc is None else float(gc["B_per_field"]["mean"]),
            "acc_parallel": float((g["acc_par"] / g["n_commit"]).mean()),
            "acc_sequential": float((g["acc_seq"] / g["n_commit"]).mean()),
        })
    use = [j for j, c in enumerate(cells) if c["k"] >= 2]
    out = {"cells": cells, "spearman": None}
    if len(use) >= 3:
        d_pt = np.array([cells[j]["disagreement"]["mean"] for j in use])
        a_grid = [cells[j]["A_per_field_grid"] for j in use]
        a_pt = np.array([cells[j]["A_per_field_recomputed"] if a is None else a
                         for j, a in zip(use, a_grid)])
        draws = spearman_draws(np.stack([d_draws[j] for j in use], axis=1),
                               np.stack([a_draws[j] for j in use], axis=1))
        out["spearman"] = {
            "rho_s": float(spearman_draws(d_pt[None], a_pt[None])[0]),
            "ci95": list(percentile_ci(draws)),
            "n_cells": len(use), "against": "A_per_field (grid JSON, confidence rule)",
            "n_boot": int(n_boot), "n_draws_defined": int(np.isfinite(draws).sum()),
        }
    gap = [abs(c["A_per_field_recomputed"] - c["A_per_field_grid"]) for c in cells
           if c["A_per_field_grid"] is not None]
    out["max_abs_cell_A_recomputed_minus_grid"] = float(max(gap)) if gap else None
    return out


def align_with_grid_steps(rows: list[dict], steps, tol: float = 1e-4) -> dict:
    """Are these the grid's states?  Compare every (instance, k, step) with steps_<tag>.parquet.

    A step is the same state when it has as many masked fields, commits as many,
    with the same exact (A), and its (B) and mean committed confidence agree to
    ``tol`` (relative for (B)).  (A) and (B) depend on which fields are committed,
    so a different commit set moves them by far more than float noise; but a
    forward pass over a different batch (``--probe-limit``) moves confidences by
    ~1e-6, so bit-for-bit agreement is counted separately.
    """
    import pandas as pd

    keys = ["instance_id", "k", "step"]
    vals = ["n_masked", "n_commit", "A", "B", "conf_commit_mean"]
    mine = pd.DataFrame(rows)[keys + vals]
    ks = sorted(mine["k"].unique())
    theirs = steps[(steps["rule"] == "confidence") & steps["k"].isin(ks)
                   & steps["instance_id"].isin(mine["instance_id"].unique())][keys + vals]
    m = mine.merge(theirs, on=keys, how="outer", suffixes=("", "_grid"), indicator=True)
    both = m[m["_merge"] == "both"]
    dA = (both["A"] - both["A_grid"]).abs()
    dB = (both["B"] - both["B_grid"]).abs()
    dc = (both["conf_commit_mean"] - both["conf_commit_mean_grid"]).abs()
    same = ((both["n_masked"] == both["n_masked_grid"]) & (both["n_commit"] == both["n_commit_grid"])
            & (dA <= 1e-9) & (dB <= tol * np.maximum(1.0, both["B_grid"].abs())) & (dc <= tol))
    bitwise = same & (dA <= ZERO_A) & (dB == 0) & (dc == 0)
    return {"steps_here": int(len(mine)), "steps_grid": int(len(theirs)),
            "steps_in_both": int(len(both)), "steps_same_state": int(same.sum()),
            "steps_bitwise": int(bitwise.sum()), "tol": tol,
            "max_abs_A_diff": float(dA.max()) if len(both) else None,
            "max_abs_B_diff": float(dB.max()) if len(both) else None,
            "max_abs_conf_commit_diff": float(dc.max()) if len(both) else None}
