"""FineWeb infilling: the free-running decoder test of set-level C3 (deviations item 10).

Each 256-token calibration document has the span ``SPAN`` = positions 64..191
masked and is decoded greedily, the whole span at once (no blocks), by one
commit policy (``c3.decode.policies``).  Greedy decoding is deterministic, so a
configuration needs no seeds.  Per document it records

    acc               fraction of span tokens equal to the original
    n_forward         forward passes; tokens/step = |span| / n_forward
    step_all_right    fraction of steps whose committed tokens were all right
    n_steps           = n_forward (one commit per forward pass)

``infill_batch`` decodes several documents together: one forward pass over the
documents still unfinished, then each document's policy choice from its own
rows.  Every per-position statistic is computed row by row (``position_stats``),
and the KLASS history is kept per (document, position), so a document's
trajectory is the one ``c3.decode.loop.decode`` gives it alone -- the tests hold
the two to the same tokens and step counts on the toy model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

SPAN = (64, 192)
HALVES = ("dev-fit", "dev-eval", "fit", "report")
STAT_FIELDS = ("top1", "conf", "ell", "p_true", "entropy", "margin")


def select_docs(split: dict, half: str) -> set[int]:
    """Document ids of ``half`` under the committed split (0 = fit, 1 = report).

    dev-fit / dev-eval re-split the fit half with ``hash_halves(., 1)``, the
    partition ``scripts/make_dev_subset.py`` writes, so the infilling and trace
    analyses use the same development documents.
    """
    from .frame import hash_halves

    if half not in HALVES:
        raise ValueError(f"half must be one of {HALVES}, got {half!r}")
    split = {int(k): int(v) for k, v in split.items()}
    if half == "report":
        return {d for d, h in split.items() if h == 1}
    fit = np.array(sorted(d for d, h in split.items() if h == 0), dtype=np.int64)
    if half == "fit":
        return set(fit.tolist())
    dev = hash_halves(fit, 1)
    return set(fit[dev == (0 if half == "dev-fit" else 1)].tolist())


def config_key(cfg: dict) -> str:
    """A stable short name for one policy configuration."""
    knobs = ",".join(f"{k}={cfg[k]}" for k in sorted(cfg) if k != "policy")
    return f"{cfg['policy']}[{knobs}]"


def config_id(cfg: dict) -> int:
    """A 31-bit id of a configuration, for state ids in on-policy logs."""
    h = hashlib.blake2b(config_key(cfg).encode(), digest_size=4).digest()
    return int.from_bytes(h, "little") & 0x7FFFFFFF


def build_policy(cfg: dict, calibrator=None, set_calibrator=None, log_freq=None):
    """A policy object from a configuration dict such as {"policy": "c3set", "alpha": 0.1}."""
    from .decode.policies import CALIBRATED, POLICIES

    name = cfg["policy"]
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; known: {sorted(POLICIES)}")
    kw = {k: v for k, v in cfg.items() if k != "policy"}
    if name in CALIBRATED:
        if calibrator is None:
            raise ValueError(f"{name} needs a token calibrator (--calibrator)")
        kw["calibrator"] = calibrator
    if name == "c3set":
        if set_calibrator is None:
            raise ValueError("c3set needs a set calibrator (--set-calibrator)")
        kw["set_calibrator"] = set_calibrator
    if name == "pc_sampler" and log_freq is not None:
        kw["log_freq"] = log_freq
    return POLICIES[name](**kw)


@dataclass
class StepLog:
    """On-policy states: every masked position at every step, for refitting calibrators."""

    columns: dict

    @classmethod
    def empty(cls) -> "StepLog":
        return cls({k: [] for k in ("doc_id", "state_id", "step", "position", "ell", "conf",
                                    "entropy", "agree", "n_masked", "mask_rate", "committed")})

    def add(self, doc_id, state_id, step, positions, stats, ref_tokens, chosen, seq_len):
        n = positions.size
        committed = np.zeros(n, dtype=bool)
        committed[chosen] = True
        c = self.columns
        c["doc_id"].append(np.full(n, doc_id, np.int64))
        c["state_id"].append(np.full(n, state_id, np.int64))
        c["step"].append(np.full(n, step, np.int64))
        c["position"].append(positions.astype(np.int64))
        c["ell"].append(np.asarray(stats["ell"], np.float64))
        c["conf"].append(np.asarray(stats["conf"], np.float64))
        c["entropy"].append(np.asarray(stats["entropy"], np.float64))
        c["agree"].append((np.asarray(stats["top1"]) == ref_tokens).astype(np.int8))
        c["n_masked"].append(np.full(n, n, np.int64))
        c["mask_rate"].append(np.full(n, n / seq_len, np.float64))
        c["committed"].append(committed)

    def frame(self):
        import pandas as pd

        if not self.columns["doc_id"]:
            return pd.DataFrame({k: [] for k in self.columns})
        return pd.DataFrame({k: np.concatenate(v) for k, v in self.columns.items()})


def scatter_masks(doc_ids, length: int, span=SPAN, rate: float = 0.5, seed: int = 0) -> np.ndarray:
    """(n_docs, length) masks hiding round(rate x |span|) positions of ``span`` per document,
    drawn uniformly by a generator keyed by (seed, doc_id): fixed per document, independent
    of which other documents are decoded with it."""
    from .rng import generator

    a, b = span
    n = int(round(rate * (b - a)))
    out = np.zeros((len(doc_ids), length), dtype=bool)
    for r, d in enumerate(doc_ids):
        pick = generator(seed, "infill-scatter", int(d)).choice(b - a, size=n, replace=False)
        out[r, a + pick] = True
    return out


def gap_masks(doc_ids, length: int, span=SPAN, gap: int = 8, rate: float = 0.5,
              seed: int = 0) -> np.ndarray:
    """(n_docs, length) masks of round(rate x |span|) / ``gap`` contiguous gaps of ``gap``
    tokens inside ``span``, separated by at least one visible token, placed uniformly at
    random per document (keyed by (seed, gap, doc_id)).

    The visible tokens are split into the n + 1 stretches around the gaps by a uniform
    composition with every interior stretch >= 1, so gaps never merge.
    """
    from .rng import generator

    a, b = span
    width = b - a
    n_masked = int(round(rate * width))
    if gap < 1 or n_masked % gap:
        raise ValueError(f"{n_masked} masked tokens do not split into gaps of {gap}")
    n = n_masked // gap
    free = width - n_masked - (n - 1)               # visible tokens beyond the separators
    if free < 0:
        raise ValueError(f"{n} gaps of {gap} do not fit in {width} positions")
    out = np.zeros((len(doc_ids), length), dtype=bool)
    for r, d in enumerate(doc_ids):
        rng = generator(seed, "infill-gaps", int(gap), int(d))
        # Stars and bars: n bars among free + n slots give the n + 1 stretch lengths.
        bars = np.sort(rng.choice(free + n, size=n, replace=False))
        stretch = np.diff(np.r_[-1, bars, free + n]) - 1     # n + 1 lengths summing to free
        pos = a + stretch[0]
        for j in range(n):
            out[r, pos:pos + gap] = True
            pos += gap + stretch[j + 1] + 1
    return out


def infill_batch(model, docs, policy, span=SPAN, topk: int = 1, log: StepLog | None = None,
                 state_base: int = 0, masks: np.ndarray | None = None) -> list[dict]:
    """Infill every document in ``docs`` (a list of (doc_id, ids)) with ``policy``: the whole
    ``span``, or with ``masks`` ((n_docs, L) bool, inside ``span``) only those positions."""
    import time

    import torch

    from .decode.policies import StepContext
    from .decode.scoring import position_stats
    from .traces.models import check_finite_logits

    t0 = time.perf_counter()
    mask_id = int(model.mask_token_id)
    ref = np.stack([np.asarray(ids, dtype=np.int64) for _, ids in docs])
    B, L = ref.shape
    a, b = span
    if not (0 <= a < b <= L):
        raise ValueError(f"span {span} does not fit length {L}")
    X = ref.copy()
    if masks is None:
        masks = np.zeros((B, L), dtype=bool)
        masks[:, a:b] = True
    masks = np.asarray(masks, dtype=bool)
    if masks.shape != (B, L) or masks[:, :a].any() or masks[:, b:].any() or not masks.any(1).all():
        raise ValueError("masks must be (n_docs, L), inside the span, at least one per document")
    X[masks] = mask_id
    region = np.arange(a, b)
    dev = getattr(model, "input_device", None)
    needs_kl = bool(getattr(policy, "needs_kl", False))
    kl_hist = [dict() for _ in range(B)]
    prev = have_prev = None
    if needs_kl:
        prev = None                                    # (B, |span|, V), allocated on first use
        have_prev = np.zeros((B, region.size), dtype=bool)
    n_forward = np.zeros(B, np.int64)
    all_right = np.zeros(B, np.int64)
    alive = np.ones(B, dtype=bool)
    step = 0
    while alive.any():
        rows = np.flatnonzero(alive)
        t = torch.as_tensor(X[rows], dtype=torch.long)
        if dev is not None:
            t = t.to(dev)
        masked = X[rows] == mask_id
        with torch.no_grad():
            lg = model.logits(t)
            check_finite_logits(lg, getattr(model, "name", "model"),
                                getattr(model, "precision", "na"), mask_token_id=mask_id)
            stats = position_stats(lg, masked, mask_id, true_ids=ref[rows], topk=topk, chunk=1024)
            if needs_kl:
                prev = _klass_update(lg, stats, rows, a, mask_id, prev, have_prev, kl_hist)
        cut = np.searchsorted(stats["rows"], np.arange(rows.size + 1))
        for i, r in enumerate(rows.tolist()):
            s, e = int(cut[i]), int(cut[i + 1])
            sub = {f: stats[f][s:e] for f in STAT_FIELDS}
            positions = stats["cols"][s:e].astype(np.int64)
            ctx = StepContext(step=step, positions=positions, gen_start=a,
                              kl_history=kl_hist[r], n_masked_total=e - s, seq_len=L)
            order, n_commit = policy.select(sub, ctx)
            n_commit = max(1, min(int(n_commit), e - s))
            chosen = np.asarray(order[:n_commit], dtype=np.int64)
            commit = positions[chosen]
            tokens = np.asarray(sub["top1"])[chosen]
            if log is not None:
                log.add(docs[r][0], state_base + int(docs[r][0]) * 1000 + step, step, positions,
                        sub, ref[r, positions], chosen, L)
            X[r, commit] = tokens
            all_right[r] += int(np.all(tokens == ref[r, commit]))
            n_forward[r] += 1
            if needs_kl:
                have_prev[r, commit - a] = False
            if not np.any(X[r, a:b] == mask_id):
                alive[r] = False
        step += 1
    secs = (time.perf_counter() - t0) / B
    out = []
    for r, (doc_id, _) in enumerate(docs):
        m = masks[r]
        out.append({"doc_id": int(doc_id), "acc": float(np.mean(X[r, m] == ref[r, m])),
                    "n_forward": int(n_forward[r]), "n_committed": int(m.sum()),
                    "tokens_per_step": float(m.sum() / n_forward[r]),
                    "step_all_right": float(all_right[r] / n_forward[r]),
                    "seconds": float(secs), "tokens": X[r, a:b].copy()})
    return out


def _klass_update(lg, stats, rows, a, mask_id, prev, have_prev, kl_hist, chunk: int = 512):
    """KL(p_prev || p_now) over the full vocabulary, per (document, position), as
    ``c3.decode.loop.decode`` computes it; returns the updated previous-log-p store."""
    import torch

    r_loc = stats["rows"]
    cols = stats["cols"].astype(np.int64)
    V = lg.shape[-1]
    if prev is None:
        prev = torch.empty((have_prev.shape[0], have_prev.shape[1], V), dtype=torch.float32,
                           device=lg.device)
    for s in range(0, r_loc.size, chunk):
        rl = r_loc[s:s + chunk]
        c = cols[s:s + chunk]
        g = rows[rl]
        z = lg[torch.as_tensor(rl, device=lg.device), torch.as_tensor(c, device=lg.device)].float()
        if 0 <= mask_id < V:
            z[:, mask_id] = float("-inf")
        logp = torch.log_softmax(z, dim=-1)
        gi = torch.as_tensor(g, device=lg.device)
        si = torch.as_tensor(c - a, device=lg.device)
        had = have_prev[g, c - a]
        if had.any():
            idx = np.flatnonzero(had)
            ti = torch.as_tensor(idx, device=lg.device)
            old = prev[gi[ti], si[ti]]
            p_old = old.exp()
            terms = torch.where(p_old > 0, p_old * (old - logp[ti]), torch.zeros_like(p_old))
            kl = terms.sum(-1).clamp_min(0.0).cpu().numpy()
            for j, v in zip(idx.tolist(), kl.tolist()):
                kl_hist[int(g[j])].setdefault(int(c[j]), []).append(float(v))
        prev[gi, si] = logp
        have_prev[g, c - a] = True
    return prev


def summarise_tokens(rows: list[dict]) -> list[dict]:
    """Per-document records without the token arrays, plus a digest of the tokens."""
    out = []
    for r in rows:
        rec = {k: v for k, v in r.items() if k != "tokens"}
        rec["tokens_sha1"] = hashlib.sha1(np.asarray(r["tokens"], np.int64).tobytes()).hexdigest()[:16]
        out.append(rec)
    return out


def load_configs(path) -> list[dict]:
    """A knob grid: {"configs": [{"policy": ..., knob: value}, ...]} or a bare list."""
    from pathlib import Path

    d = json.loads(Path(path).read_text())
    cfgs = d["configs"] if isinstance(d, dict) else d
    keys = [config_key(c) for c in cfgs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate configurations")
    return cfgs


# --------------------------------------------------------------------------- #
# Accuracy at matched speed
# --------------------------------------------------------------------------- #

def interp_at(tps: np.ndarray, acc: np.ndarray, target: float) -> float:
    """Accuracy at ``target`` tokens/step, linear in log2(tokens/step) between a
    policy's configurations; NaN outside the range the policy reached."""
    tps = np.asarray(tps, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    ok = np.isfinite(tps) & np.isfinite(acc) & (tps > 0)
    if not ok.any():
        return float("nan")
    x, y = np.log2(tps[ok]), acc[ok]
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    t = np.log2(target)
    if t < x[0] - 1e-12 or t > x[-1] + 1e-12:
        return float("nan")
    # Several configurations at the same speed (e.g. top-k k=4 and a threshold
    # that happens to land at 4): average them.
    ux, inv = np.unique(np.round(x, 12), return_inverse=True)
    uy = np.bincount(inv, weights=y) / np.bincount(inv)
    if ux.size == 1:
        return float(uy[0])
    return float(np.interp(t, ux, uy))


def interp_draws(tps: np.ndarray, acc: np.ndarray, target: float) -> np.ndarray:
    """``interp_at`` for every bootstrap draw: ``tps``, ``acc`` are (n_boot, n_configs)."""
    return np.array([interp_at(tps[i], acc[i], target) for i in range(tps.shape[0])])
