"""Per-position statistics from one forward pass (plan v2 Sec. 3.4).

For every masked position of every sequence in a batch this computes, on the
logits' device and in float32:

    top1, conf        the model's top-1 token and its probability
    ell               confidence log-odds, exactly: z_top1 - logsumexp(z_others)
    p_true            probability of the true token (NaN when no truth is given)
    entropy           of the predictive distribution, in nats
    margin            p_top1 - p_top2
    topk_ids/topk_p   the top-``topk`` tokens and probabilities

The mask token is excluded first (SUBS): a masked slot can never be filled with
[MASK].  Only the masked rows are gathered and they are processed in chunks, so
a 126k-token vocabulary at 256 positions never materialises a full-batch
softmax on an 11 GB card.  Only ``topk`` values per position leave the GPU.
"""

from __future__ import annotations

import numpy as np

FIELDS = ("top1", "conf", "ell", "p_true", "entropy", "margin", "topk_ids", "topk_p")


def position_stats(logits, masked, mask_token_id: int, true_ids=None, topk: int = 20,
                   chunk: int = 1024) -> dict:
    """Statistics for every True entry of ``masked`` (row-major order).

    ``logits``: (B, L, V) tensor.  ``masked``: (B, L) bool tensor or array.
    ``true_ids``: optional (B, L) integer tensor/array of reference tokens.
    Returns numpy arrays; ``rows`` and ``cols`` give the (b, position) of each.
    """
    import torch

    dev = logits.device
    m = torch.as_tensor(np.asarray(masked) if not torch.is_tensor(masked) else masked,
                        dtype=torch.bool, device=dev)
    rows, cols = torch.nonzero(m, as_tuple=True)
    n = rows.numel()
    V = logits.shape[-1]
    k = min(topk, V - 1)
    out = {"rows": rows.cpu().numpy(), "cols": cols.cpu().numpy()}
    if n == 0:
        for f in FIELDS:
            out[f] = np.empty((0, k) if f.startswith("topk") else 0)
        return out
    truth = None
    if true_ids is not None:
        truth = torch.as_tensor(np.asarray(true_ids) if not torch.is_tensor(true_ids) else true_ids,
                                dtype=torch.long, device=dev)[rows, cols]

    acc = {f: [] for f in FIELDS}
    for s in range(0, n, chunk):
        r, c = rows[s:s + chunk], cols[s:s + chunk]
        z = logits[r, c].float()
        if 0 <= mask_token_id < V:
            z[:, mask_token_id] = float("-inf")
        logp = torch.log_softmax(z, dim=-1)
        p = logp.exp()
        tp, ti = torch.topk(p, k + 1 if k + 1 <= V else k, dim=-1)
        z1 = z.gather(1, ti[:, :1])[:, 0]
        others = z.scatter(1, ti[:, :1], float("-inf"))
        ell = z1 - torch.logsumexp(others, dim=-1)
        ent = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(-1)
        acc["top1"].append(ti[:, 0])
        acc["conf"].append(tp[:, 0])
        acc["ell"].append(ell.double())
        acc["entropy"].append(ent)
        acc["margin"].append(tp[:, 0] - tp[:, 1])
        acc["topk_ids"].append(ti[:, :k].to(torch.int32))
        acc["topk_p"].append(tp[:, :k])
        if truth is not None:
            acc["p_true"].append(p.gather(1, truth[s:s + chunk, None])[:, 0])
    for f in FIELDS:
        if f == "p_true" and truth is None:
            out[f] = np.full(n, np.nan)
            continue
        out[f] = torch.cat(acc[f]).cpu().numpy()
    out["top1"] = out["top1"].astype(np.int64)
    return out
