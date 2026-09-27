#!/usr/bin/env python3
"""Sanity checks for the trained 110M model (plan v2 T3, gate G2).

* The perplexity upper bound exp(NELBO) on held-out documents -- by default the
  2,000 calibration documents, first 256 tokens each, never seen in training --
  with a document-bootstrap interval.
* A handful of unconditional samples, decoded, for a human to read.

The NELBO uses a fixed grid of diffusion times and fixed mask draws, so two runs
on the same checkpoint give the same number.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="final.pt or checkpoint.pt")
    ap.add_argument("--docs", required=True, help="JSONL of held-out documents (text field)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-t", type=int, default=32, help="diffusion-time grid points")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--sample-steps", type=int, default=256)
    ap.add_argument("--raw-weights", action="store_true", help="evaluate raw instead of EMA weights")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/eval_110m.json")
    a = ap.parse_args()

    import torch

    from c3.provenance import collect as provenance
    from c3.stats import doc_boot_counts, percentile_ci
    from c3.traces.models import LocalMDLM
    from c3.training.model import nelbo_eval
    from c3.training.sample import ancestral_sample

    dev = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
    m = LocalMDLM(path=a.ckpt, device=dev, use_ema=not a.raw_weights).load()
    tok = m.tokenizer
    docs = []
    with open(a.docs, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids = tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"][: a.max_tokens]
                if ids:
                    docs.append(ids)
    docs = docs[: a.limit] if a.limit else docs

    per_doc_nelbo, per_doc_tokens = [], []
    for i in range(0, len(docs), a.batch_size):
        chunk = docs[i:i + a.batch_size]
        L = max(len(d) for d in chunk)
        ids = np.full((len(chunk), L), 50256, dtype=np.int64)
        valid = np.zeros((len(chunk), L), dtype=bool)
        for b, d in enumerate(chunk):
            ids[b, :len(d)] = d
            valid[b, :len(d)] = True
        # One document at a time inside the batch so each gets its own total.
        for b in range(len(chunk)):
            s, n = nelbo_eval(m.model, torch.from_numpy(ids[b:b + 1]).to(dev), m.mask_token_id,
                              torch.from_numpy(valid[b:b + 1]).to(dev), n_t=a.n_t, seed=i + b)
            per_doc_nelbo.append(s)
            per_doc_tokens.append(n)
    nel, ntok = np.array(per_doc_nelbo), np.array(per_doc_tokens)
    point = float(nel.sum() / ntok.sum())
    counts = doc_boot_counts(len(docs), a.n_boot, seed=0, label="eval-110m").astype(float)
    draws = (counts @ nel) / (counts @ ntok)
    lo, hi = percentile_ci(draws)

    samples = []
    if a.samples:
        x = ancestral_sample(m.model, a.samples, a.max_tokens, m.mask_token_id,
                             steps=a.sample_steps, device=dev)
        samples = [tok.decode([t for t in row.tolist() if t != m.mask_token_id]) for row in x]

    res = {"checkpoint": a.ckpt, "weights": m.weights, "documents": len(docs),
           "tokens": int(ntok.sum()), "nelbo_per_token": point, "nelbo_ci": [lo, hi],
           "ppl_bound": math.exp(point), "ppl_bound_ci": [math.exp(lo), math.exp(hi)],
           "n_t": a.n_t, "samples": samples, "provenance": provenance(with_gpu=True)}
    print(json.dumps({k: v for k, v in res.items() if k not in ("provenance", "samples")}, indent=2))
    for s in samples:
        print("---\n" + s[:600])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
