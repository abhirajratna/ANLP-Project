#!/usr/bin/env python3
"""Run the trace harness over the calibration corpus for one model arm."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3.traces.collect import CollectConfig, collect
from c3.traces.models import KNOWN_MODELS, build_model
from c3.util import hard_exit


def load_docs(path, tokenizer, max_tokens, limit=None):
    from c3.data.fineweb import tokenize_documents
    docs = tokenize_documents(path, tokenizer, max_tokens)
    return docs[:limit] if limit else docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help=f"{sorted(KNOWN_MODELS)}, 'local-110m' or 'toy'")
    ap.add_argument("--model-path", default=None,
                    help="local checkpoint dir (e.g. /share1/ishita/llada_model)")
    ap.add_argument("--precision", default="4bit", choices=["4bit", "fp16", "fp32", "bf16"])
    ap.add_argument("--docs", required=True, help="calibration_docs.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="20 for the Stage 0 pilot")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--mask-rates", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--decoder-states", type=int, default=4)
    ap.add_argument("--no-decoder", action="store_true")
    ap.add_argument("--repeat", type=int, default=1,
                    help=">1 re-runs the same documents to measure the non-determinism floor")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = build_model(a.model, precision=a.precision, path=a.model_path, device=a.device)
    if a.model == "toy":
        rng = np.random.default_rng(0)
        docs = [(i, rng.integers(0, 60, a.max_tokens)) for i in range(a.limit or 20)]
    else:
        # Each model is evaluated under its own tokeniser, so token counts per
        # document differ slightly across arms; the state fixed effects absorb it.
        docs = load_docs(a.docs, model.tokenizer, a.max_tokens, a.limit)
    print(f"[collect] {len(docs)} documents, model={a.model}, precision={a.precision}")

    cfg = CollectConfig(mask_rates=tuple(a.mask_rates), decoder_states=a.decoder_states,
                        include_decoder=not a.no_decoder)
    outputs = []
    for rep in range(a.repeat):
        out = Path(a.out) if a.repeat == 1 else Path(a.out).with_name(
            Path(a.out).stem + f".rep{rep}" + Path(a.out).suffix)
        # A fresh seed per repeat would change the mask patterns; the floor is
        # about non-determinism at *identical* inputs, so the seed is held fixed.
        outputs.append(str(collect(model, docs, cfg, out)))
    print(json.dumps({"outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    # hard_exit: torch/transformers worker threads can abort at finalization too
    hard_exit(main())
