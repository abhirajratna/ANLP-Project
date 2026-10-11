#!/usr/bin/env python3
"""Run the contamination screen (c3.data.contamination) over the calibration corpus.

    python scripts/contamination_screen.py --model llada-8b-base --precision fp16 \\
        --device-map auto --max-memory 0=9GiB,1=9GiB \\
        --docs runs/corpus/calibration_docs.jsonl \\
        --out artifacts/contamination_llada-8b-base.json

The flag threshold is ``c3.CONTAM_THRESHOLD`` once it has been fixed; until then
``--threshold`` must be given explicitly, and it is recorded in the output.
Output: ``{"docs": {doc_id: {"run_length", "prefix_match", "n_match", "flagged"}},
"threshold", "n_flagged", ...}``, which ``analyze_traces.py --exclude-docs`` reads.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import c3
from c3.data.contamination import CONTINUATION_TOKENS, PREFIX_TOKENS, flag, screen_document
from c3.provenance import collect as provenance
from c3.traces.models import KNOWN_MODELS, build_model, parse_max_memory, set_deterministic
from c3.util import hard_exit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"{sorted(KNOWN_MODELS)}, local-110m or toy")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--precision", required=True, choices=["4bit", "fp16", "fp32", "bf16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None, choices=["auto"])
    ap.add_argument("--max-memory", default=None)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=int, default=getattr(c3, "CONTAM_THRESHOLD", None),
                    help="flag at this run length (default: c3.CONTAM_THRESHOLD, once fixed)")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threshold is None:
        ap.error("no contamination threshold: c3.CONTAM_THRESHOLD is not fixed yet, so pass "
                 "--threshold explicitly (the plan suggests 32 of 50)")

    set_deterministic(a.seed)
    model = build_model(a.model, precision=a.precision, path=a.model_path, device=a.device,
                        device_map=a.device_map, max_memory=parse_max_memory(a.max_memory))
    need = PREFIX_TOKENS + CONTINUATION_TOKENS
    if a.model == "toy":
        import numpy as np
        rng = np.random.default_rng(0)
        docs = [(i, rng.integers(0, 60, need)) for i in range(a.limit or 10)]
    else:
        from c3.data.fineweb import tokenize_documents
        docs = tokenize_documents(a.docs, model.tokenizer, 256)
        docs = docs[:a.limit] if a.limit else docs

    # Checkpointed so a killed job (walltime, node failure) resumes instead of
    # starting over: per-document results are appended to <out>.partial.jsonl.
    # A document whose logits overflow (fp16) is recorded and skipped rather than
    # ending the run; it is listed in "errors" and is neither flagged nor cleared.
    partial = Path(a.out).with_name(Path(a.out).name + ".partial.jsonl")
    partial.parent.mkdir(parents=True, exist_ok=True)
    results, errors = {}, {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue                          # a line cut off by the kill
            (errors if "error" in rec else results)[rec["doc_id"]] = rec.get("error", rec.get("result"))
        print(f"[contam] resuming: {len(results)} done, {len(errors)} errors in {partial}", flush=True)
    t0, n_new = time.time(), 0
    with partial.open("a") as fh:
        if partial.stat().st_size and not partial.read_bytes().endswith(b"\n"):
            fh.write("\n")                          # end the cut-off line first
        for i, (doc_id, ids) in enumerate(docs):
            key = str(int(doc_id))
            if key in results or key in errors:
                continue
            if hasattr(model, "set_reference"):      # the CPU toy model only
                model.set_reference(ids[:need])
            try:
                results[key] = screen_document(model, ids)
                fh.write(json.dumps({"doc_id": key, "result": results[key]}) + "\n")
            except FloatingPointError as e:
                errors[key] = str(e)
                fh.write(json.dumps({"doc_id": key, "error": str(e)}) + "\n")
                print(f"[contam] doc {key}: {e}", flush=True)
            fh.flush()
            n_new += 1
            if (i + 1) % 100 == 0:
                print(f"[contam] {i+1}/{len(docs)} docs, {n_new/(time.time()-t0)*3600:.0f} docs/hour",
                      flush=True)
    flagged = flag(results, a.threshold)
    runs = sorted(r["run_length"] for r in results.values())
    out = {"model": a.model, "precision": a.precision,
           "variant": getattr(model, "variant", None), "revision": getattr(model, "revision", None),
           "prefix_tokens": PREFIX_TOKENS, "continuation_tokens": CONTINUATION_TOKENS,
           "decoder": "greedy argmax, one most-confident token per step",
           "threshold": a.threshold,
           "threshold_source": ("c3.CONTAM_THRESHOLD"
                                if getattr(c3, "CONTAM_THRESHOLD", None) == a.threshold
                                else "--threshold (not yet pre-registered)"),
           "n_docs": len(results), "n_flagged": sum(r["flagged"] for r in flagged.values()),
           "errors": errors,
           "run_length_quantiles": {q: runs[min(len(runs) - 1, int(q * len(runs)))]
                                    for q in (0.5, 0.9, 0.99, 1.0)} if runs else {},
           "docs": flagged, "seconds": time.time() - t0, "source": a.docs,
           "provenance": provenance(with_gpu=True),
           "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[contam] {out['n_flagged']}/{out['n_docs']} flagged at run >= {a.threshold} -> {a.out}")
    return 0


if __name__ == "__main__":
    hard_exit(main())
