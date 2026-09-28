#!/usr/bin/env python3
"""How much disk the schema-v2 traces need, before any GPU time is spent on them.

Two measurements, then arithmetic:

* rows and forward passes per document come from the real harness on the toy
  model (exact for decoder and matched states, whose sizes are deterministic;
  in expectation for Bernoulli uniform masks), and are checked against the
  closed form used for task documents of other lengths;
* bytes per row come from writing real parquet with a toy whose vocabulary has
  each model's size, since the top-20 ids are the bulk of a row and their
  compressibility depends on the vocabulary.  The toy's top-20 below the true
  token is noise, i.e. close to incompressible, so this is an upper bound:
  a real model's lists repeat across steps and compress better.

Variants: top-20 as specified, top-10, and top-20 with float16 probabilities.

    python scripts/estimate_trace_size.py --out artifacts/trace_size_estimate.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3.provenance import collect as provenance
from c3.traces.collect import CollectConfig, Document, collect
from c3.traces.models import ToyDiffusionLM

VOCABS = {"llada-8b": 126464, "dream-7b": 152064, "local-110m": 50258}


def rows_and_passes(n: int, cfg: CollectConfig) -> tuple[float, int]:
    """Closed form: expected rows and forward passes for a maskable span of n tokens."""
    rows = cfg.masks_per_rate * sum(cfg.mask_rates) * n if cfg.include_uniform else 0.0
    passes = len(cfg.mask_rates) * cfg.masks_per_rate if cfg.include_uniform else 0
    for _ in cfg.decoder_policies if cfg.include_decoder else ():
        for k in cfg.decoder_k:
            steps = -(-n // k)
            dec_rows = sum(n - j * k for j in range(steps))
            rows += dec_rows * (2 if cfg.include_uniform_matched else 1)
            passes += steps * (2 if cfg.include_uniform_matched else 1)
    return float(rows), int(passes)


def task_span_lengths(task: str, jsonl_dir: str) -> list[int] | None:
    """Answer-span lengths of every task document, or None without the tokenizer/data."""
    try:
        from transformers import AutoTokenizer

        from c3.data.tasks import format_documents, load_task
        from c3.traces.models import KNOWN_MODELS

        spec = KNOWN_MODELS["llada-8b-instruct"]
        tok = AutoTokenizer.from_pretrained(spec["path"], revision=spec["revision"],
                                            trust_remote_code=True)
        docs = format_documents(task, load_task(task, Path(jsonl_dir) / f"{task}_test.jsonl"), tok)
    except Exception as exc:          # offline, no cache, no JSONL: fall back
        print(f"[size] {task}: real lengths unavailable ({type(exc).__name__}); using the median")
        return None
    return [d.span[1] - d.span[0] for d in docs]


def measured_rows(cfg: CollectConfig, n_docs: int, length: int) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    docs = [Document(i, rng.integers(0, 60, length)) for i in range(n_docs)]
    with tempfile.TemporaryDirectory() as d:
        out = collect(ToyDiffusionLM(), docs, cfg, Path(d) / "t.parquet", progress_every=0)
        meta = json.loads(out.with_suffix(".meta.json").read_text())
    return meta["n_rows"] / n_docs, meta["forward_passes"] / n_docs


def bytes_per_row(vocab: int, topk: int, fp16_probs: bool, n_docs: int) -> dict:
    """Parquet bytes per row, total and per column, on a toy with this vocabulary."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = CollectConfig(mask_rates=(0.1, 0.3, 0.5, 0.7, 0.9), masks_per_rate=2,
                        decoder_k=(8,), include_uniform_matched=True, topk=topk)
    rng = np.random.default_rng(1)
    docs = [Document(i, rng.integers(0, vocab - 1, 256)) for i in range(n_docs)]
    model = ToyDiffusionLM(vocab_size=vocab, mask_token_id=vocab - 1, noise_dispersion=0.3)
    with tempfile.TemporaryDirectory() as d:
        path = collect(model, docs, cfg, Path(d) / "t.parquet", progress_every=0)
        if fp16_probs:
            t = pq.read_table(path)
            i = t.schema.get_field_index("topk_p")
            flat = t.column("topk_p").combine_chunks().values.to_numpy(zero_copy_only=False)
            arr = pa.FixedSizeListArray.from_arrays(pa.array(flat.astype(np.float16)), topk)
            t = t.set_column(i, pa.field("topk_p", arr.type), arr)
            pq.write_table(t, path, compression="zstd", row_group_size=60_000)
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        per_col: dict[str, int] = {}
        for g in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(g)
            for c in range(rg.num_columns):
                col = rg.column(c)
                name = col.path_in_schema.split(".")[0]
                per_col[name] = per_col.get(name, 0) + col.total_compressed_size
        total = Path(path).stat().st_size
    return {"bytes_per_row": total / n, "rows_measured": n,
            "per_column_bytes_per_row": {k: round(v / n, 2) for k, v in
                                         sorted(per_col.items(), key=lambda kv: -kv[1])}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-for-bytes", type=int, default=2)
    ap.add_argument("--share1-free-gb", type=float, default=76.0)
    ap.add_argument("--task-lengths", default="runs/corpus/tasks",
                    help="directory with gsm8k/mbpp JSONL, to use real answer lengths")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    full = CollectConfig()                        # plan v2 Sec. 3.3 defaults
    pilot = CollectConfig(mask_rates=(0.1, 0.3, 0.5, 0.7, 0.9), decoder_k=(4,),
                          include_uniform_matched=False)
    designs = {"full (plan v2 defaults)": full,
               "full, no uniform_matched": CollectConfig(include_uniform_matched=False),
               "t2-pilot": pilot}
    per_doc = {}
    for name, cfg in designs.items():
        closed_rows, closed_passes = rows_and_passes(256, cfg)
        meas_rows, meas_passes = measured_rows(cfg, 4, 256)
        per_doc[name] = {"rows": meas_rows, "forward_passes": meas_passes,
                         "closed_form_rows": closed_rows, "closed_form_passes": closed_passes}
        print(f"[size] {name}: {meas_rows:.0f} rows/doc (closed form {closed_rows:.0f}), "
              f"{meas_passes:.0f} forward passes/doc", flush=True)

    variants = {"top20_fp32": (20, False), "top10_fp32": (10, False), "top20_fp16p": (20, True)}
    bpr = {}
    for model, vocab in VOCABS.items():
        for v, (k, f16) in variants.items():
            bpr[(model, v)] = bytes_per_row(vocab, k, f16, a.docs_for_bytes)
            print(f"[size] {model} {v}: {bpr[(model, v)]['bytes_per_row']:.1f} B/row", flush=True)

    # Task documents: decoder rows grow ~n^2 in the answer length, so use every
    # document's real span under the LLaDA-Instruct tokenizer when it is available.
    task_rows = {}
    for task, n_docs, median_len in (("gsm8k", 1319, 88), ("mbpp", 500, 50)):
        lengths, source = task_span_lengths(task, a.task_lengths), "llada-8b-instruct tokenizer"
        if lengths is None:
            lengths, source = [median_len] * n_docs, "median length (tokenizer unavailable)"
        rp = [rows_and_passes(n, full) for n in lengths]
        task_rows[task] = {"docs": len(lengths), "lengths_from": source,
                           "median_answer_tokens": float(np.median(lengths)),
                           "rows_per_doc": float(np.mean([r for r, _ in rp])),
                           "forward_passes_per_doc": float(np.mean([q for _, q in rp]))}

    fw_rows = per_doc["full (plan v2 defaults)"]["rows"]
    table = {}
    for v in variants:
        gb = lambda model, rows: rows * bpr[(model, v)]["bytes_per_row"] / 1e9   # noqa: E731
        plan = {
            "fineweb 2000 docs, local-110m": gb("local-110m", 2000 * fw_rows),
            "fineweb 2000 docs, llada-8b-base": gb("llada-8b", 2000 * fw_rows),
            "fineweb 2000 docs, dream-7b-base": gb("dream-7b", 2000 * fw_rows),
            "time-split 500 docs, llada-8b-base": gb("llada-8b", 500 * fw_rows),
            "time-split 500 docs, dream-7b-base": gb("dream-7b", 500 * fw_rows),
            "gsm8k+mbpp, llada-8b-instruct": gb("llada-8b", sum(
                t["docs"] * t["rows_per_doc"] for t in task_rows.values())),
            "gsm8k+mbpp, dream-7b-instruct": gb("dream-7b", sum(
                t["docs"] * t["rows_per_doc"] for t in task_rows.values())),
        }
        table[v] = {"gb_per_1000_fineweb_docs": {m: gb(m, 1000 * fw_rows) for m in VOCABS},
                    "plan_gb": plan, "plan_total_gb": sum(plan.values()),
                    "fits_share1": sum(plan.values()) < a.share1_free_gb}
        print(f"[size] {v}: plan total {sum(plan.values()):.1f} GB "
              f"(/share1 free ~{a.share1_free_gb:.0f} GB)", flush=True)

    result = {
        "per_document": per_doc, "task_documents": task_rows,
        "bytes_per_row": {f"{m}/{v}": r for (m, v), r in bpr.items()},
        "estimates": table, "share1_free_gb": a.share1_free_gb,
        "caveats": [
            "bytes/row are measured on a toy whose top-20 below the true token is noise: "
            "an upper bound for real models, whose lists repeat across steps",
            "uniform rows are an expectation (Bernoulli masks); decoder and matched rows are exact",
            "task rows use each document's answer length under the LLaDA-Instruct tokenizer "
            "(Dream's differs by a few tokens)",
            "the T12 decoders beyond top-k confidence would each add a decoder and a matched arm",
        ],
        "seconds": time.time() - t0, "provenance": provenance(),
    }
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"[size] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
