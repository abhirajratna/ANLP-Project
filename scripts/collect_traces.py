#!/usr/bin/env python3
"""Run the trace harness over a corpus for one model arm.

Raw text (the FineWeb calibration corpus):

    python scripts/collect_traces.py --model llada-8b-base --precision fp16 \\
        --device-map auto --max-memory 0=10GiB,1=10GiB \\
        --docs runs/corpus/calibration_docs.jsonl \\
        --shard 0 --num-shards 4 --resume --out runs/traces/llada-base/llada-base

Tasks (question visible, answer masked):

    python scripts/collect_traces.py --model llada-8b-instruct --task gsm8k ...

With ``--num-shards``/``--resume`` the output is part files under the ``--out``
stem (merge them with scripts/merge_traces.py); a single-shard run is merged
automatically.  Without either, one parquet file is written at ``--out``.

``--preset t2-pilot`` fixes the plan v2 T2 pilot design (c3.traces.collect.PRESETS);
it needs an explicit, non-4-bit ``--precision``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3.traces.collect import PRESETS, CollectConfig, Document, collect, collect_shard
from c3.traces.masks import DECODER_K, MASKS_PER_RATE, UNIFORM_RATES
from c3.traces.models import KNOWN_MODELS, build_model, parse_max_memory, set_deterministic
from c3.util import hard_exit


def load_docs(path, tokenizer, max_tokens, limit=None):
    from c3.data.fineweb import tokenize_documents
    docs = tokenize_documents(path, tokenizer, max_tokens)
    return docs[:limit] if limit else docs


def build_config(a) -> CollectConfig:
    kw = dict(mask_rates=tuple(a.mask_rates), masks_per_rate=a.masks_per_rate,
              decoder_k=tuple(a.decoder_k), decoder_policies=tuple(a.decoder_policy),
              decoder_states=a.decoder_states, min_masked=a.min_masked,
              include_uniform=not a.no_uniform, include_decoder=not a.no_decoder,
              include_uniform_matched=not a.no_uniform_matched, topk=a.topk,
              batch_size=a.batch_size, seed=a.seed)
    if a.preset:
        kw.update(PRESETS[a.preset]["config"])
        kw.update(batch_size=a.batch_size, seed=a.seed)
    return CollectConfig(**kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--model", default=None,
                    help=f"{sorted(KNOWN_MODELS)}, 'local-110m' or 'toy' (default: the preset's)")
    ap.add_argument("--model-path", default=None,
                    help="local checkpoint dir, or final.pt for local-110m")
    ap.add_argument("--precision", default=None, choices=["4bit", "fp16", "fp32", "bf16"],
                    help="default 4bit, except that a preset requires it explicitly")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None, choices=["auto"],
                    help="split an fp16/fp32 model across GPUs, or offload to CPU")
    ap.add_argument("--max-memory", default=None, help="e.g. 0=10GiB,1=10GiB,cpu=60GiB")
    ap.add_argument("--docs", default=None, help="calibration_docs.jsonl (raw text)")
    ap.add_argument("--dataset", default=None,
                    help="dataset label for --docs rows (default fineweb; e.g. timesplit)")
    ap.add_argument("--task", default=None, choices=["gsm8k", "mbpp"],
                    help="teacher-forced task documents instead of raw text")
    ap.add_argument("--task-jsonl", default=None,
                    help="scripts/prepare_tasks.py output (default: the pinned hub revision)")
    ap.add_argument("--limit", type=int, default=None, help="first N documents only")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-doc-tokens", type=int, default=None,
                    help="exclude (and record) documents whose whole sequence, prompt included, "
                         "is longer than this; the C1 task runs use 1024")
    ap.add_argument("--mask-rates", type=float, nargs="+", default=list(UNIFORM_RATES))
    ap.add_argument("--masks-per-rate", type=int, default=MASKS_PER_RATE)
    ap.add_argument("--decoder-k", type=int, nargs="+", default=list(DECODER_K))
    ap.add_argument("--decoder-policy", nargs="+", default=["topk_conf"])
    ap.add_argument("--decoder-states", type=int, default=None,
                    help="record only this many states per trajectory, spread evenly "
                         "(default: every state, as plan v2 requires)")
    ap.add_argument("--min-masked", type=int, default=1)
    ap.add_argument("--no-uniform", action="store_true")
    ap.add_argument("--no-decoder", action="store_true")
    ap.add_argument("--no-uniform-matched", action="store_true")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="states per forward pass; the floor must be measured at this value")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--part-docs", type=int, default=50, help="documents per part file")
    ap.add_argument("--merge-as", choices=["parquet", "dataset", "none"], default="parquet",
                    help="how a complete single-shard run is merged: a parquet copy, a "
                         "zero-copy <out>.dataset.json manifest, or not at all")
    ap.add_argument("--on-nonfinite", choices=["raise", "skip"], default="raise",
                    help="skip: leave out (and record) a document whose logits overflow, "
                         "instead of stopping the shard")
    ap.add_argument("--repeat", type=int, default=1,
                    help=">1 re-runs the same documents to measure the non-determinism floor")
    ap.add_argument("--out", required=True,
                    help="parquet path, or the part-file stem when sharding/resuming")
    a = ap.parse_args()

    if a.preset:
        a.model = a.model or PRESETS[a.preset]["model"]
        if a.precision in (None, "4bit"):
            ap.error(f"--preset {a.preset} needs an explicit --precision other than 4bit: "
                     f"plan v2 runs the main 8B traces unquantised, 4-bit is an ablation")
    if a.model is None:
        ap.error("--model is required without --preset")
    a.precision = a.precision or "4bit"
    if (a.docs is None) == (a.task is None) and a.model != "toy":
        ap.error("give exactly one of --docs (raw text) or --task")
    sharded = a.num_shards > 1 or a.resume
    if sharded and a.repeat > 1:
        ap.error("--repeat is for the 20-document floor run; do not combine it with sharding")

    det = set_deterministic(a.seed)
    model = build_model(a.model, precision=a.precision, path=a.model_path, device=a.device,
                        device_map=a.device_map, max_memory=parse_max_memory(a.max_memory))
    dataset = a.task or ("toy" if a.model == "toy" else (a.dataset or "fineweb"))
    if a.model == "toy":
        rng = np.random.default_rng(0)
        docs = [Document(i, rng.integers(0, 60, a.max_tokens), dataset="toy")
                for i in range(a.limit or 20)]
    elif a.task:
        from c3.data.tasks import format_documents, load_task
        docs = format_documents(a.task, load_task(a.task, a.task_jsonl), model.tokenizer,
                                max_answer_tokens=a.max_tokens)
        docs = docs[:a.limit] if a.limit else docs
    else:
        # Each model is evaluated under its own tokeniser, so token counts per
        # document differ slightly across arms; the state fixed effects absorb it.
        docs = [Document(i, x, dataset=dataset)
                for i, x in load_docs(a.docs, model.tokenizer, a.max_tokens, a.limit)]
    too_long = []
    if a.max_doc_tokens:
        # The whole sequence (a task's prompt included) goes through every forward
        # pass, and its float32 logits are L x vocab x 4 bytes: MBPP test 482 is
        # 3,986 tokens (a 3,730-token prompt of test asserts), ~1.9 GiB at LLaDA's
        # vocabulary, which no 2080 Ti holds beside 9 GiB of weights (job 17746).
        too_long = [{"doc_id": int(d.doc_id), "tokens": int(d.ids.size)} for d in docs
                    if d.ids.size > a.max_doc_tokens]
        docs = [d for d in docs if d.ids.size <= a.max_doc_tokens]
        for x in too_long:
            print(f"[collect] EXCLUDED document {x['doc_id']}: {x['tokens']} tokens > "
                  f"--max-doc-tokens {a.max_doc_tokens}", flush=True)
    cfg = build_config(a)
    print(f"[collect] {len(docs)} documents, model={a.model}, precision={a.precision}, "
          f"preset={a.preset}, config={json.dumps(cfg.to_dict())}", flush=True)

    extra = {"preset": a.preset, "deterministic": det, "limit": a.limit,
             "source": a.task_jsonl or a.docs or a.task,
             "max_doc_tokens": a.max_doc_tokens, "excluded_too_long": too_long}
    outputs = []
    if sharded:
        stem = a.out[:-len(".parquet")] if a.out.endswith(".parquet") else a.out
        meta = collect_shard(model, docs, cfg, stem, a.shard, a.num_shards, resume=a.resume,
                             part_docs=a.part_docs, dataset=dataset, extra_meta=extra,
                             on_nonfinite=a.on_nonfinite)
        outputs.append(str(meta))
        if (a.num_shards == 1 and a.merge_as != "none"
                and json.loads(Path(meta).read_text()).get("complete")):
            from c3.traces.shards import DATASET_SUFFIX, merge
            outputs.append(str(merge(stem, stem + (".parquet" if a.merge_as == "parquet"
                                                   else DATASET_SUFFIX))))
    else:
        for rep in range(a.repeat):
            out = Path(a.out) if a.repeat == 1 else Path(a.out).with_name(
                Path(a.out).stem + f".rep{rep}" + Path(a.out).suffix)
            # A fresh seed per repeat would change the mask patterns; the floor is
            # about non-determinism at *identical* inputs, so the seed is held fixed.
            path = collect(model, docs, cfg, out, dataset=dataset)
            m = json.loads(path.with_suffix(".meta.json").read_text())
            m.update(extra)
            path.with_suffix(".meta.json").write_text(json.dumps(m, indent=2))
            outputs.append(str(path))
    print(json.dumps({"outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    # hard_exit: torch/transformers worker threads can abort at finalization too
    hard_exit(main())
