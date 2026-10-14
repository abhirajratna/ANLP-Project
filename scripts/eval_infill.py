#!/usr/bin/env python3
"""FineWeb infilling with every commit policy of a knob grid (deviations.md item 10).

    python scripts/eval_infill.py --model local-110m --model-path final.pt \\
        --docs runs/corpus/calibration_docs.jsonl --manifest artifacts/split_manifest.json \\
        --half dev-eval --configs prereg/c3set_policies_dev.json \\
        --calibrator cal/tok.json --set-calibrator cal/set.json \\
        --unigram-docs runs/corpus/reserve_docs.jsonl --shard 0 --num-shards 4 --out-dir runs/infill/dev

Positions 64..191 of each 256-token document are masked and decoded greedily by
each configuration (``c3.infill``).  Output, per shard, in ``--out-dir``:

* ``infill.shard<i>of<n>.jsonl``: one record per (configuration, document) --
  accuracy, forward passes, tokens/step, per-step all-right rate, seconds, and a
  digest of the infilled tokens (two runs agree iff the digests do);
* ``steps/<config>.shard<i>of<n>.b<batch>.parquet`` with ``--record-steps``: the
  on-policy states (every masked position at every step), for refitting the
  calibrators on free-running states (``analyze_setcal.py --onpolicy``);
* ``infill.shard<i>of<n>.meta.json``: model, grid, calibrator and corpus digests.

Resumable: a batch's records are appended (and fsync'd) only after it completes,
and a restart skips every (configuration, document) already recorded.  Batches
are fixed chunks of the shard's sorted documents, so a resumed run decodes the
same batches.  The report half of the frozen split needs C3_REPORT_HALF_OK=1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3.infill import (HALVES, SPAN, StepLog, build_policy, config_id, config_key, infill_batch,
                       load_configs, select_docs, summarise_tokens)
from c3.provenance import collect as provenance, sha256_file
from c3.util import require_report_ok


def load_docs(a, wanted: set[int], model):
    if a.model == "toy":
        rng = np.random.default_rng(0)
        pool = {i: rng.integers(0, 60, a.max_tokens) for i in range(max(wanted) + 1)}
        return [(d, pool[d]) for d in sorted(wanted)]
    from c3.data.fineweb import tokenize_documents

    return [(int(d), ids) for d, ids in tokenize_documents(a.docs, model.tokenizer, a.max_tokens)
            if int(d) in wanted]


def done_pairs(path: Path) -> set[tuple[str, int]]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                      # a torn last line from a killed job
            done.add((r["config"], int(r["doc_id"])))
    return done


def gpt2_scorer(device: str):
    """Mean NLL (nats/token) of the span under GPT-2 small, given the whole document."""
    import torch
    from transformers import AutoModelForCausalLM

    lm = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

    def score(seqs: np.ndarray, span) -> np.ndarray:
        a, b = span
        with torch.no_grad():
            x = torch.as_tensor(seqs, dtype=torch.long, device=device)
            logp = torch.log_softmax(lm(x).logits[:, a - 1:b - 1].float(), dim=-1)
            nll = -logp.gather(-1, x[:, a:b, None])[..., 0]
        return nll.mean(dim=1).cpu().numpy()

    return score


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="local-110m", help="local-110m or toy")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--docs", default=None, help="calibration_docs.jsonl")
    ap.add_argument("--manifest", required=True, help="the committed split manifest")
    ap.add_argument("--half", choices=HALVES, required=True)
    ap.add_argument("--configs", required=True, help="knob grid JSON (c3.infill.load_configs)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these policies of the grid (names, e.g. topk_conf c3set)")
    ap.add_argument("--calibrator", default=None, help="token calibrator JSON (tok.json)")
    ap.add_argument("--set-calibrator", default=None, help="set calibrator JSON (set.json)")
    ap.add_argument("--unigram-docs", default=None,
                    help="background text for PC-Sampler's p_D (reserve docs; never scored)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--span", type=int, nargs=2, default=list(SPAN))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N documents of the half (0 = all)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--record-steps", action="store_true")
    ap.add_argument("--steps-dir", default=None, help="where step logs go (default <out-dir>/steps)")
    ap.add_argument("--ppl", action="store_true", help="also score the infilled span under GPT-2")
    ap.add_argument("--label", default="")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    require_report_ok(a.manifest, reads_report=a.half == "report")
    from c3.traces.models import build_model, set_deterministic

    det = set_deterministic(0)
    split = json.loads(Path(a.manifest).read_text())
    wanted = sorted(select_docs(split, a.half))
    if a.limit:
        wanted = wanted[:a.limit]
    wanted = {d for d in wanted if d % a.num_shards == a.shard}
    model = build_model(a.model, a.precision, path=a.model_path, device=a.device)
    docs = load_docs(a, wanted, model)
    missing = len(wanted) - len(docs)
    span = tuple(a.span)

    configs = load_configs(a.configs)
    if a.only:
        configs = [c for c in configs if c["policy"] in set(a.only)]
    tok = setcal = log_freq = None
    if a.calibrator:
        from c3.calibrators import ContextLogistic
        tok = ContextLogistic.load(a.calibrator)
    if a.set_calibrator:
        from c3.setcal import SetCalibrator
        setcal = SetCalibrator.load(a.set_calibrator)
    if any(c["policy"] == "pc_sampler" for c in configs) and a.unigram_docs and a.model != "toy":
        from c3.data.fineweb import tokenize_documents
        from c3.decode.policies import unigram_log_freq
        bg = tokenize_documents(a.unigram_docs, model.tokenizer, a.max_tokens)
        log_freq = unigram_log_freq([ids for _, ids in bg], int(model.vocab_size))
    scorer = gpt2_scorer(a.device) if a.ppl and a.model != "toy" else None

    out = Path(a.out_dir)
    steps_dir = Path(a.steps_dir) if a.steps_dir else out / "steps"
    out.mkdir(parents=True, exist_ok=True)
    if a.record_steps:
        steps_dir.mkdir(parents=True, exist_ok=True)
    stem = f"infill.shard{a.shard}of{a.num_shards}"
    rec_path = out / f"{stem}.jsonl"
    done = done_pairs(rec_path)
    meta = {"label": a.label, "half": a.half, "span": list(span), "shard": a.shard,
            "num_shards": a.num_shards, "n_docs": len(docs), "docs_missing_or_short": missing,
            "doc_ids": [d for d, _ in docs], "model": a.model, "precision": a.precision,
            "model_revision": getattr(model, "revision", None),
            "model_sha256": sha256_file(a.model_path) if a.model_path else None,
            "manifest": a.manifest, "manifest_sha256": sha256_file(a.manifest),
            "docs_sha256": sha256_file(a.docs) if a.docs else None,
            "configs": configs, "configs_path": a.configs, "configs_sha256": sha256_file(a.configs),
            "calibrator": a.calibrator, "calibrator_sha256": sha256_file(a.calibrator) if a.calibrator else None,
            "set_calibrator": a.set_calibrator,
            "set_calibrator_sha256": sha256_file(a.set_calibrator) if a.set_calibrator else None,
            "pc_sampler_variant": "full" if log_freq is not None else "position-only",
            "batch_size": a.batch_size, "deterministic": det, "provenance": provenance()}
    (out / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2, default=str))

    bs = 1 if hasattr(model, "set_reference") else a.batch_size
    batches = [docs[i:i + bs] for i in range(0, len(docs), bs)]
    ref_nll = {}
    if scorer is not None:
        for batch in batches:
            seqs = np.stack([ids for _, ids in batch])
            ref_nll.update(zip([d for d, _ in batch], scorer(seqs, span).tolist()))
    print(f"[infill] {len(docs)} documents ({a.half}, shard {a.shard}/{a.num_shards}), "
          f"{len(configs)} configurations, batch {bs}, {len(done)} records already done", flush=True)
    t_all = time.time()
    with rec_path.open("a") as fh:
        for cfg in configs:
            key = config_key(cfg)
            policy = build_policy(cfg, tok, setcal, log_freq)
            t0, n_new = time.time(), 0
            for bi, batch in enumerate(batches):
                if all((key, d) in done for d, _ in batch):
                    continue
                if hasattr(model, "set_reference"):
                    model.set_reference(batch[0][1])
                log = StepLog.empty() if a.record_steps else None
                rows = infill_batch(model, batch, policy, span=span, log=log,
                                    state_base=config_id(cfg) * 10_000_000)
                if scorer is not None:
                    seqs = np.stack([ids for _, ids in batch])
                    seqs[:, span[0]:span[1]] = np.stack([r["tokens"] for r in rows])
                    for r, v in zip(rows, scorer(seqs, span).tolist()):
                        r["gpt2_nll"], r["gpt2_nll_ref"] = float(v), float(ref_nll[r["doc_id"]])
                if log is not None:
                    df = log.frame()
                    df["arm"] = key
                    p = steps_dir / f"{config_id(cfg):08x}.{stem}.b{bi:04d}.parquet"
                    tmp = p.with_suffix(".tmp")
                    df.to_parquet(tmp, index=False)
                    os.replace(tmp, p)
                for r in summarise_tokens(rows):
                    fh.write(json.dumps({"config": key, **cfg, **r}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                n_new += len(rows)
            if n_new:
                print(f"[infill] {key}: {n_new} documents in {time.time() - t0:.0f} s", flush=True)
    print(f"[infill] done in {time.time() - t_all:.0f} s -> {rec_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
