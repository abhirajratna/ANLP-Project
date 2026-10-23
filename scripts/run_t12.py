#!/usr/bin/env python3
"""T12 driver: free-running LLaDA-8B-Instruct on GSM8K / MBPP with every policy of a grid.

    # sanity check (plan, Sun 25): ~200 GSM8K problems, top-k and the Fast-dLLM threshold
    python scripts/run_t12.py --model llada-8b-instruct --precision bf16 --task gsm8k \\
        --task-jsonl runs/corpus/tasks/gsm8k_test.jsonl --manifest artifacts/split_manifest_gsm8k.json \\
        --half fit --limit 200 --configs grid.json --out-dir runs/t12/sanity

    # the sweep (Sun 25 - Wed 28): the report halves, C3a with its frozen calibrator
    python scripts/run_t12.py ... --half report --calibrator artifacts/c3/llada-8b-instruct_fp16_c3a-gsm8k.json

Each problem is decoded by ``c3.decode.generate.generate``: the chat-formatted question
(as the traces format it), 256 masks, blocks of 32, greedy.  The grid is
``{"configs": [{"policy": "topk_conf", "k": 2}, {"policy": "c3a", "alpha": 0.1}, ...]}``
with the policies of ``c3.decode.policies.POLICIES``; C3a needs ``--calibrator``.

Output, per shard, in ``--out-dir``:

* ``t12.shard<i>of<n>.jsonl``: one record per (configuration, problem) -- the completion
  text and token ids, whether it stopped, forward passes, tokens per step, wall-clock;
* ``t12.shard<i>of<n>.meta.json``: model, precision, revision, grid, calibrator, task
  file and split digests, provenance.

Accuracy is not computed here: the completions are scored afterwards (GSM8K exact
match, MBPP pass@1) by the T12 scoring module.  Resumable: a record is appended (and
fsync'd) as soon as its problem is decoded, and a restart skips every (configuration,
problem) already recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.decode.generate import generate, stop_ids, task_prompt
from c3.decode.policies import CALIBRATED, POLICIES
from c3.provenance import collect as provenance, sha256_file


def config_key(cfg: dict) -> str:
    knobs = ",".join(f"{k}={cfg[k]}" for k in sorted(cfg) if k != "policy")
    return f"{cfg['policy']}[{knobs}]"


def load_configs(path) -> list[dict]:
    d = json.loads(Path(path).read_text())
    cfgs = d["configs"] if isinstance(d, dict) else d
    keys = [config_key(c) for c in cfgs]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{path}: duplicate configurations")
    unknown = sorted({c["policy"] for c in cfgs} - set(POLICIES))
    if unknown:
        raise SystemExit(f"{path}: unknown policies {unknown}; known: {sorted(POLICIES)}")
    return cfgs


def build_policy(cfg: dict, calibrator=None, log_freq=None):
    name = cfg["policy"]
    kw = {k: v for k, v in cfg.items() if k != "policy"}
    if name in CALIBRATED:
        if calibrator is None:
            raise SystemExit(f"{name} needs --calibrator")
        kw["calibrator"] = calibrator
    if name == "pc_sampler" and log_freq is not None:
        kw["log_freq"] = log_freq
    return POLICIES[name](**kw)


def select_records(records, manifest, half, limit, shard, num_shards):
    """The task records of ``half`` (fit = 0, report = 1, all), first ``limit``, this shard."""
    if half != "all":
        split = {int(k): int(v) for k, v in json.loads(Path(manifest).read_text()).items()}
        want = 0 if half == "fit" else 1
        records = [r for r in records if split[int(r["doc_id"])] == want]
    records = sorted(records, key=lambda r: int(r["doc_id"]))
    if limit:
        records = records[:limit]
    return [r for r in records if int(r["doc_id"]) % num_shards == shard]


def done_pairs(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                      # a torn last line from a killed job
            done.add((r["config"], int(r["doc_id"])))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llada-8b-instruct")
    ap.add_argument("--model-path", default=None, help="a local copy of the pinned checkpoint")
    ap.add_argument("--precision", default="bf16", help="bf16 on an A100; fp16 on Ada's 2080 Ti")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None, choices=["auto"])
    ap.add_argument("--max-memory", default=None, help="e.g. 0=9GiB,1=9GiB (Ada, fp16)")
    ap.add_argument("--task", required=True, choices=["gsm8k", "mbpp", "humaneval"])
    ap.add_argument("--task-jsonl", default=None, help="the task file (default: the pinned hub revision)")
    ap.add_argument("--manifest", default=None, help="the committed task split, for --half fit/report")
    ap.add_argument("--half", choices=["fit", "report", "all"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N problems of the half (0 = all)")
    ap.add_argument("--configs", required=True, help="policy grid JSON")
    ap.add_argument("--calibrator", default=None, help="frozen C3a calibrator (artifacts/c3/...)")
    ap.add_argument("--unigram-docs", default=None,
                    help="background text for PC-Sampler's p_D (reserve docs; never scored)")
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.half != "all" and not a.manifest:
        raise SystemExit("--half fit/report needs --manifest")

    from c3.data.tasks import load_task
    from c3.traces.models import build_model, parse_max_memory, set_deterministic

    det = set_deterministic(0)
    records = select_records(load_task(a.task, a.task_jsonl), a.manifest, a.half, a.limit,
                             a.shard, a.num_shards)
    configs = load_configs(a.configs)
    calibrator = log_freq = None
    if a.calibrator:
        from c3.calibrators import ContextLogistic
        calibrator = ContextLogistic.load(a.calibrator)
    t_load = time.perf_counter()
    model = build_model(a.model, a.precision, path=a.model_path, device=a.device,
                        device_map=a.device_map, max_memory=parse_max_memory(a.max_memory))
    t_load = time.perf_counter() - t_load
    tok = model.tokenizer
    if any(c["policy"] == "pc_sampler" for c in configs) and a.unigram_docs:
        from c3.data.fineweb import tokenize_documents
        from c3.decode.policies import unigram_log_freq
        bg = tokenize_documents(a.unigram_docs, tok, 256)
        log_freq = unigram_log_freq([ids for _, ids in bg], int(model.vocab_size))
    stops = stop_ids(tok)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"t12.shard{a.shard}of{a.num_shards}"
    rec_path = out / f"{stem}.jsonl"
    done = done_pairs(rec_path)
    meta = {"label": a.label, "model": a.model, "precision": a.precision,
            "model_revision": getattr(model, "revision", None), "model_load_seconds": t_load,
            "task": a.task, "task_jsonl": a.task_jsonl,
            "task_jsonl_sha256": sha256_file(a.task_jsonl) if a.task_jsonl else None,
            "manifest": a.manifest, "manifest_sha256": sha256_file(a.manifest) if a.manifest else None,
            "half": a.half, "limit": a.limit, "shard": a.shard, "num_shards": a.num_shards,
            "doc_ids": [int(r["doc_id"]) for r in records],
            "gen_length": a.gen_length, "block_length": a.block_length, "decoding": "greedy",
            "stop_ids": sorted(stops), "configs": configs, "configs_path": a.configs,
            "configs_sha256": sha256_file(a.configs), "calibrator": a.calibrator,
            "calibrator_sha256": sha256_file(a.calibrator) if a.calibrator else None,
            "pc_sampler_variant": "full" if log_freq is not None else "position-only",
            "deterministic": det, "provenance": provenance(with_gpu=True)}
    (out / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    prompts = {int(r["doc_id"]): task_prompt(a.task, r, tok) for r in records}
    print(f"[t12] {a.task} {a.half}: {len(records)} problems, {len(configs)} configurations, "
          f"{len(done)} records already done; model loaded in {t_load:.0f} s", flush=True)

    t_all = time.time()
    with rec_path.open("a", encoding="utf-8") as fh:
        for cfg in configs:
            key = config_key(cfg)
            policy = build_policy(cfg, calibrator, log_freq)
            t0, n_new = time.time(), 0
            for r in records:
                d = int(r["doc_id"])
                if (key, d) in done:
                    continue
                g = generate(model, prompts[d], policy, stops, a.gen_length, a.block_length)
                text = tok.decode(g["completion_ids"], skip_special_tokens=True)
                fh.write(json.dumps({"config": key, **cfg, "task": a.task, "doc_id": d,
                                     "completion": text, **g}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                n_new += 1
            if n_new:
                print(f"[t12] {key}: {n_new} problems in {time.time() - t0:.0f} s", flush=True)
    print(f"[t12] done in {time.time() - t_all:.0f} s -> {rec_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
