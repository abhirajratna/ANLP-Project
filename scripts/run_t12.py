#!/usr/bin/env python3
"""T12 driver: free-running LLaDA-8B-Instruct on GSM8K / MBPP with every policy of a grid.

    # sanity check (plan, Sun 25): fit-half GSM8K at Fast-dLLM's settings (lm-eval 5-shot)
    python scripts/run_t12.py --task gsm8k --task-jsonl runs/corpus/tasks/gsm8k_test.jsonl \\
        --manifest artifacts/split_manifest_gsm8k.json --half fit --limit 200 \\
        --prompt lmeval --fewshot 5 --configs grid_sanity.json --out-dir runs/t12/sanity

    # the sweep (Sun 25 - Thu 29): the report halves, zero-shot chat, C3a with its calibrator
    python scripts/run_t12.py ... --half report --prompt chat \\
        --calibrator artifacts/c3/llada-8b-instruct_fp16_c3a-gsm8k.json

Each problem is decoded by ``c3.decode.generate.generate``: the prompt, 256 masks, blocks
of 32, greedy.  The grid is ``{"configs": [{"policy": "topk_conf", "k": 2}, ...]}`` with
the policies of ``c3.decode.policies.POLICIES``; C3a needs ``--calibrator``.  fp16 is
the default: the C3a calibrator is fitted on fp16 traces, and Ada (the overflow) has no
bf16; an A100 runs fp16 as fast.

Output, per shard, in ``--out-dir``:

* ``t12.shard<i>of<n>.jsonl``: one record per (problem, configuration) -- the completion
  text, all 256 generated ids and the first stop index, forward passes, tokens per step,
  wall-clock, an ``error`` if the problem failed, C3a's per-step log, and the run id,
  precision and git commit;
* ``t12.shard<i>of<n>.meta.json``: the run's identity (model, precision, revision, prompt
  format, lengths, task / split / grid / calibrator digests, commit) and every start.

Problems are the outer loop, so stopping early leaves every configuration on the same
problems.  Resumable: each record is appended and fsync'd as soon as it is decoded, a
torn last line is cut off on restart, and a restart skips every (configuration, problem)
already recorded -- but only into the same run: if the meta's identity differs (or the
git commit, unless ``--allow-mixed-commits``) the driver refuses.  Accuracy is computed
afterwards by the T12 scoring module (``scripts/score_t12.py``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.decode.generate import (LMEVAL, PROMPTS, cut_at_strings, generate, lmeval_contexts,
                                lmeval_prompt, load_fewshot_pool, stop_ids, task_prompt)
from c3.decode.policies import CALIBRATED, POLICIES
from c3.provenance import collect as provenance, sha256_file

# Meta fields that define a run: a resume must match them all.
IDENTITY = ("model", "model_revision", "precision", "task", "task_jsonl_sha256",
            "manifest_sha256", "half", "gen_length", "block_length", "decoding", "prompt",
            "configs_sha256", "calibrator_sha256", "c3a_clamp", "pc_sampler_variant",
            "shard", "num_shards")


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


def build_policy(cfg: dict, calibrator=None, log_freq=None, clamp: bool = False):
    name = cfg["policy"]
    kw = {k: v for k, v in cfg.items() if k != "policy"}
    if name in CALIBRATED:
        if calibrator is None:
            raise SystemExit(f"{name} needs --calibrator")
        kw["calibrator"] = calibrator
        kw["clamp"] = clamp
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


def repair_tail(path: Path) -> None:
    """Cut a torn last line (a killed write) so the next record starts on its own line."""
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        path.write_bytes(data[:data.rfind(b"\n") + 1])


def done_pairs(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["config"], int(r["doc_id"])))
    return done


def open_run(meta_path: Path, meta: dict, allow_mixed_commits: bool) -> dict:
    """The run's meta: new, or the existing one if ``meta`` is the same run (then this
    start is appended to its ``starts``).  Refuses to resume into a different run."""
    start = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "git_commit": meta["provenance"].get("git_commit"),
             "git_dirty": meta["provenance"].get("git_dirty")}
    if not meta_path.exists():
        meta = {**meta, "run_id": uuid.uuid4().hex[:12], "starts": [start]}
    else:
        old = json.loads(meta_path.read_text())
        diff = {k: (old.get(k), meta.get(k)) for k in IDENTITY if old.get(k) != meta.get(k)}
        commit = old["starts"][0]["git_commit"]
        if commit != start["git_commit"] and not allow_mixed_commits:
            diff["git_commit"] = (commit, start["git_commit"])
        if diff:
            raise SystemExit(f"refusing to resume {meta_path.parent}: not the same run "
                             f"(old, new): {diff}; use a new --out-dir"
                             + ("" if "git_commit" not in diff else
                                " or --allow-mixed-commits"))
        meta = {**old, "starts": old["starts"] + [start]}
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return meta


def run_sweep(model, tok, records, prompts, configs, out: Path, stem: str, meta: dict,
              calibrator=None, log_freq=None, clamp: bool = False, until=None,
              allow_mixed_commits: bool = False) -> Path:
    """Decode every (problem, configuration) not yet recorded; problems are the outer loop."""
    out.mkdir(parents=True, exist_ok=True)
    meta = open_run(out / f"{stem}.meta.json", meta, allow_mixed_commits)
    stamp = {"run_id": meta["run_id"], "precision": meta["precision"],
             "git_commit": meta["starts"][-1]["git_commit"]}
    rec_path = out / f"{stem}.jsonl"
    repair_tail(rec_path)
    done = done_pairs(rec_path)
    policies = [(config_key(c), c, build_policy(c, calibrator, log_freq, clamp)) for c in configs]
    stops = stop_ids(tok)
    print(f"[t12] {meta['task']} {meta['half']}: {len(records)} problems, {len(configs)} "
          f"configurations, {len(done)} records already done (run {meta['run_id']})", flush=True)
    t_all, n_new = time.time(), 0
    with rec_path.open("a", encoding="utf-8") as fh:
        for r in records:
            d = int(r["doc_id"])
            for key, cfg, policy in policies:
                if (key, d) in done:
                    continue
                if hasattr(policy, "log"):
                    policy.log = []
                rec = {"config": key, **cfg, "task": meta["task"], "doc_id": d, **stamp}
                try:
                    g = generate(model, prompts[d], policy, stops, meta["gen_length"],
                                 meta["block_length"])
                except (FloatingPointError, MemoryError, RuntimeError) as exc:
                    if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
                        raise
                    _free_cuda()
                    rec.update({"completion": None, "error": f"{type(exc).__name__}: {exc}"})
                else:
                    text = tok.decode(g["completion_ids"], skip_special_tokens=True)
                    rec.update({"completion": cut_at_strings(text, until), **g, "error": None})
                if getattr(policy, "log", None) is not None:
                    rec["c3a_steps"] = policy.log
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                n_new += 1
    print(f"[t12] {n_new} new records in {time.time() - t_all:.0f} s -> {rec_path}", flush=True)
    return rec_path


def _free_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llada-8b-instruct")
    ap.add_argument("--model-path", default=None, help="a local copy of the pinned checkpoint")
    ap.add_argument("--precision", default="fp16", help="fp16 (default; Ada and the calibrator's "
                                                        "traces) or bf16 (A100 only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None, choices=["auto"])
    ap.add_argument("--max-memory", default=None, help="e.g. 0=9GiB,1=9GiB (Ada, fp16)")
    ap.add_argument("--task", required=True, choices=["gsm8k", "mbpp", "humaneval"])
    ap.add_argument("--task-jsonl", default=None, help="the task file (default: the pinned hub revision)")
    ap.add_argument("--manifest", default=None, help="the committed task split, for --half fit/report")
    ap.add_argument("--half", choices=["fit", "report", "all"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N problems of the half (0 = all)")
    ap.add_argument("--prompt", choices=PROMPTS, default="chat",
                    help="chat: the traces' zero-shot chat prompt (the sweep); lmeval: lm-eval's "
                         "few-shot context in one chat turn (the sanity check)")
    ap.add_argument("--fewshot", type=int, default=5, help="examples for --prompt lmeval")
    ap.add_argument("--fewshot-jsonl", default=None,
                    help="the few-shot pool (GSM8K train split; default: the pinned hub revision)")
    ap.add_argument("--configs", required=True, help="policy grid JSON")
    ap.add_argument("--calibrator", default=None, help="frozen C3a calibrator (artifacts/c3/...)")
    ap.add_argument("--clamp", action="store_true",
                    help="C3a: clamp state features to the calibrator's fit-half range")
    ap.add_argument("--unigram-docs", default=None,
                    help="background text for PC-Sampler's p_D (reserve docs; never scored)")
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--allow-mixed-commits", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.half != "all" and not a.manifest:
        raise SystemExit("--half fit/report needs --manifest")
    if a.prompt == "lmeval" and a.task not in LMEVAL:
        raise SystemExit(f"--prompt lmeval is defined for {sorted(LMEVAL)} only")

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

    until = None
    if a.prompt == "lmeval":
        ctx = lmeval_contexts(a.task, records, load_fewshot_pool(a.task, a.fewshot_jsonl), a.fewshot)
        prompts = {d: lmeval_prompt(c, tok) for d, c in ctx.items()}
        until = list(LMEVAL[a.task]["until"])
        prompt_id = {"format": "lmeval", "fewshot": a.fewshot, "seed": LMEVAL[a.task]["seed"],
                     "until": until,
                     "pool_sha256": sha256_file(a.fewshot_jsonl) if a.fewshot_jsonl else "hub"}
    else:
        prompts = {int(r["doc_id"]): task_prompt(a.task, r, tok) for r in records}
        prompt_id = {"format": "chat", "fewshot": 0}

    meta = {"label": a.label, "model": a.model, "precision": a.precision,
            "model_revision": getattr(model, "revision", None), "model_load_seconds": t_load,
            "task": a.task, "task_jsonl": a.task_jsonl,
            "task_jsonl_sha256": sha256_file(a.task_jsonl) if a.task_jsonl else None,
            "manifest": a.manifest, "manifest_sha256": sha256_file(a.manifest) if a.manifest else None,
            "half": a.half, "limit": a.limit, "shard": a.shard, "num_shards": a.num_shards,
            "doc_ids": [int(r["doc_id"]) for r in records],
            "gen_length": a.gen_length, "block_length": a.block_length, "decoding": "greedy",
            "prompt": prompt_id, "stop_ids": sorted(stop_ids(tok)),
            "configs": configs, "configs_path": a.configs, "configs_sha256": sha256_file(a.configs),
            "calibrator": a.calibrator,
            "calibrator_sha256": sha256_file(a.calibrator) if a.calibrator else None,
            "c3a_clamp": bool(a.clamp),
            "pc_sampler_variant": "full" if log_freq is not None else "position-only",
            "deterministic": det, "provenance": provenance(with_gpu=True)}
    run_sweep(model, tok, records, prompts, configs, Path(a.out_dir),
              f"t12.shard{a.shard}of{a.num_shards}", meta, calibrator, log_freq, a.clamp,
              until, a.allow_mixed_commits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
