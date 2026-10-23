#!/usr/bin/env python3
"""T10: C2 on real text -- parallel-sequential disagreement and the (B) proxies,
with the disagreement proxy calibrated on the probe (plan v2 T10).

    python scripts/run_t10.py --model-path train110m-v3.pt --run train110m-v3 \\
        --corpus /home2/$USER/c3-runs/corpus/calibration_docs.jsonl \\
        --probe ft110m-v3-visible=ft110m-v3-visible.pt --probe ft110m-v3=ft110m-v3.pt

Real text: the first ``--n-docs`` fit-half documents (``--manifest`` value 0) in
doc_id order that reach 256 tokens, one uniform mask per rate 0.1, ..., 1.0 each.
At every state the 8 most confident positions are committed in parallel (one
pass) and one by one (8 passes); per k in {2, 4, 8} this gives the disagreement
rate, and at the same positions Delta and the cross-entropy (k = 1 as well).
Probe: for each ``--probe TAG=CKPT`` the same proxy on the states of the
confidence-rule trajectories behind ``<grid-dir>/grid_<TAG>.json``, per (rho, k)
cell, rank-correlated with the exact (A) per field.  The checkpoint and the spec
must be the ones the grid was run with (sha256 checked), and every state is
compared with ``steps_<TAG>.parquet``.  See ``c3.disagreement``.

Writes ``<out-dir>/t10_<model>_<precision>_<run>.json`` (per k and per k x mask
rate with document-bootstrap 95% intervals, contrasts between k, the probe cells
and Spearman, the config and its assumptions, provenance) and beside it
``.text.parquet`` (one row per state and rank) and ``.probe.parquet`` (one row
per probe state).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3 import N_BOOT
from c3.disagreement import B_KS, N_DOCS, T10_KS, T10_RATES, ZERO_A
from c3.provenance import collect as provenance, sha256_file
from c3.util import hard_exit

ASSUMPTIONS = [
    "Documents: the first n_docs fit-half documents in doc_id order that reach max_tokens "
    "tokens under the model's tokenizer (the C1 length filter); shorter ones are skipped and "
    "listed. Report-half lines are never parsed.",
    "States: c3.traces.masks.uniform_masks with min_masked = max(k) = 8, so every state can "
    "commit 8 positions; realised rates are recorded.",
    "Ranking on real text by the exact log-odds ell (as c3.traces.masks.topk_confidence; ties "
    "to the lower position). On the probe by the grid's own rule (float32 top-1 probability, "
    "ties to the lower field) so that the states and commit sets are the grid's.",
    "The sequential chain fills earlier ranks with the model's own sequential (greedy) tokens, "
    "not the true tokens. Every argmax is over the full vocabulary with [MASK] excluded (SUBS), "
    "on the probe too (not restricted to a field's values).",
    "Per document: the mean over its committed positions. On real text every state contributes "
    "exactly k, so this is also the mean over its states. On the probe a trajectory commits each "
    "field once, so it is disagreements / fields, the counterpart of the grid's A_per_field.",
    "(B) proxies are also reported at k = 1 (the most confident position); disagreement at "
    "k = 1 is reported only as a check (0 by construction).",
    "Probe calibration: confidence-rule cells only (the proxy's set is the k most confident "
    "positions); the k = 1 cells are left out of the Spearman (both quantities 0 by "
    "construction). Point Spearman against the grid JSON's A_per_field; its interval resamples "
    "instances within each rho, one resample shared by every k and both quantities, with (A) "
    f"per instance recomputed on the same states. |(A)| < {ZERO_A:g} nats is treated as 0 "
    "(exactly 0 analytically at rho = 0; the float residue would otherwise be ranked).",
    "No torch deterministic settings, as scripts/run_probe.py, so the probe passes are the "
    "grid's; probe batches and length groups are the grid's (--probe-batch-size 32).",
    "At rate 1.0 every document's input is the all-[MASK] sequence, so the parallel and "
    "sequential tokens (hence the disagreement) are identical for every document there; only "
    "Delta and CE vary. per_k therefore weights one shared state 1/10; per_k_rates_below_1 "
    "gives the same estimands over rates 0.1-0.9 (same resamples).",
]


def model_record(model, name: str, precision: str, path: str | None, tokenizer: str) -> dict:
    rec = {"name": name, "precision": precision, "tokenizer": tokenizer,
           "revision": getattr(model, "revision", None), "path": path}
    if path and Path(path).is_file():
        rec["checkpoint_sha256"] = sha256_file(path)
        rec["weights"] = getattr(model, "weights", None)
    return rec


def free(model) -> None:
    import gc

    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_real_text(a) -> tuple[dict, dict]:
    from c3.disagreement import load_fit_documents, run_text, summarise_text
    from c3.traces.models import build_model, parse_max_memory

    model = build_model(a.model, precision=a.precision, path=a.model_path, device=a.device,
                        device_map=a.device_map, max_memory=parse_max_memory(a.max_memory))
    tok_name = "gpt2" if a.model == "local-110m" else a.model
    docs, info = load_fit_documents(a.corpus, a.manifest, model.tokenizer, a.n_docs, a.max_tokens)
    print(f"[t10] {len(docs)} fit-half documents (doc_id {docs[0][0]}..{docs[-1][0]}; "
          f"{len(info['skipped_short'])} short ones skipped)", flush=True)
    counter = {"forward_passes": 0}
    t0 = time.time()
    cols = run_text(model, docs, rates=a.rates, depth=max(max(a.ks), max(a.b_ks)), seed=a.seed,
                    batch_size=a.batch_size, counter=counter,
                    log=lambda s: print(s, flush=True))
    seconds = round(time.time() - t0, 1)
    summary = summarise_text(cols, ks=a.ks, b_ks=a.b_ks, rates=a.rates, n_boot=a.n_boot,
                             seed=a.boot_seed)
    rec = {"model": model_record(model, a.model, a.precision, a.model_path, tok_name),
           "documents": {"doc_ids": [int(d) for d, _ in docs], **info},
           "forward_passes": counter["forward_passes"], "seconds": seconds, **summary}
    free(model)
    return rec, cols


def run_probe_calibration(a, tag: str, ckpt: str, spec, spec_sha: str) -> tuple[dict, list]:
    import pandas as pd

    from c3.disagreement import align_with_grid_steps, probe_disagreement, summarise_probe
    from c3.probe.grid import prepare
    from c3.probe.instances import eval_instances
    from c3.traces.models import build_model

    grid_path = Path(a.grid_dir) / f"grid_{tag}.json"
    grid = json.loads(grid_path.read_text())
    ck_sha = sha256_file(ckpt)
    want = grid["model"].get("checkpoint_sha256")
    if want != ck_sha:
        raise SystemExit(f"[t10] {tag}: {ckpt} (sha256 {ck_sha[:12]}) is not the checkpoint "
                         f"{grid_path} was run with ({str(want)[:12]})")
    if grid.get("spec_sha256") != spec_sha:
        raise SystemExit(f"[t10] {tag}: {a.probe_spec} is not the spec {grid_path} was run with")
    precision = grid["model"].get("precision", "fp32")
    model = build_model(grid["model"]["name"], precision=precision, path=ckpt, device=a.device)
    instances = eval_instances(spec, per_rho=a.probe_per_rho)
    if a.probe_limit:
        instances = instances[: a.probe_limit]
    prepared = prepare(spec, instances, model.tokenizer, grid["model"].get("tokenizer", "gpt2"))
    counter = {"forward_passes": 0}
    t0 = time.time()
    rows = []
    for k in a.probe_ks:
        rows += probe_disagreement(model, spec, prepared, k, batch_size=a.probe_batch_size,
                                   counter=counter, log=lambda s: print(s, flush=True))
    seconds = round(time.time() - t0, 1)
    summary = summarise_probe(rows, grid, n_boot=a.n_boot, seed=a.boot_seed, label=tag)
    steps = Path(a.grid_dir) / f"steps_{tag}.parquet"
    alignment = (align_with_grid_steps(rows, pd.read_parquet(steps)) if steps.exists()
                 else {"note": f"{steps} not found; states not compared"})
    rec = {"model": {"name": grid["model"]["name"], "precision": precision, "path": ckpt,
                     "checkpoint_sha256": ck_sha, "weights": getattr(model, "weights", None),
                     "revision": getattr(model, "revision", None)},
           "grid": str(grid_path), "grid_sha256": sha256_file(grid_path),
           "n_instances": len(prepared), "ks": a.probe_ks, "forward_passes": counter["forward_passes"],
           "seconds": seconds, "alignment": alignment, **summary}
    free(model)
    for r in rows:
        r["tag"] = tag
    return rec, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="local-110m")
    ap.add_argument("--model-path", default=None, help="final.pt for local-110m")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None, choices=["auto"])
    ap.add_argument("--max-memory", default=None, help="e.g. 0=9GiB,1=9GiB")
    ap.add_argument("--run", required=True, help="run label, e.g. train110m-v3")
    ap.add_argument("--out-dir", default="artifacts/c2")
    ap.add_argument("--corpus", default="runs/corpus/calibration_docs.jsonl")
    ap.add_argument("--manifest", default="artifacts/split_manifest.json")
    ap.add_argument("--n-docs", type=int, default=N_DOCS)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--rates", type=float, nargs="+", default=list(T10_RATES))
    ap.add_argument("--ks", type=int, nargs="+", default=list(T10_KS))
    ap.add_argument("--b-ks", type=int, nargs="+", default=list(B_KS))
    ap.add_argument("--seed", type=int, default=20260917, help="mask seed")
    ap.add_argument("--batch-size", type=int, default=32, help="states per forward pass")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--no-text", action="store_true", help="probe calibration only")
    ap.add_argument("--probe", action="append", default=[], metavar="TAG=CKPT",
                    help="a probe model and its checkpoint; grid_<TAG>.json must exist")
    ap.add_argument("--probe-spec", default="artifacts/probe/spec.json")
    ap.add_argument("--grid-dir", default="artifacts/probe/results")
    ap.add_argument("--probe-per-rho", type=int, default=200)
    ap.add_argument("--probe-limit", type=int, default=None, help="first N instances (testing)")
    ap.add_argument("--probe-ks", type=int, nargs="+", default=list(T10_KS))
    ap.add_argument("--probe-batch-size", type=int, default=32,
                    help="as the grid's run_probe.py --batch-size, so its states are reproduced")
    a = ap.parse_args()
    if a.no_text and not a.probe:
        ap.error("--no-text needs at least one --probe")
    if any(k < 2 for k in a.ks):
        ap.error("--ks are >= 2 (k = 1 is 0 by construction)")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = str(out / f"t10_{a.model}_{a.precision}_{a.run}")
    record = {"task": "T10", "run": a.run,
              "config": {k: v for k, v in vars(a).items() if k != "probe"},
              "assumptions": ASSUMPTIONS}
    record["config"]["probe"] = dict(p.split("=", 1) for p in a.probe)
    t0 = time.time()

    import pandas as pd

    if not a.no_text:
        record["config"]["corpus_sha256"] = sha256_file(a.corpus)
        record["config"]["manifest_sha256"] = sha256_file(a.manifest)
        text, cols = run_real_text(a)
        record["real_text"] = text
        pd.DataFrame(cols).to_parquet(stem + ".text.parquet", index=False)
    if a.probe:
        from c3.probe.instances import ProbeSpec
        spec = ProbeSpec.load(a.probe_spec)
        spec_sha = sha256_file(a.probe_spec)
        record["config"]["probe_spec_sha256"] = spec_sha
        record["probe_calibration"] = {}
        rows = []
        for item in a.probe:
            tag, ckpt = item.split("=", 1)
            rec, got = run_probe_calibration(a, tag, ckpt, spec, spec_sha)
            record["probe_calibration"][tag] = rec
            rows += got
        pd.DataFrame(rows).to_parquet(stem + ".probe.parquet", index=False)
    record["seconds"] = round(time.time() - t0, 1)
    record["provenance"] = provenance(with_gpu=a.device.startswith("cuda"))
    path = Path(stem + ".json")
    path.write_text(json.dumps(record, indent=1))

    if "real_text" in record:
        rt = record["real_text"]
        print(f"[t10] real text: {rt['n_docs']} docs x {rt['n_states'] // max(1, rt['n_docs'])} "
              f"states, {rt['forward_passes']} forward passes in {rt['seconds']} s", flush=True)
        for r in rt["per_k"]:
            parts = [f"{m} {r[m]['mean']:.4f} [{r[m]['ci95'][0]:.4f}, {r[m]['ci95'][1]:.4f}]"
                     for m in ("disagreement", "delta", "ce") if m in r]
            print(f"[t10]   k={r['k']}: " + "; ".join(parts), flush=True)
    for tag, rec in record.get("probe_calibration", {}).items():
        al, sp = rec["alignment"], rec["spearman"]
        print(f"[t10] probe {tag}: {al.get('steps_same_state')}/{al.get('steps_here')} states are "
              f"the grid's ({al.get('steps_bitwise')} bit for bit)"
              + (f"; Spearman {sp['rho_s']:.3f} [{sp['ci95'][0]:.3f}, {sp['ci95'][1]:.3f}] over "
                 f"{sp['n_cells']} cells" if sp else "; too few cells for a Spearman"), flush=True)
        if al.get("steps_same_state") is not None and al["steps_same_state"] != al["steps_here"]:
            print(f"[t10] WARNING: {tag}: {al['steps_here'] - al['steps_same_state']} states differ "
                  f"from the grid's", flush=True)
    print(f"[t10] -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
