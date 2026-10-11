#!/usr/bin/env python3
"""Run the T9 grid (rho x k x selection rule) on one model and write (A), (B) and their sum.

    python scripts/run_probe.py --model local-110m --model-path final.pt --tag untuned-v2

Writes ``<out-dir>/grid_<tag>.json`` (per cell: mean (A), (B), (A)+(B), (A)-(B)
per instance and per field with document-bootstrap 95% intervals; the crossover
k per rule and rho; the identity check against the directly computed KL;
provenance and the checkpoint's sha256) and ``steps_<tag>.parquet`` (one row per
instance and decoding step).  Exits non-zero if the identity fails by more than
1e-6 nats anywhere it was checked.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3 import N_BOOT
from c3.probe import KS, RULES
from c3.probe.grid import prepare, run_grid, summarise
from c3.probe.instances import ProbeSpec, eval_instances
from c3.provenance import collect as provenance, sha256_file
from c3.util import hard_exit

TOKENIZER_OF = {"local-110m": "gpt2", "llada-8b-base": "llada-8b-base",
                "llada-8b-instruct": "llada-8b-base", "dream-7b-base": "dream-7b-base",
                "dream-7b-instruct": "dream-7b-base"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="artifacts/probe/spec.json")
    ap.add_argument("--model", default="local-110m")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", required=True, help="e.g. untuned-v2, ft-v2")
    ap.add_argument("--out-dir", default="artifacts/probe/results")
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    ap.add_argument("--per-rho", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="first N instances only (testing)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--summarise-only", action="store_true",
                    help="recompute the summary of an existing grid_<tag>.json from its "
                         "steps_<tag>.parquet (no model is loaded)")
    a = ap.parse_args()
    if a.summarise_only:
        return resummarise(a)

    from c3.traces.models import build_model
    spec = ProbeSpec.load(a.spec)
    model = build_model(a.model, precision=a.precision, path=a.model_path, device=a.device)
    tok_name = TOKENIZER_OF[a.model]
    instances = eval_instances(spec, per_rho=a.per_rho)
    if a.limit:
        instances = instances[: a.limit]
    prepared = prepare(spec, instances, model.tokenizer, tok_name)
    t0 = time.time()
    rows = run_grid(model, spec, prepared, ks=a.ks, rules=a.rules, seed=a.seed,
                    batch_size=a.batch_size)
    summary = summarise(rows, n_boot=a.n_boot, seed=a.seed)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_parquet(out / f"steps_{a.tag}.parquet", index=False)
    model_id = {"name": a.model, "precision": a.precision, "tokenizer": tok_name,
                "revision": getattr(model, "revision", None)}
    if a.model_path and Path(a.model_path).is_file():
        model_id["checkpoint_sha256"] = sha256_file(a.model_path)
        model_id["weights"] = getattr(model, "weights", None)
    record = {"tag": a.tag, "model": model_id, "spec": a.spec, "spec_sha256": sha256_file(a.spec),
              "n_instances": len(prepared), "ks": a.ks, "rules": a.rules, "seed": a.seed,
              "n_boot": a.n_boot, "seconds": round(time.time() - t0, 1), **summary,
              "provenance": provenance(with_gpu=a.device.startswith("cuda"))}
    (out / f"grid_{a.tag}.json").write_text(json.dumps(record, indent=1))
    ident = summary["identity"]
    print(f"[probe] {a.tag}: {len(rows)} steps in {record['seconds']} s; identity checked on "
          f"{ident['steps_checked']}/{ident['steps_total']} steps, max |A+B-KL| = "
          f"{ident['max_abs_err_nats']:.2e} nats -> {out}/grid_{a.tag}.json")
    for key, v in summary["crossover"].items():
        print(f"[probe]   crossover {key}: A > B first at k = {v['first_k_A_exceeds_B']} "
              f"(interval excludes 0 at k = {v['first_k_A_exceeds_B_ci']})")
    return 0 if not ident["max_abs_err_nats"] > 1e-6 else 1


def resummarise(a) -> int:
    import pandas as pd
    out = Path(a.out_dir)
    path = out / f"grid_{a.tag}.json"
    record = json.loads(path.read_text())
    rows = pd.read_parquet(out / f"steps_{a.tag}.parquet").to_dict("records")
    summary = summarise(rows, n_boot=record["n_boot"], seed=record["seed"])
    record.update(summary)
    record.setdefault("resummarised", []).append(provenance())
    path.write_text(json.dumps(record, indent=1))
    print(f"[probe] {a.tag}: summary recomputed from {len(rows)} steps -> {path}")
    return 0


if __name__ == "__main__":
    hard_exit(main())
