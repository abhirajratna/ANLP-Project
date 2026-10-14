#!/usr/bin/env python3
"""C: set-level C3 on the JSON probe, where the joint truth is exact (deviations.md item 12).

    python scripts/probe_c3set.py --model local-110m --model-path ft110m-v4/final.pt \\
        --tag ft110m-v4 --out-dir artifacts/probe/results

1. Calibration: 2,000 new instances (label ``probe-c3calib``, disjoint from the
   training, validation and evaluation sets) are decoded by confidence top-k
   (k = 1, 2, 4, teacher-forced); every masked field of every state is a row.  The
   token calibrator and the set calibrator are fitted on them exactly as on traces
   (``c3.setcal``: out-of-fold c_hat, prefix-set table, ``SetCalibrator``).
2. Evaluation: the 1,000 evaluation instances under every rule -- random and
   confidence top-k, Fast-dLLM threshold, EB-Sampler, C3a, C3-indep, C3-set -- with
   the exact (A), (B), the exact joint correctness J of each committed set, and the
   calibrated rules' P_ind / P_set (``c3.probe.grid.run_trajectories``).
3. Summary: the T9 summary (identity check included) and every rule at matched
   fields per step (``summarise_matched``): A, B, error per field, mean J, and the
   calibration error |P - J| of the set model against independence.

Gate G6 (exit 1 if it fails): |A + B - KL| <= 1e-6 wherever checked, and A = 0 at
rho = 0 for every rule.  The adjacency coefficient of the set model is reported
(probe dependence is between random fields, so it should be about 0).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3 import N_BOOT
from c3.endpoints.fieldtest import DECODE_FEATURES
from c3.infill import build_policy, config_key
from c3.probe import KS
from c3.probe.grid import prepare, run_trajectories, summarise, summarise_matched
from c3.probe.instances import ProbeSpec, eval_instances, train_instances
from c3.provenance import collect as provenance, sha256_file
from c3.setcal import SetCalibrator, build_set_table, crossfit_token_probs
from c3.util import hard_exit

TOKENIZER_OF = {"local-110m": "gpt2"}
TAUS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)
GAMMAS = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
ALPHAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def grid(a) -> list[dict]:
    cfgs = [{"policy": r, "k": k} for r in ("random", "confidence") for k in a.ks]
    cfgs += [{"policy": "threshold", "tau": t} for t in a.taus]
    cfgs += [{"policy": "entropy_bounded", "gamma": g} for g in a.gammas]
    cfgs += [{"policy": r, "alpha": al} for r in ("c3a", "c3indep", "c3set") for al in a.alphas]
    return cfgs


def calibrate(model, spec, a, log=print):
    import pandas as pd

    inst = list(train_instances(spec, a.n_calib, label="probe-c3calib"))
    prepared = prepare(spec, inst, model.tokenizer, TOKENIZER_OF[a.model])
    frames = []
    for j, k in enumerate(a.calib_ks):
        states: list = []
        run_trajectories(model, spec, prepared, k, "confidence", a.seed, a.batch_size,
                         max_cells=0, states=states)
        df = pd.DataFrame(states)
        df["state_id"] += (j + 1) * 10_000_000        # distinct states across the k runs
        frames.append(df)
        log(f"[probe-c3] calibration k={k}: {len(df):,} field rows")
    df = pd.concat(frames, ignore_index=True)
    feats = () if a.no_field else tuple(a.token_features)
    c_hat, tok = crossfit_token_probs(df, np.ones(len(df), dtype=bool), features=feats)
    table = build_set_table(df, c_hat)
    setcal = SetCalibrator(use_field=not a.no_field).fit(table, table["weight"].to_numpy())
    return df, table, tok, setcal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="artifacts/probe/spec.json")
    ap.add_argument("--model", default="local-110m", choices=sorted(TOKENIZER_OF))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="artifacts/probe/results")
    ap.add_argument("--n-calib", type=int, default=2000)
    ap.add_argument("--calib-ks", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--token-features", nargs="*", default=list(DECODE_FEATURES))
    ap.add_argument("--no-field", action="store_true",
                    help="calibrators on own confidence only (the item-8 gate's 'no' branch)")
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--taus", type=float, nargs="+", default=list(TAUS))
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMAS))
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    ap.add_argument("--speeds", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--per-rho", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="first N evaluation instances (testing)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-cells", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    from c3.traces.models import build_model

    t0 = time.time()
    spec = ProbeSpec.load(a.spec)
    model = build_model(a.model, precision=a.precision, path=a.model_path, device=a.device)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    calib_df, table, tok, setcal = calibrate(model, spec, a)
    tok.save(out / f"c3cal_tok_{a.tag}.json")
    setcal.save(out / f"c3cal_set_{a.tag}.json")
    calib_df.to_parquet(out / f"c3calib_states_{a.tag}.parquet", index=False)
    print(f"[probe-c3] calibrators: token {tok.used}, set coef "
          f"{ {k: round(v, 3) for k, v in setcal.coef().items() if not k.startswith('f_u')} }", flush=True)

    instances = eval_instances(spec, per_rho=a.per_rho)
    if a.limit:
        instances = instances[: a.limit]
    prepared = prepare(spec, instances, model.tokenizer, TOKENIZER_OF[a.model])
    rows = []
    for cfg in grid(a):
        rule = cfg["policy"]
        if rule in ("random", "confidence"):
            knob, policy = cfg["k"], None
        else:
            policy = build_policy(cfg, tok, setcal)
            knob = next(v for key, v in cfg.items() if key != "policy")
        t1 = time.time()
        got = run_trajectories(model, spec, prepared, knob, rule, a.seed, a.batch_size,
                               max_cells=a.max_cells, policy=policy)
        rows += got
        n_steps = len(got)
        fps = sum(r["n_commit"] for r in got) / max(n_steps, 1)
        print(f"[probe-c3] {config_key(cfg)}: {n_steps} steps, {fps:.2f} fields/step, "
              f"mean J {np.mean([r['J'] for r in got]):.3f} ({time.time() - t1:.0f} s)", flush=True)

    import pandas as pd

    pd.DataFrame(rows).to_parquet(out / f"steps_c3set_{a.tag}.parquet", index=False)
    summary = summarise(rows, n_boot=a.n_boot, seed=a.seed)
    matched = summarise_matched(rows, speeds=a.speeds, n_boot=a.n_boot, seed=a.seed)
    ident = summary["identity"]
    coef = setcal.coef()
    g6 = {"identity_max_abs_err_nats": ident["max_abs_err_nats"],
          "identity_ok": not ident["max_abs_err_nats"] > 1e-6,
          "max_A_at_rho0": matched["max_A_at_rho0"],
          "A_zero_at_rho0": not matched["max_A_at_rho0"] > 1e-9,
          "adj_frac_coef": coef.get("adj_frac")}
    g6["pass"] = bool(g6["identity_ok"] and g6["A_zero_at_rho0"])
    table_rows = [{"rule": r, "fields_per_step": float(t), **{m: v["value"] for m, v in by_m.items()}}
                  for r, by_t in matched["matched"].items() for t, by_m in by_t.items()]
    record = {"tag": a.tag, "model": {"name": a.model, "path": a.model_path,
                                      "checkpoint_sha256": sha256_file(a.model_path),
                                      "revision": getattr(model, "revision", None)},
              "spec": a.spec, "spec_sha256": sha256_file(a.spec),
              "config": {k: v for k, v in vars(a).items() if k not in ("model_path",)},
              "calibration": {"n_instances": a.n_calib, "n_rows": int(len(calib_df)),
                              "n_prefix_rows": int(len(table)), "token": tok.to_dict(),
                              "set": setcal.to_dict(), "set_coef": coef},
              "t9": summary, "matched": matched, "g6": g6, "table": table_rows,
              "summary": {"g6_pass": g6["pass"], **{f"{c}@{t}/{m}": v["value"]
                          for c, by_t in matched["contrasts"].items()
                          for t, by_m in by_t.items() for m, v in by_m.items()},
                          **{f"calibration/{r}/{k}": v for r, rec in matched["calibration"].items()
                             for k, v in rec.items() if not isinstance(v, dict)}},
              "seconds": round(time.time() - t0, 1), "provenance": provenance(with_gpu=a.device.startswith("cuda"))}
    path = out / f"c3set_{a.tag}.json"
    path.write_text(json.dumps(record, indent=1, default=float))
    print(f"[probe-c3] G6 {'PASS' if g6['pass'] else 'FAIL'}: identity {ident['max_abs_err_nats']:.2e} "
          f"nats, max |A| at rho=0 {matched['max_A_at_rho0']:.2e}, adjacency coef "
          f"{coef.get('adj_frac', float('nan')):+.3f} -> {path}")
    return 0 if g6["pass"] else 1


if __name__ == "__main__":
    hard_exit(main())
