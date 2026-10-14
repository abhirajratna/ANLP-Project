#!/usr/bin/env python3
"""Log set-level C3 results to W&B: one run per analysis, grouped per study.

    python scripts/wandb_log_results.py --group c3-setlevel-110m-v4 --name v4-dev-infill \\
        --job-type dev --tags dev exploratory --infill runs/c3set/dev/infill_sweep1.json

Any of --infill, --fieldtest, --setcal, --setcal-onpolicy, --probe, --floor may be
given; each becomes summary scalars and tables, and every input JSON is attached as
a ``results`` artifact.  ``--config`` (e.g. prereg/c3set_frozen_v4.json) becomes the
run config.  The results JSONs on /home2 stay the source of truth: this script
never fails the job (exit 0 always), and when the online start fails it logs
offline to ./wandb for slurm/wandb_backfill.sbatch (C3_SYNC_OFFLINE=1) to sync.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3.util import flat_scalars

OLD = ("topk_conf", "margin", "pc_sampler", "threshold", "entropy_bounded", "klass", "c3a")


def start(a, config: dict):
    import wandb

    rid = hashlib.sha1(f"{a.entity}/{a.project}/{a.group}/{a.name}".encode()).hexdigest()[:12]
    kw = dict(project=a.project, entity=a.entity, group=a.group, name=a.name, id=rid,
              job_type=a.job_type, tags=a.tags, config=config, resume="allow")
    try:
        return wandb, wandb.init(**kw)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[wandb] online start failed ({exc}); logging offline")
    try:
        if hasattr(wandb, "teardown"):
            wandb.teardown()
        return wandb, wandb.init(mode="offline", **kw)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[wandb] disabled: {exc}")
        return wandb, None


def infill_summary(r: dict) -> dict:
    out = {"infill/n_docs": r["n_docs"], "infill/n_configs": r["n_configs"]}
    for t, p in r["primary"].items():
        k = f"infill/delta_best@tps{float(t):g}"
        out.update({k: p["value"], f"{k}_lo": p["ci"][0], f"{k}_hi": p["ci"][1],
                    f"{k}_p_holm": p["p_holm"], f"{k}_improves": p["improves"],
                    f"infill/acc_{r['new']}@tps{float(t):g}": p["new_acc"],
                    f"infill/acc_best_baseline@tps{float(t):g}": p["best_baseline_acc"],
                    f"infill/best_baseline@tps{float(t):g}": p["best_baseline"]})
    for name, by_t in r.get("secondary", {}).items():
        for t, s in by_t.items():
            out[f"infill/secondary/{name}@tps{float(t):g}"] = s["value"]
            out[f"infill/secondary/{name}@tps{float(t):g}_lo"] = s["ci"][0]
            out[f"infill/secondary/{name}@tps{float(t):g}_hi"] = s["ci"][1]
    return out


def infill_tables(wandb, run, r: dict, new: str) -> None:
    cols = ["policy", "group", "config", "tokens_per_step", "log2_tps", "acc", "acc_lo", "acc_hi",
            "step_all_right", "seconds_per_doc"]
    rows = [[c["policy"], "new" if c["policy"] == new else ("old" if c["policy"] in OLD else "other"),
             c["config"], c["tokens_per_step"], float(np.log2(c["tokens_per_step"])), c["acc"],
             c["acc_ci"][0], c["acc_ci"][1], c["step_all_right"], c["seconds_per_doc"]]
            for c in r["curves"]]
    run.log({"infill/curves": wandb.Table(columns=cols, data=rows)})
    mrows = [[p, float(t), v["acc"], v["acc_ci"][0], v["acc_ci"][1], v["step_all_right"]]
             for p, by_t in r["at_speed"].items() for t, v in by_t.items()]
    run.log({"infill/matched": wandb.Table(columns=["policy", "tokens_per_step", "acc", "acc_lo",
                                                    "acc_hi", "step_all_right"], data=mrows)})
    series = {}
    for c in sorted(r["curves"], key=lambda c: c["tokens_per_step"]):
        series.setdefault(c["policy"], ([], []))
        series[c["policy"]][0].append(float(np.log2(c["tokens_per_step"])))
        series[c["policy"]][1].append(c["acc"])
    keys = sorted(series, key=lambda p: (p != new, p))
    run.log({"infill/accuracy_vs_speed": wandb.plot.line_series(
        xs=[series[k][0] for k in keys], ys=[series[k][1] for k in keys],
        keys=[f"{k} (new)" if k == new else k for k in keys],
        title="Infilling accuracy vs log2 tokens/step", xname="log2 tokens/step")})


def setcal_summary(r: dict, prefix: str) -> dict:
    out = {}
    for arm, block in r.get("arms", {}).items():
        for cmp, by_m in block["compare"].items():
            for m, s in by_m.items():
                k = f"{prefix}/{arm}/{cmp}/{m}"
                out.update({f"{k}_gain": s["gain"], f"{k}_lo": s["ci"][0], f"{k}_hi": s["ci"][1],
                            f"{k}_rel": s["rel_gain"], f"{k}_helps": s["helps"]})
        for name, e in block["ece"].items():
            out[f"{prefix}/{arm}/ece_{name}"] = e["ece"]
    for k, v in r.get("set_coef", {}).items():
        if not k.startswith("f_u"):
            out[f"{prefix}/coef/{k}"] = v
    return out


def setcal_table(wandb, run, r: dict, prefix: str) -> None:
    rows = []
    for arm, block in r.get("arms", {}).items():
        for name, recs in block["gap_by_j"].items():
            for rec in recs:
                rows.append([arm, name, rec["j"], rec["gap"], rec["ci_low"], rec["ci_high"]])
    if rows:
        run.log({f"{prefix}/gap_by_j": wandb.Table(
            columns=["arm", "estimate", "j", "gap", "lo", "hi"], data=rows)})


def fieldtest_summary(r: dict) -> dict:
    out = {"fieldtest/use_field_features": r["gate"]["use_field_features"]}
    for arm, by_set in r["arms"].items():
        for fs, res in by_set.items():
            for cmp in ("spline_vs_context", "isotonic_vs_context"):
                for m, s in res[cmp].items():
                    if isinstance(s, dict) and "gain" in s:
                        k = f"fieldtest/{arm}/{fs}/{cmp}/{m}"
                        out.update({f"{k}_gain": s["gain"], f"{k}_lo": s["ci"][0],
                                    f"{k}_hi": s["ci"][1], f"{k}_helps": s["helps"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="c3-dllm-calibration")
    ap.add_argument("--entity", default="ANLP-project")
    ap.add_argument("--group", default="c3-setlevel-110m-v4")
    ap.add_argument("--name", required=True)
    ap.add_argument("--job-type", default="analysis")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--config", default=None, help="JSON to use as the run config")
    for k in ("infill", "fieldtest", "setcal", "setcal-onpolicy", "probe", "floor"):
        ap.add_argument(f"--{k}", default=None)
    a = ap.parse_args()

    try:
        inputs = {k: getattr(a, k.replace("-", "_")) for k in
                  ("infill", "fieldtest", "setcal", "setcal-onpolicy", "probe", "floor")}
        inputs = {k: v for k, v in inputs.items() if v and Path(v).exists()}
        data = {k: json.loads(Path(v).read_text()) for k, v in inputs.items()}
        config = json.loads(Path(a.config).read_text()) if a.config else {}
        config["inputs"] = inputs
        wandb, run = start(a, config)
        if run is None:
            return 0
        summary = {}
        if "infill" in data:
            summary.update(infill_summary(data["infill"]))
            infill_tables(wandb, run, data["infill"], data["infill"]["new"])
        if "fieldtest" in data:
            summary.update(fieldtest_summary(data["fieldtest"]))
        for k, prefix in (("setcal", "setcal"), ("setcal-onpolicy", "setcal_onpolicy")):
            if k in data:
                summary.update(setcal_summary(data[k], prefix))
                setcal_table(wandb, run, data[k], prefix)
        if "probe" in data:
            summary.update(flat_scalars(data["probe"].get("summary", {}), "probe/"))
            if data["probe"].get("table"):
                t = data["probe"]["table"]
                run.log({"probe/matched": wandb.Table(columns=list(t[0]), data=[list(r.values()) for r in t])})
        if "floor" in data:
            summary.update(flat_scalars({k: v for k, v in data["floor"].items()
                                         if not isinstance(v, (list, dict))}, "floor/"))
        for k, v in summary.items():
            run.summary[k] = v
        art = wandb.Artifact(f"{a.name}-results", type="results")
        for v in inputs.values():
            art.add_file(v)
        run.log_artifact(art)
        run.finish()
        print(f"[wandb] logged {len(summary)} summary values, {len(inputs)} files -> "
              f"{getattr(run, 'url', None) or 'offline run'}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[wandb] logging failed, results JSONs unaffected: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
