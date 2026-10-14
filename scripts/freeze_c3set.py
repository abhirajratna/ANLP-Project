#!/usr/bin/env python3
"""Freeze the set-level C3 analysis before the report half is read (Phase 7).

    python scripts/freeze_c3set.py --grid prereg/c3set_policies_final.json \\
        --cal-dir /home2/$USER/c3-runs/c3set/dev/cal/final-traces --cal-source traces \\
        --dev-dir /home2/$USER/c3-runs/c3set/dev --dev-infill infill_sweep1.json \\
        --trace-manifest runs/main-110m-v4.dataset.json --validate artifacts/c1/main-110m-v4.json \\
        --checkpoint-sha256 c146c86e... --probe artifacts/probe/results/c3set_ft110m-v4.json \\
        --out prereg/c3set_frozen_v4.json

Writes one JSON holding everything the final run depends on, each with its sha256:
the code commit (the tree must be clean), the checkpoint, the trace manifest and its
validation, the split manifest and the dev rule, the knob grid, the frozen
calibrators and how they were chosen, the item-8 gate result, the primary statistic
and its bootstrap, and the W&B group.  ``prereg/c3set_commands.sh`` refuses to run
unless the checked-out code is the tag that commits this file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.provenance import collect as provenance, sha256_file


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def digest(path) -> dict:
    p = Path(path)
    return {"path": str(p), "sha256": sha256_file(p)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--cal-dir", required=True, help="the frozen tok.json / set.json")
    ap.add_argument("--cal-source", choices=["traces", "onpolicy"], required=True)
    ap.add_argument("--copy-cal-to", default="prereg/c3set_cal_v4",
                    help="the calibrators are copied into the repository and frozen with it")
    ap.add_argument("--dev-dir", required=True, help="dev results (fieldtest.json, setcal*.json)")
    ap.add_argument("--dev-infill", nargs="+", default=[], help="dev infill analyses (in --dev-dir)")
    ap.add_argument("--trace-manifest", required=True)
    ap.add_argument("--validate", required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--split", default="artifacts/split_manifest.json")
    ap.add_argument("--speeds", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--boot-label", default="c3set-v4-report")
    ap.add_argument("--wandb-group", default="c3-setlevel-110m-v4")
    ap.add_argument("--note", default="")
    ap.add_argument("--allow-dirty", action="store_true", help="testing only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not a.allow_dirty:
        raise SystemExit(f"refusing to freeze a dirty tree:\n{dirty}")
    dev = Path(a.dev_dir)
    field = json.loads((dev / "fieldtest.json").read_text())
    validate = json.loads(Path(a.validate).read_text())
    import shutil

    cal = Path(a.copy_cal_to)
    cal.mkdir(parents=True, exist_ok=True)
    for f in ("tok.json", "set.json"):
        shutil.copyfile(Path(a.cal_dir) / f, cal / f)
    rec = {
        "what": "Frozen specification of the set-level C3 final run (deviations.md items 7-13). "
                "Nothing in it may change after the report half is read.",
        "note": a.note,
        "code": {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty)},
        "checkpoint": {"path": "/share1/<user>/train110m-v4/final.pt",
                       "sha256": a.checkpoint_sha256},
        "traces": {"manifest": digest(a.trace_manifest), "validate": digest(a.validate),
                   "all_ok": validate.get("all_ok"), "n_rows": validate.get("n_rows")},
        "split": {**digest(a.split), "report_half": 1,
                  "dev_rule": "hash_halves(fit_docs, 1): 0 = dev-fit, 1 = dev-eval"},
        "gate_item8": {"source": digest(dev / "fieldtest.json"), **field["gate"]},
        "calibrators": {"source": a.cal_source, "fitted_in": str(a.cal_dir),
                        "token": digest(cal / "tok.json"),
                        "set": digest(cal / "set.json"),
                        "chosen_by": "lower held-out set log-loss on free-running dev-eval states "
                                     "(setcal_onpolicy.json vs setcal_onpolicy_tracecal.json)"},
        "grid": {**digest(a.grid), "configs": json.loads(Path(a.grid).read_text())["configs"]},
        "primary": {"statistic": "Delta_max(t) = acc_c3set(t) - max over the six existing "
                                 "decoders of acc_b(t), max inside each bootstrap draw",
                    "speeds_tokens_per_step": a.speeds,
                    "interpolation": "linear in log2(tokens/step) between a policy's "
                                     "configurations; NaN outside its range",
                    "baselines": ["topk_conf", "margin", "pc_sampler", "threshold",
                                  "entropy_bounded", "klass"],
                    "multiplicity": "Holm over the speeds",
                    "declare": "Holm p <= 0.05, 95% interval excludes 0, |gain| > 2 x floor, gain > 0",
                    "floor": 0.0,
                    "n_boot": a.n_boot, "boot_label": a.boot_label},
        "dev_results": {f: digest(dev / f) for f in
                        ["fieldtest.json", "setcal.json", "setcal_onpolicy.json",
                         "setcal_onpolicy_tracecal.json", *a.dev_infill] if (dev / f).exists()},
        "probe": digest(a.probe) if a.probe else None,
        "wandb": {"entity": "ANLP-project", "project": "c3-dllm-calibration",
                  "group": a.wandb_group},
        "provenance": provenance(),
    }
    out = Path(a.out)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[freeze] -> {out}; commit it, then tag: git tag c3set-freeze-v4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
