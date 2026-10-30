#!/usr/bin/env python3
"""Pick T12's per-task decoder knobs from fit-half calibration runs.

    python3 -I scripts/pick_knobs.py runs/t12/cal-gsm8k runs/t12/cal-mbpp [--out-dir DIR]

Input: one ``run_t12.py`` output directory per task, run with ``prereg/t12_cal.json`` and
``--half fit`` (``jarvis_t12.sh calibrate`` makes both).  Output, per task, a grid with the
7 decoders x 3 speeds of T12, written as ``DIR/t12_grid_<task>.json`` with ``--out-dir``
(nothing is written without it).  Run T12 once per task with that task's file, e.g.
``GRID=prereg/t12_grid_gsm8k.json bash scripts/jarvis_t12.sh run sweep-gsm8k gsm8k report``.

* Speed of a configuration = ``gen_length * N / sum(n_forward)`` over its N problems, the
  aggregate tokens per forward pass.  It is not the mean of per-problem ratios (those
  differ by ~10% at tau 0.9).  Only problems every configuration of the run finished count.
* topk_conf, margin and pc_sampler are exact: k = 2, 4, 8 tokens per step (block 32).
* threshold, entropy_bounded, klass and c3a: the knob is interpolated so that the speed is
  2, 4 and 8 tokens per step, linearly in log(speed) against logit(tau) / logit(alpha) /
  log(gamma) / (logit(tau), log(eps_kl)) along the calibrated path, between the first pair
  of neighbouring settings that crosses the target.  The picks are therefore monotone in the
  target.  Knobs follow speed only, never fit-half accuracy.
* A target above the fastest calibrated setting is served by that setting if it is within
  10% of the target.  Otherwise the run is refused (extend the grid by a new PR), except
  KLASS at the top target: the fastest KLASS setting takes that slot, flagged
  ``unmatched_at_target`` in ``picks`` with the speed it achieved.  Downstream matching
  reads it as '---' at 8 (history 2 caps KLASS at 32/3 = 10.7 tokens per step).
* Refuses any run whose meta ``half`` is not ``fit``, which has no manifest to check against,
  whose prompt is not the chat prompt, or that holds a record of a report-half problem.

Standalone (standard library only): ``jarvis_t12.sh calibrate`` runs it with ``python -I``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

TARGETS = (2.0, 4.0, 8.0)
TOLERANCE = 0.10
KNOB_FIELDS = ("policy", "k", "tau", "alpha", "gamma", "eps_kl", "history", "fallback", "lam")
ADAPTIVE = ("threshold", "entropy_bounded", "klass", "c3a")
SPEED_METRIC = ("tokens/step of a configuration = gen_length * N / sum(n_forward) over its N "
                "problems (aggregate, not the mean of per-problem ratios)")
MATCHING = ("for each C3a setting and task, each baseline at its setting whose tokens/step "
            "(speed_metric, same task) is nearest, used only if within +-10% of C3a's, else "
            "'---'; difference = paired accuracy difference (C3a - baseline) on the shared "
            "doc_ids, 95% percentile CI from 2,000 bootstrap resamples of problems.")


class OutOfRange(ValueError):
    """The calibrated settings do not reach a target (and the 10% allowance does not cover it)."""


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def expit(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def config_key(cfg: dict) -> str:
    """The same key as scripts/run_t12.py:config_key (a test keeps them equal)."""
    knobs = ",".join(f"{k}={cfg[k]}" for k in sorted(cfg) if k != "policy")
    return f"{cfg['policy']}[{knobs}]"


def aggregate_speed(records: list[dict], gen_length: int) -> float:
    """Tokens per forward pass over a set of problems: gen_length * N / sum(n_forward)."""
    if not records:
        raise ValueError("no records")
    return gen_length * len(records) / sum(r["n_forward"] for r in records)


# ---- the knob of each adaptive policy -----------------------------------------------------
# axis: larger = more aggressive (faster).  coords: the interpolation coordinates of a setting.
# build: coordinates -> the knob fields written into the grid.

def _sig2(x: float) -> float:
    return float(f"{x:.2g}")


def _axis(policy: str, cfg: dict) -> float:
    return {"threshold": lambda: -cfg["tau"], "klass": lambda: -cfg["tau"],
            "entropy_bounded": lambda: cfg["gamma"], "c3a": lambda: cfg["alpha"]}[policy]()


def _coords(policy: str, cfg: dict) -> tuple:
    if policy == "threshold":
        return (logit(cfg["tau"]),)
    if policy == "c3a":
        return (logit(cfg["alpha"]),)
    if policy == "entropy_bounded":
        return (math.log(cfg["gamma"]),)
    return (logit(cfg["tau"]), math.log(cfg["eps_kl"]))                      # klass


def _build(policy: str, x: tuple, template: dict) -> dict:
    if policy == "threshold":
        return {"tau": round(expit(x[0]), 3)}
    if policy == "c3a":
        return {"alpha": round(expit(x[0]), 3)}
    if policy == "entropy_bounded":
        return {"gamma": _sig2(math.exp(x[0]))}
    return {"tau": round(expit(x[0]), 3), "eps_kl": _sig2(math.exp(x[1])),
            "history": template["history"], "fallback": template["fallback"]}


def _knobs_of(policy: str, cfg: dict) -> dict:
    keys = {"threshold": ("tau",), "c3a": ("alpha",), "entropy_bounded": ("gamma",),
            "klass": ("tau", "eps_kl", "history", "fallback")}[policy]
    return {k: cfg[k] for k in keys}


def pick_for_target(policy: str, points: list[tuple], target: float,
                    tolerance: float = TOLERANCE) -> tuple[dict, dict]:
    """The knob of ``policy`` whose speed is ``target``: (knob fields, info).

    ``points``: [(tokens/step, config)] of the calibrated settings of this policy, in any
    order.  Raises OutOfRange when the target is not reached (see the module docstring).
    ``info``: ``unmatched_at_target``, ``achieved_tokens_per_step`` (the speed of the setting
    used when no interpolation happened, else None) and ``bracket`` (the two speeds
    interpolated between, else None).
    """
    pts = sorted(points, key=lambda p: _axis(policy, p[1]))
    ys = [p[0] for p in pts]

    def whole(i, unmatched=False):
        return (_knobs_of(policy, pts[i][1]),
                {"unmatched_at_target": unmatched, "achieved_tokens_per_step": ys[i],
                 "bracket": None})

    j = next((i for i, y in enumerate(ys) if y >= target), None)
    if j is None:                                         # the target is above every setting
        fastest = max(range(len(ys)), key=ys.__getitem__)
        if ys[fastest] >= target * (1 - tolerance):
            return whole(fastest)
        if policy == "klass" and target == max(TARGETS):
            return whole(fastest, unmatched=True)
        raise OutOfRange(f"{policy}: fastest calibrated setting is {ys[fastest]:.2f} tok/step, "
                         f"target {target:g}")
    if j == 0:
        if ys[0] <= target * (1 + tolerance):
            return whole(0)
        raise OutOfRange(f"{policy}: slowest calibrated setting is {ys[0]:.2f} tok/step, "
                         f"target {target:g}")
    y0, y1 = ys[j - 1], ys[j]                             # y0 < target <= y1
    w = (math.log(target) - math.log(y0)) / (math.log(y1) - math.log(y0))
    x0, x1 = _coords(policy, pts[j - 1][1]), _coords(policy, pts[j][1])
    x = tuple(a + w * (b - a) for a, b in zip(x0, x1))
    return (_build(policy, x, pts[j - 1][1]),
            {"unmatched_at_target": False, "achieved_tokens_per_step": None,
             "bracket": [y0, y1]})


# ---- reading a calibration run --------------------------------------------------------------

def load_run(run_dir) -> tuple[dict, dict[str, list[dict]]]:
    """(meta, records by configuration key) of a fit-half calibration run, or SystemExit."""
    run = str(run_dir)
    metas = sorted(glob.glob(run + "/t12.shard*.meta.json"))
    if not metas:
        sys.exit(f"{run}: no t12.shard*.meta.json")
    meta = json.loads(Path(metas[0]).read_text())
    for m in metas:
        half = json.loads(Path(m).read_text()).get("half")
        if half != "fit":
            sys.exit(f"{run}: meta half={half!r}; the knob calibration uses the fit half only")
    if (meta.get("prompt") or {}).get("format") != "chat":
        sys.exit(f"{run}: prompt {meta.get('prompt')}; the calibration must use the chat "
                 "prompt of the sweep")
    if not meta.get("manifest"):
        sys.exit(f"{run}: the meta names no split manifest, so the fit half cannot be verified")
    split = json.loads(Path(meta["manifest"]).read_text())
    by: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(glob.glob(run + "/t12.shard*.jsonl")):
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                                   # a torn last line
            half = split.get(str(r["doc_id"]))
            if half != 0:
                sys.exit(f"{run}: doc {r['doc_id']} is "
                         f"{'report half' if half == 1 else 'not in the manifest'}; "
                         "the knob calibration uses the fit half only")
            if not r.get("error"):
                by[r["config"]].append(r)
    if not by:
        sys.exit(f"{run}: no usable records")
    return meta, by


def measure(by: dict[str, list[dict]], gen_length: int) -> tuple[dict, int]:
    """Per configuration, on the problems every configuration finished:
    {key: {config, n, tokens_per_step, mean_of_ratios, ms_per_forward}}, and that N."""
    common = set.intersection(*[{r["doc_id"] for r in rs} for rs in by.values()])
    if not common:
        sys.exit("no problem was finished by every configuration")
    out = {}
    for key, rs in sorted(by.items()):
        rs = [r for r in rs if r["doc_id"] in common]
        nfe = sum(r["n_forward"] for r in rs)
        out[key] = {"config": {k: rs[0][k] for k in KNOB_FIELDS if k in rs[0]},
                    "n": len(rs), "tokens_per_step": aggregate_speed(rs, gen_length),
                    "mean_of_ratios": sum(r["tokens_per_step"] for r in rs) / len(rs),
                    "ms_per_forward": 1000 * sum(r["seconds"] for r in rs) / nfe}
    return out, len(common)


# ---- the grid of one task --------------------------------------------------------------------

def exact_configs() -> list[dict]:
    """topk_conf, margin and pc_sampler at k = 2, 4, 8 (block 32 is divisible by each)."""
    return ([{"policy": "topk_conf", "k": k} for k in (2, 4, 8)]
            + [{"policy": "margin", "k": k} for k in (2, 4, 8)]
            + [{"policy": "pc_sampler", "k": k, "lam": 0.25, "alpha": 10.0} for k in (2, 4, 8)])


def build_grid(task: str, meas: dict, n_problems: int, meta: dict) -> dict:
    configs = exact_configs()
    slots = {config_key(c): 1 + (2, 4, 8).index(c["k"]) for c in configs}
    by_policy = defaultdict(list)
    for m in meas.values():
        by_policy[m["config"]["policy"]].append((m["tokens_per_step"], m["config"]))
    picks, chosen = [], {p: [] for p in ADAPTIVE}
    for pol in ADAPTIVE:
        if len(by_policy[pol]) < 2:
            sys.exit(f"{task}: {len(by_policy[pol])} calibrated {pol} settings; need at least 2")
        for slot, target in enumerate(TARGETS, 1):
            try:
                knobs, info = pick_for_target(pol, by_policy[pol], target)
            except OutOfRange as exc:
                sys.exit(f"{task}: {exc}; the calibration grid does not bracket this target")
            cfg = {"policy": pol, **knobs}
            chosen[pol].append(cfg)
            slots[config_key(cfg)] = slot
            picks.append({"policy": pol, "speed": slot, "target_tokens_per_step": target,
                          "config_key": config_key(cfg), "knobs": knobs, **info})
    for pol in ADAPTIVE:
        configs += chosen[pol]
    keys = [config_key(c) for c in configs]
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        sys.exit(f"{task}: two speeds picked the same setting {dup}; run_t12 refuses duplicate "
                 "configurations. Look at the calibration table and decide by PR.")
    return {
        "what": (f"T12 report-half sweep grid for {task} (plan v2 T12; plan to the final, Parth "
                 "5): LLaDA-8B-Instruct fp16, zero-shot chat prompt, gen_length 256, "
                 "block_length 32, greedy. 7 decoders x 3 speeds, target tokens/step 2 / 4 / 8 "
                 "(speed 1 / 2 / 3). Knobs interpolated from FIT-half calibration runs of this "
                 "task only (scripts/pick_knobs.py)."),
        "status": "calibrated", "task": task, "half_used_for_calibration": "fit",
        "speed_metric": SPEED_METRIC,
        "targets": {"speed1": TARGETS[0], "speed2": TARGETS[1], "speed3": TARGETS[2]},
        "matching": MATCHING,
        "calibration": {"n_problems": n_problems, "run_id": meta.get("run_id"),
                        "configs_sha256": meta.get("configs_sha256"),
                        "precision": meta.get("precision"),
                        "settings": {k: {f: m[f] for f in ("n", "tokens_per_step",
                                                           "mean_of_ratios", "ms_per_forward")}
                                     for k, m in meas.items()}},
        "picks": picks,
        "speed_of_config": slots,
        "configs": configs,
    }


def pick_runs(run_dirs) -> dict[str, dict]:
    """{task: grid} from fit-half calibration runs, one run per task."""
    grids = {}
    for run in run_dirs:
        meta, by = load_run(run)
        task = meta["task"]
        if task in grids:
            sys.exit(f"two runs for task {task}: one calibration run per task")
        meas, n = measure(by, int(meta["gen_length"]))
        grids[task] = build_grid(task, meas, n, meta)
    return grids


def report(grid: dict) -> str:
    cal = grid["calibration"]
    lines = [f"== {grid['task']}: {cal['n_problems']} common fit-half problems"]
    for key, m in cal["settings"].items():
        lines.append(f"  {key:56s} n={m['n']:3d}  {m['tokens_per_step']:6.2f} tok/step "
                     f"(mean of ratios {m['mean_of_ratios']:6.2f})  {m['ms_per_forward']:5.1f} ms/forward")
    lines.append(f"  picks (target tok/step 2 / 4 / 8):")
    for p in grid["picks"]:
        how = (f"setting at {p['achieved_tokens_per_step']:.2f}" if p["bracket"] is None
               else f"between {p['bracket'][0]:.2f} and {p['bracket'][1]:.2f}")
        flag = "   UNMATCHED at target: '---' downstream" if p["unmatched_at_target"] else ""
        lines.append(f"  {p['policy']:16s} speed {p['speed']} (target {p['target_tokens_per_step']:g}): "
                     f"{json.dumps(p['knobs'])}  [{how}]{flag}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="fit-half calibration run directories, one per task")
    ap.add_argument("--out-dir", default=None,
                    help="write t12_grid_<task>.json here (default: print only)")
    a = ap.parse_args(argv)
    grids = pick_runs(a.runs)
    for grid in grids.values():
        print(report(grid))
    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for task, grid in grids.items():
            path = out / f"t12_grid_{task}.json"
            path.write_text(json.dumps(grid, indent=1) + "\n", encoding="utf-8", newline="\n")
            print(f"[pick_knobs] wrote {path}")
    else:
        print("[pick_knobs] nothing written; add --out-dir prereg to write t12_grid_<task>.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
