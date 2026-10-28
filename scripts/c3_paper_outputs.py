#!/usr/bin/env python3
"""C3 for the paper: Sec. 6 and appendix A.3 macros, table and figure from the T12 outputs.

    python scripts/c3_paper_outputs.py \\
        --records runs/t12/jl-report-<sha7> --verdicts artifacts/t12/<run>_verdicts.jsonl \\
        --grids prereg --calibrators artifacts/c3 --sanity artifacts/t12/llada-8b-instruct_fp16_gsm8k-sanity.json \\
        --out-dir artifacts

No torch, no GPU, no model: it reads files the other scripts wrote, on the laptop.  Every input is
optional; what is absent is ``---`` in a table and an undefined macro (the paper's ``\\Cres`` then
prints a bold ?), never a crash.

Inputs
  --records     ``scripts/run_t12.py`` output: ``t12.shard<i>of<n>.jsonl`` (+ ``.meta.json``), a
                directory (searched recursively), a glob, or files.  Only report-half, non-lmeval
                shards count (the meta file's ``half`` / ``prompt.format``); a (task, config,
                doc_id) seen twice (a resumed shard) counts once.
  --verdicts    ``scripts/score_t12.py --per-record`` JSONL: whether each record is correct.  The
                scoring is not redone here.
  --scores      ``scripts/score_t12.py --out`` JSON: when there are no verdicts for a configuration,
                its row (n, n_correct, ci95) gives the accuracy; differences need the verdicts.
  --grids       ``prereg/t12_grid_{gsm8k,mbpp}.json`` (``scripts/pick_knobs.py``): which configuration
                is which decoder at which speed slot, C3a's alpha per slot, and ``picks[*]
                .unmatched_at_target``.
  --calibrators ``artifacts/c3/llada-8b-instruct_fp16_c3a-{gsm8k,mbpp}.json`` (T11).
  --sanity      the Ada sanity run, ``score_t12.py`` output.

Computation (prereg/deviations.md, entry 17)
  speed of a configuration   256 N / sum(NFE) over its N problems, not the mean of per-problem ratios.
  accuracy                   from the verdicts, with the 95% Wilson interval (``score_t12.wilson``).
  matched speed              for each C3a setting (speed slot s) each baseline is taken at its setting
                             whose speed is nearest to C3a's, within +-10%, else ``---``; KLASS where the
                             grid says ``unmatched_at_target``.  The difference C3a - baseline is paired on
                             the shared doc_ids, with a 95% percentile CI from 2,000 bootstrap resamples of
                             problems, seed 20261030 (reseeded for every comparison).
  OutsideShare               share of C3a decode steps (``c3a_steps``) with a feature outside the
                             calibrator's fit range, pooled over C3a's settings.
  primary                    speed 2 is Sec. 6's table slot.

Writes, under ``--out-dir`` (default ``artifacts``)
  c3/macros_c3.tex                                \\CThreeTEleven... \\CThreeTTwelve... (names: 06_c3.tex)
  t12/llada-8b-instruct_fp16_t12-summary.json     everything below, with provenance, input sha256s, the
                                                  pairing rule and seed
  t12/t12_matched_speed.tex                       Sec. 6 table: a bare booktabs tabular
  t12/t12_sanity_cost.tex                         the sanity row and the cost per decoder (slides, README)
  t12/figures/t12_matched_speed.{pdf,png}         appendix figure, one panel per task, x axes not shared
  t12/zip/t12_sanity_cost.{md,csv}, t12_sanity.csv, t12_speed1_speed3.{md,csv}   code archive tables
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from c3.provenance import collect as provenance, sha256_file  # noqa: E402
from c1_paper_outputs import DASH, Macros, est_ci, num  # noqa: E402

try:                                                      # one Wilson interval for the whole project
    from score_t12 import wilson                          # noqa: E402
except Exception:                                         # pragma: no cover - scoring deps absent
    def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
        if n == 0:
            return math.nan, math.nan
        p = k / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return centre - half, centre + half

SEED = 20261030
N_BOOT = 2000
TOL = 0.10
GEN_LENGTH = 256
PRIMARY_SPEED = 2
SLOTS = (1, 2, 3)
TASKS = ("gsm8k", "mbpp")
TASK_WORD = {"gsm8k": "Gsm", "mbpp": "Mbpp"}
TASK_LABEL = {"gsm8k": "GSM8K", "mbpp": "MBPP"}
SPEED_WORD = {1: "SpeedOne", 2: "SpeedTwo", 3: "SpeedThree"}
DECODERS = ("TopK", "Threshold", "Margin", "Entropy", "Klass", "PcSampler", "CThreeA")
POLICY_DEC = {"topk_conf": "TopK", "threshold": "Threshold", "margin": "Margin",
              "entropy_bounded": "Entropy", "klass": "Klass", "pc_sampler": "PcSampler",
              "c3a": "CThreeA"}
LABEL = {"TopK": "Top-$k$", "Threshold": "Threshold", "Margin": "Margin", "Entropy": "Entropy",
         "Klass": "KLASS", "PcSampler": "PC-Sampler", "CThreeA": "C3a"}
PLAIN = {"TopK": "top-k", "Threshold": "threshold", "Margin": "margin", "Entropy": "entropy-bounded",
         "Klass": "KLASS", "PcSampler": "PC-Sampler", "CThreeA": "C3a"}
EXACT_K = {2: 1, 4: 2, 8: 3}                              # top-k, margin, PC-Sampler: k -> speed slot
# Fast-dLLM, Table 1 (LLaDA-8B-Instruct, GSM8K, 5-shot): design constants of prereg/t12_sanity.json
PUBLISHED = {"topk_conf[k=1]": 0.793, "threshold[tau=0.9]": 0.792}
SANITY_DEC = {"topk_conf[k=1]": "TopK", "threshold[tau=0.9]": "Threshold"}
RULE = ("each baseline is taken at its setting whose aggregate tokens per step (256 N / sum NFE over "
        "the task's report-half problems) is nearest to C3a's at that slot, within +-10%, else '---' "
        "(KLASS: '---' where the grid says unmatched_at_target); difference = C3a minus baseline "
        "accuracy, paired on the shared doc_ids, 95% percentile CI from 2,000 bootstrap resamples "
        "of problems, seed 20261030 (reseeded per comparison)")
MODEL_TAG = "llada-8b-instruct_fp16"


# --------------------------------------------------------------------------- small helpers
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def config_key(cfg: dict) -> str:
    """The same key as ``scripts/run_t12.py`` (policy, then the sorted knobs)."""
    knobs = ",".join(f"{k}={cfg[k]}" for k in sorted(cfg) if k != "policy")
    return f"{cfg['policy']}[{knobs}]"


def fnum(x, nd: int) -> str:
    """A macro body: ``0.123``; a negative as ``\\ensuremath{-}0.123`` (a math minus in text too)."""
    s = f"{abs(x):.{nd}f}"
    return ("\\ensuremath{-}" + s) if x < 0 and float(s) != 0 else s


def _read_json(path: Path, warnings: list):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"unreadable {path}: {exc}")
        return None


def _expand(specs, pattern: str) -> list[Path]:
    """Files named by ``specs``: a file, a directory (``pattern`` below it), or a glob."""
    out = []
    for spec in specs or []:
        p = Path(spec)
        if p.is_dir():
            out += sorted(p.rglob(pattern))
        elif p.is_file():
            out.append(p)
        elif any(c in str(spec) for c in "*?["):
            out += sorted(Path(x) for x in _glob.glob(str(spec), recursive=True) if Path(x).is_file())
    seen, uniq = set(), []
    for f in out:
        if str(f) not in seen:
            seen.add(str(f))
            uniq.append(f)
    return uniq


# --------------------------------------------------------------------------- statistics
def aggregate_speed(nfes, gen_length: int = GEN_LENGTH) -> float:
    """Tokens per step of a configuration: total generated positions over total forward passes,
    ``gen_length * N / sum(NFE)`` (not the mean of the per-problem ratios)."""
    nfes = [n for n in nfes if _finite(n) and n > 0]
    return gen_length * len(nfes) / sum(nfes) if nfes else math.nan


def match_baseline(c3a_slot: int, c3a_tps: float, settings: dict, tol: float = TOL):
    """The setting of a baseline matched to C3a's slot ``c3a_slot`` at ``c3a_tps``.

    ``settings``: ``{slot: {"tps": aggregate tokens per step, "unmatched": bool}}``.  Returns
    ``{"slot", "tps", "rel_diff"}`` of the setting nearest to ``c3a_tps`` if it is within
    ``tol`` of it, else None.  The baseline's setting at ``c3a_slot`` flagged
    ``unmatched_at_target`` (KLASS cannot reach 8 tokens per step) is None whatever its speed."""
    if not _finite(c3a_tps) or c3a_tps <= 0:
        return None
    if (settings.get(c3a_slot) or {}).get("unmatched"):
        return None
    best = None
    for slot in sorted(settings):
        tps = settings[slot].get("tps")
        if not _finite(tps):
            continue
        rel = abs(tps - c3a_tps) / c3a_tps
        if best is None or rel < best[0] - 1e-12:
            best = (rel, slot, tps)
    if best is None or best[0] > tol + 1e-9:
        return None
    return {"slot": best[1], "tps": best[2], "rel_diff": best[0]}


def paired_bootstrap(a: dict, b: dict, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired accuracy difference ``a - b`` on the doc_ids both have (``{doc_id: 0/1}``), with the
    95% percentile interval of ``n_boot`` resamples of problems.  The generator is seeded here, per
    call, so a result does not depend on which other comparisons ran."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"diff": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    d = np.array([int(a[i]) - int(b[i]) for i in ids], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(n_boot, len(ids)))
    means = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": len(ids)}


# --------------------------------------------------------------------------- loading
def load_grids(specs, warnings: list, inputs: dict) -> dict:
    """``{task: {"slot_of": {config_key: slot}, "dec_of": {config_key: decoder}, "unmatched":
    {(decoder, slot): bool}, "alpha": {slot: alpha}}}``."""
    out = {}
    for f in _expand(specs, "t12_grid_*.json"):
        d = _read_json(f, warnings)
        if not isinstance(d, dict):
            continue
        m = re.search(r"t12_grid_(\w+)\.json$", f.name)
        task = d.get("task") or (m.group(1) if m else None)
        if task not in TASKS:
            warnings.append(f"{f}: no task")
            continue
        inputs[f"grid:{task}"] = {"path": str(f), "sha256": sha256_file(f)}
        keys = [config_key(c) for c in d.get("configs", []) if "policy" in c]
        slot_of = {k: int(v) for k, v in (d.get("speed_of_config") or {}).items()}
        picks = d.get("picks") or []
        for p in picks:
            slot_of.setdefault(p["config_key"], int(p["speed"]))
        targets = {1: 2.0, 2: 4.0, 3: 8.0}
        for k, v in (d.get("targets") or {}).items():
            m2 = re.fullmatch(r"speed(\d)", k)
            if m2 and _finite(v):
                targets[int(m2.group(1))] = float(v)
        for c in d.get("configs", []):                    # exact-k decoders without a speed_of_config
            key = config_key(c) if "policy" in c else None
            if key and key not in slot_of and "k" in c:
                for s, t in targets.items():
                    if c["k"] == t:
                        slot_of[key] = s
        dec_of = {k: POLICY_DEC[k.split("[", 1)[0]] for k in set(keys) | set(slot_of)
                  if k.split("[", 1)[0] in POLICY_DEC}
        unmatched = {}
        for p in picks:
            dec = POLICY_DEC.get(p.get("policy", ""))
            if dec:
                unmatched[(dec, int(p["speed"]))] = bool(p.get("unmatched_at_target"))
        alpha = {}
        for key, s in slot_of.items():
            if dec_of.get(key) == "CThreeA":
                m3 = re.search(r"alpha=([0-9.eE+-]+)", key)
                if m3:
                    alpha[s] = float(m3.group(1))
        out[task] = {"slot_of": slot_of, "dec_of": dec_of, "unmatched": unmatched, "alpha": alpha}
    return out


def load_records(specs, warnings: list, inputs: dict) -> dict:
    """``{task: {config_key: {doc_id: record}}}`` of the report half."""
    out, gen_length = {}, {}
    skipped = 0
    for f in _expand(specs, "t12.shard*.jsonl"):
        if f.name.endswith(".meta.json"):
            continue
        meta, meta_path = None, f.with_name(f.name[:-len(".jsonl")] + ".meta.json")
        if meta_path.exists():
            meta = _read_json(meta_path, warnings)
        meta = meta if isinstance(meta, dict) else {}
        fmt = (meta.get("prompt") or {}).get("format") if isinstance(meta.get("prompt"), dict) else meta.get("prompt")
        if meta.get("half") not in (None, "report") or fmt == "lmeval":
            skipped += 1
            continue
        inputs[f"records:{f}"] = {"path": str(f), "sha256": sha256_file(f)}
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("half") not in (None, "report"):
                continue
            task = r.get("task") or meta.get("task")
            if task not in TASKS or "config" not in r or "doc_id" not in r:
                continue
            gen_length.setdefault(task, meta.get("gen_length"))
            out.setdefault(task, {}).setdefault(r["config"], {}).setdefault(int(r["doc_id"]), r)
    if skipped:
        warnings.append(f"{skipped} record shard(s) skipped: not the report half, or the lmeval prompt")
    out["_gen_length"] = {t: g or GEN_LENGTH for t, g in gen_length.items()}
    return out


class Verdicts(dict):
    """``{(task, config): {doc_id: 0/1}}`` plus ``timeouts``: ``{(task, config): {doc_id}}``."""

    def __init__(self):
        super().__init__()
        self.timeouts = {}


def load_verdicts(specs, warnings: list, inputs: dict) -> Verdicts:
    """Verdicts of ``score_t12.py --per-record`` (lmeval verdicts, the sanity run's, are ignored)."""
    out = Verdicts()
    for f in _expand(specs, "*.jsonl"):
        inputs[f"verdicts:{f}"] = {"path": str(f), "sha256": sha256_file(f)}
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                v = json.loads(line)
            except ValueError:
                continue
            if v.get("prompt_format") == "lmeval" or "doc_id" not in v or "config" not in v:
                continue
            out.setdefault((v.get("task"), v["config"]), {}).setdefault(int(v["doc_id"]), int(bool(v.get("correct"))))
            if v.get("error") == "timeout":
                out.timeouts.setdefault((v.get("task"), v["config"]), set()).add(int(v["doc_id"]))
    return out


def load_scores(specs, warnings: list, inputs: dict) -> dict:
    """``{(task, config): row}`` of the report half, chat prompt, from ``score_t12.py --out``."""
    out = {}
    for f in _expand(specs, "*.json"):
        d = _read_json(f, warnings)
        if not isinstance(d, dict):
            continue
        inputs[f"scores:{f}"] = {"path": str(f), "sha256": sha256_file(f)}
        for r in d.get("rows", []):
            if r.get("prompt_format") in (None, "chat") and r.get("half") in (None, "report"):
                out.setdefault((r.get("task"), r.get("config")), r)
    return out


def load_calibrators(specs, warnings: list, inputs: dict) -> dict:
    out = {}
    for f in _expand(specs, "*c3a-*.json"):
        m = re.search(r"c3a-(gsm8k|mbpp)\.json$", f.name)
        if not m:
            continue
        d = _read_json(f, warnings)
        if not isinstance(d, dict):
            continue
        inputs[f"calibrator:{m.group(1)}"] = {"path": str(f), "sha256": sha256_file(f)}
        ece = (d.get("dev_check") or {}).get("ece") or {}
        out[m.group(1)] = {"fit_docs": d.get("n_fit_docs"), "monotone_share": d.get("monotone_share"),
                           "dev_ece_raw": (ece.get("raw") or {}).get("value"),
                           "dev_ece_cal": (ece.get("calibrated") or {}).get("value")}
    return out


def load_sanity(path, warnings: list, inputs: dict):
    if not path or not Path(path).is_file():
        return None
    d = _read_json(Path(path), warnings)
    if not isinstance(d, dict):
        return None
    inputs["sanity"] = {"path": str(path), "sha256": sha256_file(path)}
    if "provenance" not in d:
        warnings.append(f"{path} carries no provenance stamp")
    rows = {}
    for r in d.get("rows", []):
        if r.get("task") == "gsm8k" and r.get("config") in SANITY_DEC \
                and r.get("prompt_format") in (None, "lmeval"):
            lo, hi = r.get("ci95") or [None, None]
            rows[r["config"]] = {"n": r["n"], "n_correct": r.get("n_correct"), "acc": r["accuracy"],
                                 "ci95": [lo, hi], "published": PUBLISHED[r["config"]]}
    if not rows:
        return None
    return {"file": str(path), "provenance": d.get("provenance"), "rows": rows,
            "n": max(r["n"] for r in rows.values()), "published": dict(PUBLISHED)}


# --------------------------------------------------------------------------- per-task statistics
def config_stats(recs: dict, gen_length: int, timeouts: frozenset = frozenset()) -> dict:
    """Speed, cost and failures of one configuration from its ``{doc_id: record}``.  ``failures``
    are driver errors; ``n_failed_any`` also counts problems that never stopped (no stop string in
    256 tokens) or whose program timed out when scored."""
    ok = [r for r in recs.values() if r.get("error") is None and r.get("completion") is not None
          and _finite(r.get("n_forward")) and r["n_forward"] > 0]
    nfe = [r["n_forward"] for r in ok]
    secs = [r["seconds"] for r in ok if _finite(r.get("seconds"))]
    bad = {d for d, r in recs.items() if r.get("error") is not None or r.get("completion") is None
           or r.get("stopped") is False} | (set(timeouts) & set(recs))
    return {"n": len(recs), "n_ok": len(ok), "failures": len(recs) - len(ok),
            "n_unstopped": sum(r.get("stopped") is False for r in recs.values()),
            "n_timeout": len(set(timeouts) & set(recs)), "n_failed_any": len(bad),
            "tps": aggregate_speed(nfe, gen_length),
            "mean_of_ratios": float(np.mean([gen_length / n for n in nfe])) if nfe else math.nan,
            "nfe": float(np.mean(nfe)) if nfe else math.nan,
            "seconds": float(np.mean(secs)) if secs else math.nan}


def outside_share(recs_by_config: dict, keys: list) -> dict | None:
    steps = out = 0
    for k in keys:
        for r in recs_by_config.get(k, {}).values():
            for s in r.get("c3a_steps") or []:
                steps += 1
                out += bool(s.get("outside"))
    return {"n_steps": steps, "n_outside": out, "share": out / steps} if steps else None


def build_task(task, recs, verdicts, scores, grid, warnings) -> dict:
    gen_length = (recs or {}).get("_gen_length", {}).get(task, GEN_LENGTH) if recs else GEN_LENGTH
    by_cfg = (recs or {}).get(task, {})
    t = {"n": None, "decoders": {}, "matched": {}, "alpha": {}, "outside_share": None}
    if by_cfg:
        sizes = {len(v) for v in by_cfg.values()}
        t["n"] = max(sizes)
        if len(sizes) > 1:
            warnings.append(f"{task}: configurations have different numbers of problems {sorted(sizes)}")
    correct = {}                                          # (dec, slot) -> {doc_id: 0/1}
    g = grid.get(task) if grid else None
    if g:
        t["alpha"] = {str(s): a for s, a in sorted(g["alpha"].items())}
    for key, docs in by_cfg.items():
        dec, slot = (g or {}).get("dec_of", {}).get(key), (g or {}).get("slot_of", {}).get(key)
        if dec is None or slot is None:
            if g:
                warnings.append(f"{task}: record configuration {key} is not in the grid; ignored")
            continue
        st = config_stats(docs, gen_length, (getattr(verdicts, 'timeouts', {}) or {}).get((task, key), frozenset()))
        st["config"] = key
        vmap = {d: v for d, v in (verdicts.get((task, key)) or {}).items() if d in docs}
        if verdicts:
            for d, r in docs.items():                    # a failed record with no verdict is wrong
                if d not in vmap and (r.get("error") is not None or r.get("completion") is None):
                    vmap[d] = 0
            missing = len(docs) - len(vmap)
            if vmap and missing:
                warnings.append(f"{task} {key}: {missing} record(s) without a verdict; left out of the accuracy")
        row = (scores or {}).get((task, key))
        if vmap:
            k, n = sum(vmap.values()), len(vmap)
            lo, hi = wilson(k, n)
            st.update({"n_correct": k, "n_scored": n, "acc": k / n, "ci95": [lo, hi]})
            correct[(dec, slot)] = vmap
        elif row and row.get("n"):
            st.update({"n_correct": row.get("n_correct"), "n_scored": row["n"], "acc": row["accuracy"],
                       "ci95": list(row["ci95"])})
        t["decoders"].setdefault(dec, {"speeds": {}})["speeds"][str(slot)] = st
    # matching
    unmatched = (g or {}).get("unmatched", {})
    for s in SLOTS:
        c3a = t["decoders"].get("CThreeA", {}).get("speeds", {}).get(str(s))
        per = {}
        for dec in DECODERS:
            if dec == "CThreeA":
                continue
            speeds = t["decoders"].get(dec, {}).get("speeds", {})
            settings = {int(sl): {"tps": st["tps"], "unmatched": unmatched.get((dec, int(sl)), False)}
                        for sl, st in speeds.items()}
            # a slot the grid flags with no record still blocks: put it in as unmatched
            if unmatched.get((dec, s)) and s not in settings:
                settings[s] = {"tps": math.nan, "unmatched": True}
            c_tps = c3a["tps"] if c3a else math.nan
            m = match_baseline(s, c_tps, settings)
            if m is None:
                if not c3a or not _finite(c_tps):
                    reason = "no_c3a"
                elif unmatched.get((dec, s)):
                    reason = "unmatched_at_target"
                elif not any(_finite(v["tps"]) for v in settings.values()):
                    reason = "no_records"
                else:
                    reason = "outside_tolerance"
                per[dec] = {"matched": False, "reason": reason}
                continue
            e = {"matched": True, "reason": None, "slot": m["slot"], "tps": m["tps"],
                 "rel_diff": m["rel_diff"], "diff": None, "lo": None, "hi": None, "n_paired": 0}
            a, b = correct.get(("CThreeA", s)), correct.get((dec, m["slot"]))
            if a and b:
                r = paired_bootstrap(a, b)
                if r["n"]:
                    e.update({"diff": r["diff"], "lo": r["lo"], "hi": r["hi"], "n_paired": r["n"]})
            per[dec] = e
        t["matched"][str(s)] = per
    steps_keys = [k for k, d in (g or {}).get("dec_of", {}).items() if d == "CThreeA"] or \
        [k for k in by_cfg if k.startswith("c3a[")]
    t["outside_share"] = outside_share(by_cfg, steps_keys)
    if t["outside_share"] and g:
        per_slot = {}
        for k in steps_keys:
            o = outside_share(by_cfg, [k])
            if o:
                per_slot[str(g["slot_of"].get(k))] = o
        t["outside_share"]["per_slot"] = per_slot
    return t


# --------------------------------------------------------------------------- macros
def build_macros(S: dict) -> Macros:
    m = Macros()
    m.comment("Generated by scripts/c3_paper_outputs.py -- do not edit by hand.")
    m.comment("Names: paper/sections/06_c3.tex.  Fractions on the 0-1 scale, 3 decimals; TPS and Sec 2,")
    m.comment("NFE 1; N and FitDocs integers.  --- = a comparison with no match (no setting within 10%,")
    m.comment("or KLASS unmatched at the target).  A macro that is not defined has no input yet.")
    for task in TASKS:
        w = TASK_WORD[task]
        c = S["calibrators"].get(task)
        if c:
            if isinstance(c["fit_docs"], int):
                m.add(f"CThreeTEleven{w}FitDocs", str(c["fit_docs"]))
            for name, key in (("MonotoneShare", "monotone_share"), ("DevECERaw", "dev_ece_raw"),
                              ("DevECECal", "dev_ece_cal")):
                if _finite(c[key]):
                    m.add(f"CThreeTEleven{w}{name}", fnum(c[key], 3))
    for task in TASKS:
        w, t = TASK_WORD[task], S["tasks"][task]
        if t["n"]:
            m.add(f"CThreeTTwelve{w}N", str(t["n"]))
        for dec in DECODERS:
            for s in SLOTS:
                st = t["decoders"].get(dec, {}).get("speeds", {}).get(str(s))
                if not st:
                    continue
                base = f"CThreeTTwelve{w}{dec}{SPEED_WORD[s]}"
                if _finite(st.get("acc")):
                    m.add(base + "Acc", fnum(st["acc"], 3))
                    lo, hi = st["ci95"]
                    if _finite(lo) and _finite(hi):
                        m.add(base + "AccLo", fnum(lo, 3))
                        m.add(base + "AccHi", fnum(hi, 3))
                if _finite(st["tps"]):
                    m.add(base + "TPS", fnum(st["tps"], 2))
                if _finite(st["nfe"]):
                    m.add(base + "NFE", fnum(st["nfe"], 1))
                if _finite(st["seconds"]):
                    m.add(base + "Sec", fnum(st["seconds"], 2))
        for s in SLOTS:
            for dec, e in (t["matched"].get(str(s)) or {}).items():
                base = f"CThreeTTwelve{w}MatchedDiff{dec}{SPEED_WORD[s]}"
                if e["matched"] and e["diff"] is not None:
                    m.add(base, fnum(e["diff"], 3))
                    m.add(base + "Lo", fnum(e["lo"], 3))
                    m.add(base + "Hi", fnum(e["hi"], 3))
                elif not e["matched"] and e["reason"] in ("unmatched_at_target", "outside_tolerance"):
                    for suffix in ("", "Lo", "Hi"):
                        m.add(base + suffix, DASH)
        if t["outside_share"]:
            m.add(f"CThreeTTwelve{w}OutsideShare", fnum(t["outside_share"]["share"], 3))
        for s, a in t["alpha"].items():
            m.add(f"CThreeTTwelve{w}CThreeA{SPEED_WORD[int(s)]}Alpha", fnum(a, 3))
        if str(PRIMARY_SPEED) in t["alpha"]:
            m.add(f"CThreeTTwelve{w}AlphaPrimary", fnum(t["alpha"][str(PRIMARY_SPEED)], 3))
    san = S["sanity"]
    if san:
        m.add("CThreeTTwelveSanityN", str(san["n"]))
        for key, dec in SANITY_DEC.items():
            r = san["rows"].get(key)
            if r:
                m.add(f"CThreeTTwelveSanity{dec}Acc", fnum(r["acc"], 3))
                if all(_finite(x) for x in r["ci95"]):
                    m.add(f"CThreeTTwelveSanity{dec}AccLo", fnum(r["ci95"][0], 3))
                    m.add(f"CThreeTTwelveSanity{dec}AccHi", fnum(r["ci95"][1], 3))
    return m


def write_macros(m: Macros, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(m.lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- tables
HEADER = ("% Generated by scripts/c3_paper_outputs.py from {src} -- do not edit by hand.\n"
          "% Needs booktabs.  --- is an em dash: no setting within 10% of the reference speed.\n")
OPEN = "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
CLOSE = "\\endgroup\n"


def _est(acc, ci) -> str:
    if not _finite(acc):
        return DASH
    return est_ci({"value": acc, "ci": list(ci) if ci and all(_finite(x) for x in ci) else None})


def matched_table(S: dict) -> str:
    """Sec. 6's table: per task, one row per decoder at the setting matched to C3a's speed-2 slot."""
    s = PRIMARY_SPEED
    rows = [HEADER.format(src="the T12 report-half records, verdicts and grids"), OPEN,
            "\\begin{tabular}{lrll}\n\\toprule\n",
            "Decoder & tok/step & Acc.\\ [95\\% CI] & $\\Delta$ vs C3a [95\\% CI] \\\\\n"]
    for task in TASKS:
        t = S["tasks"][task]
        head = f"{TASK_LABEL[task]}" + (f" ($N={t['n']}$)" if t["n"] else "")
        rows.append("\\midrule\n")
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{{head}}}}} \\\\\n")
        c3a = t["decoders"].get("CThreeA", {}).get("speeds", {}).get(str(s))
        for dec in DECODERS:
            if dec == "CThreeA":
                if c3a and _finite(c3a["tps"]):
                    rows.append(" & ".join([LABEL[dec], num(c3a["tps"], 2), _est(c3a.get("acc"), c3a.get("ci95")),
                                            "ref."]) + " \\\\\n")
                else:
                    rows.append(" & ".join([LABEL[dec], DASH, DASH, DASH]) + " \\\\\n")
                continue
            e = (t["matched"].get(str(s)) or {}).get(dec)
            if not e or not e["matched"]:
                rows.append(" & ".join([LABEL[dec], DASH, DASH, DASH]) + " \\\\\n")
                continue
            st = t["decoders"][dec]["speeds"][str(e["slot"])]
            diff = DASH if e["diff"] is None else est_ci({"value": e["diff"], "ci": [e["lo"], e["hi"]]})
            rows.append(" & ".join([LABEL[dec], num(e["tps"], 2), _est(st.get("acc"), st.get("ci95")), diff])
                        + " \\\\\n")
    rows += ["\\bottomrule\n\\end{tabular}\n", CLOSE]
    return "".join(rows)


def sanity_cost_table(S: dict) -> str:
    """The sanity rows and the cost (tokens per step, forward passes, seconds per problem) per
    decoder and speed, for the slides and the code archive's README."""
    rows = [HEADER.format(src="the Ada sanity run and the T12 report-half records"), OPEN,
            "\\begin{tabular}{l" + "rrr" * 3 + "r}\n\\toprule\n"]
    san = S["sanity"]
    rows.append("\\multicolumn{11}{l}{\\textbf{Sanity: GSM8K fit half"
                + (f", $n={san['n']}$" if san else "") + ", 5-shot, Fast-dLLM settings}} \\\\\n")
    for key, dec in SANITY_DEC.items():
        r = (san or {}).get("rows", {}).get(key)
        ours = _est(r["acc"], r["ci95"]) if r else DASH
        rows.append(f"\\multicolumn{{4}}{{l}}{{{LABEL[dec]} ({tex_cfg(key)})}} & "
                    f"\\multicolumn{{4}}{{l}}{{ours {ours}}} & "
                    f"\\multicolumn{{3}}{{l}}{{published ${PUBLISHED[key]:.3f}$}} \\\\\n")
    rows.append("\\midrule\n")
    rows.append("Decoder & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{speed {s}}}" for s in SLOTS) + " & fail \\\\\n")
    rows.append(" & " + " & ".join("tok/st & NFE & s" for _ in SLOTS) + " & \\\\\n")
    for task in TASKS:
        t = S["tasks"][task]
        rows.append("\\midrule\n")
        rows.append(f"\\multicolumn{{11}}{{l}}{{\\textbf{{{TASK_LABEL[task]}}}"
                    + (f" ($N={t['n']}$)" if t["n"] else "") + "} \\\\\n")
        for dec in DECODERS:
            cells, fails = [], 0
            for s in SLOTS:
                st = t["decoders"].get(dec, {}).get("speeds", {}).get(str(s))
                if st:
                    cells += [_plain(st["tps"], 2), _plain(st["nfe"], 1), _plain(st["seconds"], 2)]
                    fails += st["n_failed_any"]
                else:
                    cells += [DASH] * 3
            have = t["decoders"].get(dec)
            rows.append(" & ".join([LABEL[dec]] + cells + [str(fails) if have else DASH]) + " \\\\\n")
        o = t["outside_share"]
        rows.append("\\multicolumn{11}{l}{C3a steps outside the fit range: "
                    + (num(o["share"], 3) if o else DASH) + "} \\\\\n")
    rows += ["\\bottomrule\n\\end{tabular}\n", CLOSE]
    return "".join(rows)


def _plain(x, nd: int) -> str:
    """A cost number in math mode, never in scientific notation (NFE reaches 128)."""
    return f"${fnum(x, nd)}$" if _finite(x) else DASH


def tex_cfg(key: str) -> str:
    return {"topk_conf[k=1]": "$k{=}1$", "threshold[tau=0.9]": "$\\tau{=}0.9$"}.get(key, key)


# --------------------------------------------------------------------------- zip tables
def _fmt(x, nd=3) -> str:
    return "" if not _finite(x) else f"{x:.{nd}f}"


def _csv(path: Path, head: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(head)
        w.writerows(rows)


def _md(head: list, rows: list) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) if c != "" else DASH for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write_zip_tables(S: dict, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    san = S["sanity"]
    srows = []
    for key, dec in SANITY_DEC.items():
        r = (san or {}).get("rows", {}).get(key)
        srows.append(["gsm8k", dec, key, r["n"] if r else "", _fmt(r["acc"]) if r else "",
                      _fmt(r["ci95"][0]) if r else "", _fmt(r["ci95"][1]) if r else "",
                      f"{PUBLISHED[key]:.3f}", _fmt(r["acc"] - PUBLISHED[key]) if r else ""])
    shead = ["task", "decoder", "config", "n", "acc", "acc_lo", "acc_hi", "published", "acc_minus_published"]
    _csv(d / "t12_sanity.csv", shead, srows)
    chead = ["task", "decoder", "speed", "config", "n", "failures", "unstopped", "timeouts", "failed_any", "tokens_per_step", "nfe",
             "seconds_per_problem", "outside_share"]
    crows = []
    for task in TASKS:
        t = S["tasks"][task]
        for dec in DECODERS:
            for s in SLOTS:
                st = t["decoders"].get(dec, {}).get("speeds", {}).get(str(s))
                if st:
                    o = t["outside_share"]["per_slot"].get(str(s)) if (dec == "CThreeA" and t["outside_share"]) else None
                    crows.append([task, dec, s, st["config"], st["n"], st["failures"], st["n_unstopped"],
                                  st["n_timeout"], st["n_failed_any"], _fmt(st["tps"], 3),
                                  _fmt(st["nfe"], 2), _fmt(st["seconds"], 3), _fmt(o["share"]) if o else ""])
    _csv(d / "t12_sanity_cost.csv", chead, crows)
    md = ["# T12 sanity check and cost\n",
          "Sanity: the Ada fit-half GSM8K run (5-shot, Fast-dLLM settings, fp16, 2080 Ti) against the "
          "published LLaDA-8B-Instruct numbers (0.793 top-k k=1; 0.792 threshold tau=0.9); 95% Wilson "
          "intervals.\n", _md(shead, srows),
          "\nCost, report halves (zero-shot chat, 256 tokens, blocks of 32, greedy): tokens per step = "
          "256 N / sum NFE; NFE and seconds are per-problem means over problems that decoded; failures = "
          "problems the driver could not decode (scored wrong); unstopped = no stop string within 256 tokens; "
          "timeouts = programs that timed out when scored; failed_any = problems with any of the three.\n", _md(chead, crows)]
    (d / "t12_sanity_cost.md").write_text("\n".join(md), encoding="utf-8")

    hhead = ["task", "decoder", "speed", "config", "tokens_per_step", "acc", "acc_lo", "acc_hi",
             "matched_slot", "matched_tps", "matched_diff", "diff_lo", "diff_hi", "n_paired"]
    hrows = []
    for task in TASKS:
        t = S["tasks"][task]
        for s in (1, 3):
            for dec in DECODERS:
                st = t["decoders"].get(dec, {}).get("speeds", {}).get(str(s))
                if not st:
                    continue
                e = None if dec == "CThreeA" else (t["matched"].get(str(s)) or {}).get(dec)
                lo, hi = st.get("ci95") or (None, None)
                hrows.append([task, dec, s, st["config"], _fmt(st["tps"], 3), _fmt(st.get("acc")), _fmt(lo), _fmt(hi),
                              e["slot"] if e and e["matched"] else "", _fmt(e["tps"], 3) if e and e["matched"] else "",
                              _fmt(e["diff"]) if e and e["matched"] and e["diff"] is not None else "",
                              _fmt(e["lo"]) if e and e["matched"] and e["lo"] is not None else "",
                              _fmt(e["hi"]) if e and e["matched"] and e["hi"] is not None else "",
                              e["n_paired"] if e and e["matched"] else ""])
    _csv(d / "t12_speed1_speed3.csv", hhead, hrows)
    (d / "t12_speed1_speed3.md").write_text(
        "# T12 at speeds 1 and 3\n\nPer decoder: accuracy at its own speed-s setting (95% Wilson), and, "
        "when matched, the paired accuracy difference C3a minus the decoder at the setting nearest to "
        "C3a's slot s (within 10%; matched_slot / matched_tps), 95% percentile bootstrap CI (2,000 "
        "resamples of problems, seed 20261030). A blank is '---': no setting within 10%, or KLASS "
        "unmatched at the target.\n\n" + _md(hhead, hrows), encoding="utf-8")


# --------------------------------------------------------------------------- figure
COLORS = {"TopK": "#0072b2", "Threshold": "#009e73", "Margin": "#56b4e9", "Entropy": "#cc79a7",
          "Klass": "#e69f00", "PcSampler": "#7f7f7f", "CThreeA": "#d55e00"}
MARKERS = {"TopK": "o", "Threshold": "s", "Margin": "^", "Entropy": "v", "Klass": "D", "PcSampler": "P",
           "CThreeA": "*"}


def make_figure(S: dict, out: Path) -> list[Path]:
    """Accuracy against aggregate tokens per step, one panel per task, one line per decoder across
    its three speeds, Wilson error bars; the x axes are not shared."""
    if not any(st for t in S["tasks"].values() for d in t["decoders"].values() for st in d["speeds"].values()
               if _finite(st.get("acc"))):
        return []
    try:
        import figures_t9 as F
        plt = F._plt()
    except ImportError as exc:
        print(f"[c3-paper] figure skipped: {exc}", file=sys.stderr)
        return []
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig, axes = plt.subplots(1, 2, figsize=(F.COLUMN_IN, 2.35), sharex=False)
    for ax, task in zip(axes, TASKS):
        F._style(ax)
        xs_all = []
        for dec in DECODERS:
            pts = sorted((st["tps"], st["acc"], st["ci95"]) for st in
                         t_speeds(S, task, dec) if _finite(st.get("acc")) and _finite(st["tps"]))
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            err = [[max(0.0, p[1] - p[2][0]) for p in pts], [max(0.0, p[2][1] - p[1]) for p in pts]]
            c3a = dec == "CThreeA"
            ax.errorbar(x, y, yerr=err, color=COLORS[dec], marker=MARKERS[dec], ms=4.5 if c3a else 3,
                        lw=1.4 if c3a else 0.8, elinewidth=0.6, capsize=1.2, zorder=3 if c3a else 2,
                        label=PLAIN[dec] if dec != "CThreeA" else "C3a")
            xs_all += x
        ax.set_title(TASK_LABEL[task], loc="left")
        ax.set_xlabel("tokens per step")
        ax.set_xscale("log", base=2)
        if xs_all:
            ticks = [t for t in (1, 2, 4, 8, 16, 32) if min(xs_all) / 1.5 <= t <= max(xs_all) * 1.5]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel("accuracy")
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, handlelength=1.6, columnspacing=1.0,
                   bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.2)
    out.mkdir(parents=True, exist_ok=True)
    paths = F._save(fig, out, "t12_matched_speed", "T12: accuracy against tokens per step")
    plt.close(fig)
    return paths


def t_speeds(S, task, dec):
    return list(S["tasks"][task]["decoders"].get(dec, {}).get("speeds", {}).values())


# --------------------------------------------------------------------------- driver
def jsonable(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k)): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="*", default=[], help="run_t12.py shard files, directories or globs")
    ap.add_argument("--verdicts", nargs="*", default=[], help="score_t12.py --per-record JSONL")
    ap.add_argument("--scores", nargs="*", default=[], help="score_t12.py --out JSON (fallback accuracy)")
    ap.add_argument("--grids", nargs="*", default=["prereg"], help="t12_grid_<task>.json files or a directory")
    ap.add_argument("--calibrators", nargs="*", default=["artifacts/c3"],
                    help="llada-8b-instruct_fp16_c3a-<task>.json files or a directory")
    ap.add_argument("--sanity", default="artifacts/t12/llada-8b-instruct_fp16_gsm8k-sanity.json")
    ap.add_argument("--out-dir", default="artifacts", help="artifacts root: c3/ and t12/ below it")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)

    warnings, inputs = [], {}
    recs = load_records(a.records, warnings, inputs) if a.records else {}
    verdicts = load_verdicts(a.verdicts, warnings, inputs) if a.verdicts else {}
    scores = load_scores(a.scores, warnings, inputs) if a.scores else {}
    grid = load_grids(a.grids, warnings, inputs)
    cal = load_calibrators(a.calibrators, warnings, inputs)
    sanity = load_sanity(a.sanity, warnings, inputs)

    missing = []
    if not any(k.startswith("records:") for k in inputs):
        missing.append("records")
    if not any(k.startswith("verdicts:") for k in inputs) and not any(k.startswith("scores:") for k in inputs):
        missing.append("verdicts")
    missing += [f"grids ({t})" for t in TASKS if t not in grid]
    missing += [f"calibrators ({t})" for t in TASKS if t not in cal]
    if sanity is None:
        missing.append("sanity")

    S = {"what": "T12 summary for the paper (scripts/c3_paper_outputs.py): LLaDA-8B-Instruct fp16, "
                 "report halves of GSM8K and MBPP, zero-shot chat, 256 tokens, blocks of 32, greedy",
         "primary_speed": PRIMARY_SPEED,
         "pairing": {"rule": RULE, "seed": SEED, "n_boot": N_BOOT, "tolerance": TOL,
                     "speed_metric": "256 N / sum(NFE) over the task's report-half problems",
                     "deviation": "prereg/deviations.md entry 17"},
         "tasks": {t: build_task(t, recs, verdicts, scores, grid, warnings) for t in TASKS},
         "calibrators": cal, "sanity": sanity,
         "inputs": inputs, "missing_inputs": missing, "warnings": warnings}
    S["provenance"] = provenance()

    out = Path(a.out_dir)
    t12 = out / "t12"
    t12.mkdir(parents=True, exist_ok=True)
    macros = build_macros(S)
    write_macros(macros, out / "c3" / "macros_c3.tex")
    (t12 / "t12_matched_speed.tex").write_text(matched_table(S), encoding="utf-8")
    (t12 / "t12_sanity_cost.tex").write_text(sanity_cost_table(S), encoding="utf-8")
    write_zip_tables(S, t12 / "zip")
    figs = [] if a.no_figures else make_figure(S, t12 / "figures")
    S["outputs"] = sorted([str(out / "c3" / "macros_c3.tex"), str(t12 / "t12_matched_speed.tex"),
                           str(t12 / "t12_sanity_cost.tex")] + [str(p) for p in figs])
    S["macro_names"] = sorted(macros.names)
    (t12 / f"{MODEL_TAG}_t12-summary.json").write_text(json.dumps(jsonable(S), indent=2, default=str),
                                                         encoding="utf-8")
    n = {t: S["tasks"][t]["n"] for t in TASKS}
    print(f"[c3-paper] {len(macros.names)} macros, {len(figs)} figure files -> {out} "
          f"(report problems: {n['gsm8k'] or DASH} GSM8K, {n['mbpp'] or DASH} MBPP)")
    if missing:
        print(f"[c3-paper] no input for: {', '.join(missing)} (their cells are {DASH}, their macros undefined)",
              file=sys.stderr)
    for w in warnings:
        print(f"[c3-paper] warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
