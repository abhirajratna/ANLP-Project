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
                its row (n, n_correct, ci95) gives the accuracy; differences need the verdicts.  Not used
                for a task from which problems were dropped (its rows cover the dropped problems too).
  --grids       ``prereg/t12_grid_{gsm8k,mbpp}.json`` (``scripts/pick_knobs.py``): which configuration
                is which decoder at which speed slot, C3a's alpha per slot, and ``picks[*]
                .unmatched_at_target``.
  --calibrators ``artifacts/c3/llada-8b-instruct_fp16_c3a-{gsm8k,mbpp}.json`` (T11).
  --sanity      the Ada sanity run, ``score_t12.py`` output.

Completeness rule (prereg/deviations.md, entry 17), the only thing ``--allow-partial`` changes
  Per task the kept problems are the report-half doc_ids that have a record for EVERY configuration of
  the task's committed grid (an ``error`` record counts as finished; a driver error is scored wrong).
  Records of every other problem are dropped before anything is computed.  Accuracy, speed, matching,
  the paired bootstrap, OutsideShare, every macro, table, the figure, the zip tables and the summary
  use only the kept problems; ``N`` in every header is ``n_complete``.  Without ``--allow-partial`` an
  incomplete run (a shard missing, a configuration short of problems) is refused instead.  Per task the
  summary records n_planned, n_complete, n_dropped_problems, n_dropped_records and the rule text.
  Sanity: within a shard file the kept problems should be a prefix of its problem order (run_t12.py
  runs problem by problem); if not, a loud warning goes to stderr, the macros file and the summary.

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

Sensitivity mode (``--allow-mixed-machines --tag TAG``; never for the paper)
  Deviation 17 (PR #74): a task whose records come from more than one run / machine / output directory is
  refused, because the paper's numbers are one run on one machine.  A sensitivity check (MBPP's completion run to
  250 problems, split over two H200 instances) may merge them.  Each such condition then becomes a warning and is
  recorded in the summary as ``mixed_machines`` (with ``sensitivity: true`` and ``sensitivity_tag``); every other
  guard stays on.  EVERY output goes under ``<out-dir>/t12/sensitivity/<TAG>/`` (macros as
  ``macros_c3_sensitivity.tex`` with the prefix ``CThreeTTwelveSens``, so an ``\\input`` cannot redefine the
  paper's macros); the run refuses if an output path would be outside that directory.
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob
import hashlib
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
from c1_paper_outputs import DASH, Macros, ci_decimals, est_ci, num  # noqa: E402

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

# score_t12.RUN_FIELDS: what makes shards one run
RUN_FIELDS = ("model", "model_revision", "precision", "task", "task_jsonl_sha256",
              "manifest_sha256", "half", "gen_length", "block_length", "decoding", "prompt",
              "configs_sha256", "calibrator_sha256", "c3a_clamp", "pc_sampler_variant")
try:
    from score_t12 import RUN_FIELDS                      # noqa: E402,F811
except Exception:                                         # pragma: no cover
    pass
DASH_MACRO = "\\textnormal{---}"          # an em dash in text; a bare --- is a minus inside \\CresCI
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
# (decoder, slot) pairs reported as computed but off the decoder's published operating point (daggered)
OFF_OPERATING_POINT = frozenset({("Klass", 2), ("Klass", 3)})   # deviation 17 / PR #66: tau below published range
OFF_OP_NOTE = "KLASS slots 2-3: confidence threshold below its published range (deviation 17, PR #66)"
DAGGER = "$^\\dagger$"
MD_DAGGER = "†"
OFF_OP_MD = (f"{MD_DAGGER} = off the decoder's published operating point ({OFF_OP_NOTE}); "
             "the CSV has it as a boolean column.\n\n")
EXACT_K = {2: 1, 4: 2, 8: 3}                             # top-k, margin, PC-Sampler: k -> speed slot
# Fast-dLLM, Table 1 (LLaDA-8B-Instruct, GSM8K, 5-shot): design constants of prereg/t12_sanity.json
PUBLISHED = {"topk_conf[k=1]": 0.793, "threshold[tau=0.9]": 0.792}
SANITY_DEC = {"topk_conf[k=1]": "TopK", "threshold[tau=0.9]": "Threshold"}
RULE = ("each baseline is taken at its setting whose aggregate tokens per step (256 N / sum NFE over "
        "the task's report-half problems) is nearest to C3a's at that slot, within +-10%, else '---' "
        "(KLASS: '---' where the grid says unmatched_at_target); difference = C3a minus baseline "
        "accuracy, paired on the shared doc_ids, 95% percentile CI from 2,000 bootstrap resamples "
        "of problems, seed 20261030 (reseeded per comparison); on a tie (two settings equally near) the "
        "lower speed slot is taken")
COMPLETE_RULE = ("kept problems: the report-half problems with a record for every configuration of the task's "
                 "committed grid (an error record counts as finished); the records of every other problem are "
                 "dropped before accuracy, speed, matching and the paired bootstrap are computed, so all of them, "
                 "and N, use only the kept problems (prereg/deviations.md entry 17)")
MODEL_TAG = "llada-8b-instruct_fp16"


# --------------------------------------------------------------------------- small helpers
def off_op(dec: str, slot) -> bool:
    return (dec, int(slot)) in OFF_OPERATING_POINT


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
    {(decoder, slot): bool}, "alpha": {slot: alpha}, "keys": [every committed config key]}}``."""
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
        out[task] = {"slot_of": slot_of, "dec_of": dec_of, "unmatched": unmatched, "alpha": alpha,
                     "keys": sorted(set(keys))}
    return out


def run_key(meta: dict) -> str:
    """The run a shard belongs to: score_t12's hash of ``RUN_FIELDS`` (the shard number left out),
    so a verdict's ``run`` and a record's run agree."""
    ident = {f: meta.get(f) for f in RUN_FIELDS}
    return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _prompt_format(meta: dict):
    p = meta.get("prompt")
    return p.get("format") if isinstance(p, dict) else p


def load_manifest(manifest_dir, task: str, warnings: list, inputs: dict):
    """``(report doc_ids, fit doc_ids)`` of ``<manifest_dir>/split_manifest_<task>.json``, or None."""
    f = Path(manifest_dir) / f"split_manifest_{task}.json" if manifest_dir else None
    if f is None or not f.is_file():
        warnings.append(f"{task}: no split manifest ({f}); the fit-half check and the expected "
                        "problem count use the shard metas only")
        return None
    d = _read_json(f, warnings)
    if not isinstance(d, dict):
        return None
    inputs[f"manifest:{task}"] = {"path": str(f), "sha256": sha256_file(f)}
    return ({int(k) for k, v in d.items() if int(v) == 1}, {int(k) for k, v in d.items() if int(v) == 0})


def load_records(specs, warnings: list, inputs: dict, manifest_dir=None, allow_mixed: bool = False) -> dict:
    """The report-half records, ``{task: {config_key: {doc_id: record}}}``, and how complete they are.

    Refuses (``SystemExit``): a shard without a meta file, a report-half shard whose meta is not
    fp16 and the chat prompt, a doc_id that is not in the manifest's report half, more than one run
    for a task (deviation 17: one machine per task), and a shard whose meta says ``half: fit`` (the
    calibration and sanity runs).  Shortfalls (a shard missing, a configuration with
    fewer than the expected problems) go in ``out["_shortfalls"]`` for ``main`` to refuse or allow.
    ``out["_meta"][task]``: ``run``, ``n_expected``, ``shards_missing``.  ``out["_order"][task]``:
    shard file -> its doc_ids in order of first appearance.  The records are not yet reduced to the
    complete problems: ``reduce_to_complete`` does that.

    ``allow_mixed`` (sensitivity mode only): the two refusals about more than one run / host / gpu /
    num_shards / label / output directory per task become warnings, and the runs of a task are merged
    under the first run hash (``out["_meta"][task]["runs"]`` lists them all); what differed, per task, is
    ``out["_mixed"][task]`` (a list of ``{"field", "values"}``).  Every other refusal stays."""
    by_run = {}
    mixed = {}                                            # task -> field -> set of values seen (allow_mixed)                                           # task -> run -> {config: {doc: record}}
    info = {}                                             # task -> run -> shard bookkeeping
    gen_length, dup_diff, dup_same = {}, {}, {}
    first, origin = {}, {}                                # task -> identity of its first shard; (task, config, doc) -> file
    order = {}                                            # task -> shard file -> {doc_id: rank of first appearance}
    for f in _expand(specs, "t12.shard*.jsonl"):
        if f.name.endswith(".meta.json"):
            continue
        meta_path = f.with_name(f.name[:-len(".jsonl")] + ".meta.json")
        meta = _read_json(meta_path, warnings) if meta_path.exists() else None
        if not isinstance(meta, dict):
            raise SystemExit(f"{f}: no readable meta file ({meta_path.name}); cannot tell its half, "
                             "precision, prompt or run")
        if meta.get("half") != "report":
            raise SystemExit(f"{f}: meta half is {meta.get('half')!r}, not 'report'; a fit-half shard (a "
                             "calibration or sanity run) must not be under --records: pass the report run's directory")
        if meta.get("precision") != "fp16":
            raise SystemExit(f"{f}: meta precision is {meta.get('precision')!r}; T12 is fp16 (deviation 17)")
        if _prompt_format(meta) != "chat":
            raise SystemExit(f"{f}: meta prompt format is {_prompt_format(meta)!r}; the sweep uses the chat prompt")
        task = meta.get("task")
        if task not in TASKS:
            continue
        rk = run_key(meta)
        prov = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
        ident = {"run hash": rk, "num_shards": meta.get("num_shards"), "label": meta.get("label"),
                 "host": prov.get("host"), "gpu": prov.get("gpu"), "output directory": str(f.parent.resolve())}
        if task not in first:
            first[task] = (str(f), ident)
        else:
            diff = [k for k in ident if ident[k] != first[task][1][k]]
            if diff and allow_mixed:
                for k in diff:
                    mixed.setdefault(task, {}).setdefault(k, {first[task][1][k]}).add(ident[k])
            elif diff:
                raise SystemExit(
                    f"{task}: records of more than one run under --records: {first[task][0]} and {f} differ in "
                    + ", ".join(f"{k} ({first[task][1][k]!r} vs {ident[k]!r})" for k in diff)
                    + "; one run, one machine, one output directory per task (prereg/deviations.md 17): "
                      "pass the directory of one run")
        inputs[f"records:{f}"] = {"path": str(f), "sha256": sha256_file(f)}
        ri = info.setdefault(task, {}).setdefault(rk, {"shards": set(), "num_shards": set(), "doc_ids": set(),
                                                       "limit": set(), "files": []})
        ri["shards"].add(meta.get("shard"))
        ri["num_shards"].add(meta.get("num_shards"))
        ri["doc_ids"] |= {int(d) for d in meta.get("doc_ids") or []}
        ri["limit"].add(meta.get("limit") or 0)
        ri["files"].append(str(f))
        gen_length.setdefault(task, meta.get("gen_length"))
        cfgs = by_run.setdefault(task, {}).setdefault(rk, {})
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "config" not in r or "doc_id" not in r or r.get("task", task) != task:
                continue
            d = int(r["doc_id"])
            seen = order.setdefault(task, {}).setdefault(str(f), {})
            seen.setdefault(d, len(seen))
            old = cfgs.setdefault(r["config"], {}).get(d)
            if old is None:
                cfgs[r["config"]][d] = r
                origin[(task, r["config"], d)] = str(f)
            else:
                if origin[(task, r["config"], d)] != str(f):
                    raise SystemExit(f"{task}: ({r['config']}, doc {d}) is in two shard files, "
                                     f"{origin[(task, r['config'], d)]} and {f}: two runs or two copies of one "
                                     "shard; refusing to merge them")
                if old.get("completion") != r.get("completion"):
                    dup_diff[task] = dup_diff.get(task, 0) + 1
                else:
                    dup_same[task] = dup_same.get(task, 0) + 1
    out = {"_gen_length": {t: g or GEN_LENGTH for t, g in gen_length.items()}, "_meta": {}, "_shortfalls": [],
           "_order": {t: {f: sorted(ds, key=ds.get) for f, ds in files.items()} for t, files in order.items()},
           "_reduced": {}}
    if allow_mixed:
        out["_mixed"] = {t: [{"field": k, "values": sorted(str(v) for v in vals)} for k, vals in fields.items()]
                         for t, fields in mixed.items()}
    for task, runs in by_run.items():
        if len(runs) > 1 and allow_mixed:                 # merge under the first run hash; no (config, doc) is in two
            rks = sorted(runs)                            # files (refused above), so the union is a plain update
            rk0 = rks[0]
            for rk in rks[1:]:
                for cfg, docs in runs.pop(rk).items():
                    runs[rk0].setdefault(cfg, {}).update(docs)
                a, b = info[task][rk0], info[task].pop(rk)
                a["shards"] |= b["shards"]
                a["num_shards"] |= b["num_shards"]
                a["doc_ids"] |= b["doc_ids"]
                a["limit"] |= b["limit"]
                a["files"] += b["files"]
            info[task][rk0]["runs"] = rks
            warnings.append(f"{task}: MIXED RUNS merged (--allow-mixed-machines, sensitivity only): "
                            f"{len(rks)} run hashes {rks}")
        elif len(runs) > 1:
            where = "; ".join(f"{rk}: {info[task][rk]['files'][0]} (+{len(info[task][rk]['files']) - 1})"
                              for rk in sorted(runs))
            raise SystemExit(f"{task}: records of more than one run under --records ({where}); one machine "
                             "per task (prereg/deviations.md 17): pass the directory of one run")
        (rk, cfgs), = runs.items()
        ri = info[task][rk]
        out[task] = cfgs
        if dup_same.get(task):
            warnings.append(f"{task}: {dup_same[task]} identical repeated (config, doc_id) line(s) within a shard "
                            "file (a resumed run); counted once")
        if dup_diff.get(task):
            warnings.append(f"{task}: {dup_diff[task]} repeated (config, doc_id) with a different completion; "
                            "the first was used")
        man = load_manifest(manifest_dir, task, warnings, inputs)
        rec_ids = {d for docs in cfgs.values() for d in docs}
        if man:
            report, fit = man
            bad = (rec_ids | ri["doc_ids"]) - report
            if bad:
                raise SystemExit(f"{task}: doc_ids {sorted(bad)[:5]} are not in the report half of "
                                 f"split_manifest_{task}.json (fit-half ids: {sorted(bad & fit)[:5]}...); "
                                 "refusing a fit-half record")
            limit = max(ri["limit"]) if ri["limit"] else 0
            expected = set(sorted(report)[:limit]) if limit else set(report)
        else:
            expected = set(ri["doc_ids"])
            if rec_ids - expected:
                raise SystemExit(f"{task}: record doc_ids {sorted(rec_ids - expected)[:5]} are not in the metas' doc_ids")
        shard_n = {n for n in ri["num_shards"] if n}
        missing = []
        if len(shard_n) == 1:
            missing = sorted(set(range(shard_n.pop())) - {s for s in ri["shards"] if s is not None})
        out["_meta"][task] = {"run": rk, "n_expected": len(expected) or None, "shards_missing": missing,
                              "files": ri["files"]}
        if ri.get("runs"):
            out["_meta"][task]["runs"] = ri["runs"]
        if missing:
            out["_shortfalls"].append(f"{task}: shard(s) {missing} missing")
        short = {k: len(docs) for k, docs in cfgs.items() if len(docs) < len(expected)}
        if short:
            k, n = min(short.items(), key=lambda kv: kv[1])
            out["_shortfalls"].append(f"{task}: {len(short)} configuration(s) have fewer than the "
                                      f"{len(expected)} expected report problems; fewest {n}/{len(expected)} ({k})")
    for task, fields in sorted(mixed.items()):
        warnings.append(f"{task}: MIXED MACHINES (--allow-mixed-machines, sensitivity only): "
                        + "; ".join(f"{k}: {sorted(str(v) for v in vals)}" for k, vals in fields.items()))
    return out


def reduce_to_complete(recs: dict, grid: dict, warnings: list) -> dict:
    """The completeness rule (prereg/deviations.md 17), applied in place to ``recs``.

    Per task the kept doc_ids are those with a record for every configuration key of the task's
    committed grid (``grid[task]["keys"]``; with no grid, every configuration seen in the records, with
    a warning).  An ``error`` record counts as finished.  The records of every other doc_id are removed
    from ``recs[task]`` (and a configuration left with none), so nothing downstream can see them.
    Returns ``{task: info}`` (also in ``recs["_reduced"]``): ``n_planned``, ``n_complete``,
    ``n_dropped_problems``, ``n_dropped_records``, ``n_configs_required``, ``missing_configs``,
    ``prefix_check`` ``{"ok", "violations"}`` and ``rule``.  A grid configuration with no record at
    all makes every problem incomplete (``n_complete`` 0) and is added to ``recs["_shortfalls"]``."""
    info = {}
    for task in TASKS:
        cfgs = recs.get(task)
        if not cfgs:
            continue
        g = (grid or {}).get(task)
        required = list(g["keys"]) if g and g.get("keys") else sorted(cfgs)
        if not (g and g.get("keys")):
            warnings.append(f"{task}: no committed grid; the completeness rule uses the {len(required)} "
                            "configuration(s) seen in the records")
        all_docs = {d for docs in cfgs.values() for d in docs}
        missing_cfgs = [k for k in required if not cfgs.get(k)]
        kept = set() if missing_cfgs else set.intersection(*(set(cfgs[k]) for k in required))
        dropped = all_docs - kept
        n_dropped_records = sum(1 for docs in cfgs.values() for d in docs if d in dropped)
        recs[task] = {k: {d: r for d, r in docs.items() if d in kept} for k, docs in cfgs.items()
                      if any(d in kept for d in docs)}
        planned = (recs["_meta"].get(task) or {}).get("n_expected") or len(all_docs)
        violations = []
        for f, docs in (recs.get("_order") or {}).get(task, {}).items():
            flags = [d in kept for d in docs]
            n_prefix = 0
            while n_prefix < len(flags) and flags[n_prefix]:
                n_prefix += 1
            if any(flags[n_prefix:]):
                violations.append({"file": f, "n_problems": len(docs), "n_kept": sum(flags),
                                   "first_dropped": docs[n_prefix],
                                   "kept_after_gap": [d for d, k in zip(docs[n_prefix:], flags[n_prefix:]) if k][:10]})
        info[task] = {"n_planned": planned, "n_complete": len(kept), "n_dropped_problems": len(dropped),
                      "n_dropped_records": n_dropped_records, "n_configs_required": len(required),
                      "missing_configs": missing_cfgs, "rule": COMPLETE_RULE,
                      "prefix_check": {"ok": not violations, "violations": violations}}
        if missing_cfgs:
            recs["_shortfalls"].append(f"{task}: {len(missing_cfgs)} configuration(s) of the grid have no records "
                                       f"(first: {missing_cfgs[0]})")
        elif len(kept) < planned and not any(x.startswith(f"{task}:") for x in recs["_shortfalls"]):
            recs["_shortfalls"].append(f"{task}: only {len(kept)}/{planned} report problems have a record for "
                                       f"all {len(required)} configurations")
        if dropped:
            warnings.append(f"{task}: completeness rule: kept {len(kept)} of {planned} planned problems; dropped "
                            f"{len(dropped)} incomplete problem(s) = {n_dropped_records} record(s)")
        for v in violations:
            warnings.append(f"WARNING (NOT A PREFIX): {task}: in {v['file']} the kept problems are not a prefix of "
                            f"the shard's problem order: problem {v['first_dropped']} is incomplete but "
                            f"later problem(s) are complete ({v['kept_after_gap']}); a crash or a resume that "
                            "skipped problems? check before using these numbers")
    recs["_reduced"] = info
    return info


class Verdicts(dict):
    """``{(run, task, config): {doc_id: 0/1}}`` plus ``timeouts``: ``{(run, task, config): {doc_id}}``.
    ``run`` is score_t12's run hash, which is the records' ``run_key``."""

    def __init__(self):
        super().__init__()
        self.timeouts = {}


def load_verdicts(specs, warnings: list, inputs: dict, runs=None, canon=None) -> Verdicts:
    """Verdicts of ``score_t12.py --per-record``.  Lines that are not verdicts (no ``correct``: a record
    file) and lmeval verdicts (the sanity run's) are skipped.  Once the records are loaded (``runs``, their
    run hashes) a verdict must carry a ``run`` that is one of them: a run-less line (an older file) or
    another run's is ignored and counted in a warning.  ``canon`` (sensitivity mode, merged runs): run hash ->
    the run hash the merged records carry, so the verdicts of every merged run are found under it."""
    out = Verdicts()
    other = runless = 0
    for f in _expand(specs, "*.jsonl"):
        inputs[f"verdicts:{f}"] = {"path": str(f), "sha256": sha256_file(f)}
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                v = json.loads(line)
            except ValueError:
                continue
            if v.get("prompt_format") == "lmeval" or "doc_id" not in v or "config" not in v or "correct" not in v:
                continue
            if runs:
                if v.get("run") is None:
                    runless += 1
                    continue
                if v["run"] not in runs:
                    other += 1
                    continue
            key = ((canon or {}).get(v.get("run"), v.get("run")), v.get("task"), v["config"])
            out.setdefault(key, {}).setdefault(int(v["doc_id"]), int(bool(v.get("correct"))))
            if v.get("error") == "timeout":
                out.timeouts.setdefault(key, set()).add(int(v["doc_id"]))
    if other:
        warnings.append(f"{other} verdict(s) of another run ignored (their run is not the records' run)")
    if runless:
        warnings.append(f"{runless} verdict(s) without a run ignored (an older score_t12 file?): a verdict must "
                        "name the run of the records")
    return out


def load_scores(specs, warnings: list, inputs: dict, runs=None) -> dict:
    """``{(task, config): row}`` of the report half, chat prompt, from ``score_t12.py --out``."""
    out = {}
    for f in _expand(specs, "*.json"):
        d = _read_json(f, warnings)
        if not isinstance(d, dict):
            continue
        inputs[f"scores:{f}"] = {"path": str(f), "sha256": sha256_file(f)}
        for r in d.get("rows", []):
            if r.get("prompt_format") in (None, "chat") and r.get("half") in (None, "report") \
                    and not (runs and r.get("run") not in runs):
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
    t = {"n": None, "n_planned": None, "n_complete": None, "n_dropped_problems": None, "n_dropped_records": None,
         "rule": COMPLETE_RULE, "prefix_check": None, "partial": False, "shards_missing": [], "run": None,
         "decoders": {}, "matched": {}, "alpha": {}, "outside_share": None}
    rmeta = ((recs or {}).get("_meta") or {}).get(task) or {}
    red = ((recs or {}).get("_reduced") or {}).get(task)
    if red:                                               # N is the number of complete problems, everywhere
        t.update({k: red[k] for k in ("n_planned", "n_complete", "n_dropped_problems", "n_dropped_records",
                                      "prefix_check")})
        t["n"] = red["n_complete"]
        t["shards_missing"] = rmeta.get("shards_missing") or []
        t["run"] = rmeta.get("run")
        t["partial"] = bool(t["shards_missing"]) or red["n_complete"] < red["n_planned"]
    if by_cfg:
        sizes = {len(v) for v in by_cfg.values()}
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
        st = config_stats(docs, gen_length, (getattr(verdicts, "timeouts", {}) or {}).get((t["run"], task, key), frozenset()))
        st["config"] = key
        st["n_planned"] = t["n_planned"]
        run = t["run"]
        real = {d: v for d, v in (verdicts.get((run, task, key)) or {}).items() if d in docs}
        row = (scores or {}).get((task, key))
        if real:
            # a failed record with no verdict is wrong (score_t12 would have said so); only imputed
            # next to verdicts that are real, never as the whole accuracy
            vmap = dict(real)
            for d, r in docs.items():
                if d not in vmap and (r.get("error") is not None or r.get("completion") is None):
                    vmap[d] = 0
            if len(vmap) < len(docs):
                warnings.append(f"{task} {key}: {len(docs) - len(vmap)} of {len(docs)} record(s) without a "
                                "verdict; accuracy left blank")
            else:
                k, n = sum(vmap.values()), len(vmap)
                lo, hi = wilson(k, n)
                st.update({"n_correct": k, "n_scored": n, "acc": k / n, "ci95": [lo, hi]})
                correct[(dec, slot)] = vmap
        elif row and row.get("n") == len(docs) and not t["n_dropped_records"]:
            # an aggregate over the problems score_t12 saw: only when none were dropped here
            st.update({"n_correct": row.get("n_correct"), "n_scored": row["n"], "acc": row["accuracy"],
                       "ci95": list(row["ci95"])})
        elif verdicts or row:
            warnings.append(f"{task} {key}: no usable verdicts or score row for its {len(docs)} records; "
                            "accuracy left blank")
        if off_op(dec, slot):
            st["off_operating_point"] = True
        t["decoders"].setdefault(dec, {"speeds": {}})["speeds"][str(slot)] = st
    errs = [(st["config"], st["failures"]) for d in t["decoders"].values() for st in d["speeds"].values()
            if st["failures"]]
    if errs:
        warnings.append(f"{task}: driver errors, scored wrong; speed is over the problems that decoded: "
                        + ", ".join(f"{k} {n}" for k, n in errs))
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
            if off_op(dec, m["slot"]):
                e["off_operating_point"] = True
            a, b =correct.get(("CThreeA", s)), correct.get((dec, m["slot"]))
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
def est_macros(m: Macros, name: str, value, lo=None, hi=None) -> None:
    """``name``, ``name``Lo and ``name``Hi at the decimals ``est_ci`` prints in the tables
    (c1_paper_outputs.ci_decimals: 3, raised where rounding would show an interval end as 0.000 or
    [x, x]), so the prose and the table agree."""
    if not _finite(value):
        return
    has_ci = _finite(lo) and _finite(hi)
    nd = ci_decimals({"value": value, "ci": [lo, hi]}, 3) if has_ci else 3
    m.add(name, fnum(value, nd))
    if has_ci:
        m.add(name + "Lo", fnum(lo, nd))
        m.add(name + "Hi", fnum(hi, nd))


SENS_HEADER = ("SENSITIVITY OUTPUT (deviation 17, PR #74): records merged from more than one machine.  "
               "NOT FOR THE PAPER: do not \\input this file or copy a number from it into a table or the text.")


def build_macros(S: dict, prefix: str = "CThreeTTwelve", t11_prefix: str = "CThreeTEleven") -> Macros:
    m = Macros()
    if S.get("sensitivity"):
        m.comment(SENS_HEADER)
        m.comment(f"Tag {S['sensitivity_tag']}; macro prefixes {prefix} / {t11_prefix} "
                  "(the paper's are CThreeTTwelve / CThreeTEleven).")
    m.comment("Generated by scripts/c3_paper_outputs.py -- do not edit by hand.")
    m.comment("Names: paper/sections/06_c3.tex.  Fractions on the 0-1 scale, 3 decimals; TPS and Sec 2,")
    m.comment("NFE 1; N and FitDocs integers.  --- = a comparison with no match (no setting within 10%,")
    m.comment("or KLASS unmatched at the target), written \\textnormal{---}.  A macro that is not defined has")
    m.comment("no input yet.")
    for task in TASKS:
        if S["tasks"][task]["partial"]:
            t = S["tasks"][task]
            m.comment(f"PARTIAL: {task} N={t['n_complete']} of {t['n_planned']} planned (--allow-partial: only "
                      f"problems with every configuration; {t['n_dropped_problems']} problem(s) = "
                      f"{t['n_dropped_records']} record(s) dropped); shards missing: {t['shards_missing'] or 'none'}")
        for v in ((S["tasks"][task]["prefix_check"] or {}).get("violations") or []):
            m.comment(f"WARNING: {task}: kept problems are NOT a prefix of the problem order in "
                      f"{Path(v['file']).name} (problem {v['first_dropped']} incomplete, later ones complete)")
    for task in TASKS:
        w = TASK_WORD[task]
        c = S["calibrators"].get(task)
        if c:
            if isinstance(c["fit_docs"], int):
                m.add(f"{t11_prefix}{w}FitDocs", str(c["fit_docs"]))
            for name, key in (("MonotoneShare", "monotone_share"), ("DevECERaw", "dev_ece_raw"),
                              ("DevECECal", "dev_ece_cal")):
                if _finite(c[key]):
                    m.add(f"{t11_prefix}{w}{name}", fnum(c[key], 3))
    for task in TASKS:
        w, t = TASK_WORD[task], S["tasks"][task]
        if t["n"] is not None:
            m.add(f"{prefix}{w}N", str(t["n"]))
        for dec in DECODERS:
            for s in SLOTS:
                st = t["decoders"].get(dec, {}).get("speeds", {}).get(str(s))
                if not st:
                    continue
                base = f"{prefix}{w}{dec}{SPEED_WORD[s]}"
                if _finite(st.get("acc")):
                    est_macros(m, base + "Acc", st["acc"], *st["ci95"])
                if _finite(st["tps"]):
                    m.add(base + "TPS", fnum(st["tps"], 2))
                if _finite(st["nfe"]):
                    m.add(base + "NFE", fnum(st["nfe"], 1))
                if _finite(st["seconds"]):
                    m.add(base + "Sec", fnum(st["seconds"], 2))
        for s in SLOTS:
            for dec, e in (t["matched"].get(str(s)) or {}).items():
                base = f"{prefix}{w}MatchedDiff{dec}{SPEED_WORD[s]}"
                if e["matched"] and e["diff"] is not None:
                    est_macros(m, base, e["diff"], e["lo"], e["hi"])
                elif not e["matched"] and e["reason"] in ("unmatched_at_target", "outside_tolerance"):
                    for suffix in ("", "Lo", "Hi"):
                        m.add(base + suffix, DASH_MACRO)
        if t["outside_share"]:
            m.add(f"{prefix}{w}OutsideShare", fnum(t["outside_share"]["share"], 3))
        for s, a in t["alpha"].items():
            m.add(f"{prefix}{w}CThreeA{SPEED_WORD[int(s)]}Alpha", fnum(a, 3))
        if str(PRIMARY_SPEED) in t["alpha"]:
            m.add(f"{prefix}{w}AlphaPrimary", fnum(t["alpha"][str(PRIMARY_SPEED)], 3))
    san = S["sanity"]
    if san:
        m.add(f"{prefix}SanityN", str(san["n"]))
        for key, dec in SANITY_DEC.items():
            r = san["rows"].get(key)
            if r:
                est_macros(m, f"{prefix}Sanity{dec}Acc", r["acc"], *r["ci95"])
    return m


def write_macros(m: Macros, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(m.lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- tables
HEADER = ("% Generated by scripts/c3_paper_outputs.py from {src} -- do not edit by hand.\n"
          "% Needs booktabs.  --- is an em dash: no setting within 10% of the reference speed.\n")
OPEN = "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
CLOSE = "\\endgroup\n"
# the matched-speed table is tighter: Sec. 6 has ~0.15 page for it
OPEN_MATCHED = ("\\begingroup\\footnotesize\\setlength{\\tabcolsep}{2.5pt}\\renewcommand{\\arraystretch}{0.8}"
                "\\setlength{\\aboverulesep}{0.15ex}\\setlength{\\belowrulesep}{0.25ex}\n")


def _est(acc, ci) -> str:
    if not _finite(acc):
        return DASH
    return est_ci({"value": acc, "ci": list(ci) if ci and all(_finite(x) for x in ci) else None})


def nlabel(t: dict, tex: bool = True) -> str:
    """``$N=30$``: the number of complete problems (the problems every figure in the output is over)."""
    if t["n"] is None:
        return ""
    return f"$N={t['n']}$" if tex else f"N={t['n']}"


def matched_table(S: dict) -> str:
    """Sec. 6's table, compact: one row per decoder at the setting matched to C3a's speed-2 slot, the
    two tasks side by side (GSM8K | MBPP), each with tok/step, accuracy and C3a - decoder.  No ``n``
    column: ``N`` is in the task header, and a decoder whose difference is paired on fewer problems
    than ``N`` is named in a LaTeX comment.  Body: 2 header lines + one row per decoder."""
    s = PRIMARY_SPEED
    head = [HEADER.format(src="the T12 report-half records, verdicts and grids"), OPEN_MATCHED,
            "\\begin{tabular}{l" + "@{\\hspace{6pt}}".join(["rll"] * len(TASKS)) + "}\n\\toprule\n"]
    groups, rules = [], []
    for i, task in enumerate(TASKS):
        t = S["tasks"][task]
        label = TASK_LABEL[task] + (f" ({nlabel(t)})" if t["n"] is not None else "")
        groups.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{label}}}}}")
        rules.append(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}")
    head.append(" & ".join([""] + groups) + " \\\\\n" + "".join(rules) + "\n")
    head.append(" & ".join(["Decoder"] + ["tok/step & Acc.\\ [95\\% CI] & C3a $-$ decoder [95\\% CI]"] * len(TASKS))
                + " \\\\\n\\midrule\n")
    cells = {dec: [] for dec in DECODERS}
    notes, paired, daggered = [], [], set()
    for task in TASKS:
        t = S["tasks"][task]
        c3a = t["decoders"].get("CThreeA", {}).get("speeds", {}).get(str(s))
        for dec in DECODERS:
            shown = None
            if dec == "CThreeA":
                if c3a and _finite(c3a["tps"]):
                    cells[dec] += [num(c3a["tps"], 2), _est(c3a.get("acc"), c3a.get("ci95")), "ref."]
                    shown = c3a
                else:
                    cells[dec] += [DASH] * 3
            else:
                e = (t["matched"].get(str(s)) or {}).get(dec)
                if not e or not e["matched"]:
                    cells[dec] += [DASH] * 3
                    continue
                shown = t["decoders"][dec]["speeds"][str(e["slot"])]
                if off_op(dec, e["slot"]):                # the slot shown is the matched one, not nec. s
                    daggered.add(dec)
                diff = DASH if e["diff"] is None else est_ci({"value": e["diff"], "ci": [e["lo"], e["hi"]]})
                cells[dec] += [num(e["tps"], 2), _est(shown.get("acc"), shown.get("ci95")), diff]
                if e["n_paired"] and t["n"] is not None and e["n_paired"] != t["n"]:
                    paired.append(f"{TASK_LABEL[task]} {LABEL[dec]}: {int(e['n_paired'])}")
            if shown and shown["failures"]:
                notes.append(f"{TASK_LABEL[task]} {LABEL[dec]}: {shown['n_ok']}/{shown['n']} decoded")
    rows = ["".join([LABEL[dec] + (DAGGER if dec in daggered else ""), " & ", " & ".join(cells[dec]), " \\\\\n"])
            for dec in DECODERS]
    # No footnote row: Sec. 6 has no line to spare.  Driver errors and short pairings are LaTeX comments
    # here (no height), `err` and the footnote in t12_sanity_cost.tex, n_ok in the zip tables, and
    # `warnings` / `n_ok` / `n_paired` in the summary JSON.
    comments = []
    if daggered:
        comments.append("% \\dagger: off the decoder's published operating point (caption explains)\n")
    if notes:
        comments.append("% driver errors scored wrong; speed is over the problems that decoded: "
                        + "; ".join(notes) + "\n")
    if paired:
        comments.append("% difference paired on fewer than N problems: " + "; ".join(paired) + "\n")
    return "".join(head + rows + ["\\bottomrule\n\\end{tabular}\n"] + comments + [CLOSE])


def _int(x) -> str:
    return f"${int(x)}$" if _finite(x) and x else DASH


def sanity_cost_table(S: dict) -> str:
    """The sanity rows and the cost (tokens per step, forward passes, seconds per problem) per
    decoder and speed, for the slides and the code archive's README.  Problems that went wrong are
    counted apart: ``err`` driver errors (scored wrong), ``t/o`` scoring timeouts (wrong), ``uns``
    generations with no stop string in 256 tokens (not necessarily wrong); each over the three speeds."""
    rows = [HEADER.format(src="the Ada sanity run and the T12 report-half records"), OPEN,
            "\\begin{tabular}{l" + "rrr" * 4 + "}\n\\toprule\n"]
    san = S["sanity"]
    daggered = False
    rows.append("\\multicolumn{13}{l}{\\textbf{Sanity: GSM8K fit half"
                + (f", $n={san['n']}$" if san else "") + ", 5-shot, Fast-dLLM settings}} \\\\\n")
    for key, dec in SANITY_DEC.items():
        r = (san or {}).get("rows", {}).get(key)
        ours = _est(r["acc"], r["ci95"]) if r else DASH
        rows.append(f"\\multicolumn{{4}}{{l}}{{{LABEL[dec]} ({tex_cfg(key)})}} & "
                    f"\\multicolumn{{5}}{{l}}{{ours {ours}}} & "
                    f"\\multicolumn{{4}}{{l}}{{published ${PUBLISHED[key]:.3f}$}} \\\\\n")
    rows.append("\\midrule\n")
    rows.append("Decoder & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{speed {s}}}" for s in SLOTS)
                + " & \\multicolumn{3}{c}{problems} \\\\\n")
    rows.append(" & " + " & ".join("tok/st & NFE & s" for _ in SLOTS) + " & err & t/o & uns \\\\\n")
    for task in TASKS:
        t = S["tasks"][task]
        rows.append("\\midrule\n")
        rows.append(f"\\multicolumn{{13}}{{l}}{{\\textbf{{{TASK_LABEL[task]}}}"
                    + (f" ({nlabel(t)})" if t["n"] is not None else "") + "} \\\\\n")
        for dec in DECODERS:
            cells, tot = [], [0, 0, 0]
            for s in SLOTS:
                st = t["decoders"].get(dec, {}).get("speeds", {}).get(str(s))
                if st:
                    mark = DAGGER if off_op(dec, s) else ""
                    daggered = daggered or bool(mark)
                    cells += [_plain(st["tps"], 2) + mark, _plain(st["nfe"], 1), _plain(st["seconds"], 2)]
                    for i, k in enumerate(("failures", "n_timeout", "n_unstopped")):
                        tot[i] += st[k]
                else:
                    cells += [DASH] * 3
            have = t["decoders"].get(dec)
            rows.append(" & ".join([LABEL[dec]] + cells + ([f"${x}$" for x in tot] if have else [DASH] * 3))
                        + " \\\\\n")
        o = t["outside_share"]
        rows.append("\\multicolumn{13}{l}{C3a steps outside the fit range: "
                    + (num(o["share"], 3) if o else DASH) + "} \\\\\n")
    rows.append("\\midrule\n\\multicolumn{13}{l}{\\scriptsize err: driver errors (scored wrong; speed is over the "
                "problems that decoded); t/o: scoring "
                "timeouts (wrong); uns: no stop string in 256 tokens (not necessarily wrong); each summed over the "
                "three speeds."
                + (f" {DAGGER}: off the decoder's published operating point (deviation 17)." if daggered else "")
                + "} \\\\\n")
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


def _flag(b) -> str:
    return "true" if b else "false"


def _md_dagger(head: list, rows: list, flag_cols: list) -> tuple[list, list]:
    """Markdown view of rows whose last columns are the boolean ``flag_cols``: the first flag puts a
    plain dagger on the decoder, a second one (a matched setting) on its matched_slot; the flag columns
    are dropped (the CSV keeps them)."""
    n, di = len(flag_cols), head.index("decoder")
    out = []
    for r in rows:
        flags, r = r[len(r) - n:], list(r[:len(r) - n])
        if flags[0] == "true":
            r[di] = f"{r[di]} {MD_DAGGER}"
        if len(flags) > 1 and flags[1] == "true":
            ms = head.index("matched_slot")
            r[ms] = f"{r[ms]} {MD_DAGGER}"
        out.append(r)
    return head[:len(head) - n], out


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
    chead = ["task", "decoder", "speed", "config", "n", "n_ok", "failures", "unstopped", "timeouts", "failed_any", "tokens_per_step", "nfe",
             "seconds_per_problem", "outside_share", "off_operating_point"]
    crows = []
    for task in TASKS:
        t = S["tasks"][task]
        for dec in DECODERS:
            for s in SLOTS:
                st = t["decoders"].get(dec, {}).get("speeds", {}).get(str(s))
                if st:
                    o = t["outside_share"]["per_slot"].get(str(s)) if (dec == "CThreeA" and t["outside_share"]) else None
                    crows.append([task, dec, s, st["config"], st["n"], st["n_ok"], st["failures"], st["n_unstopped"],
                                  st["n_timeout"], st["n_failed_any"], _fmt(st["tps"], 3),
                                  _fmt(st["nfe"], 2), _fmt(st["seconds"], 3), _fmt(o["share"]) if o else "",
                                  _flag(off_op(dec, s))])
    _csv(d / "t12_sanity_cost.csv", chead, crows)
    mhead, mrows = _md_dagger(chead, crows, ["off_operating_point"])
    nnote = "Report problems: " + "; ".join(f"{t} {nlabel(S['tasks'][t], False) or DASH}" for t in TASKS) + ".\n"
    md = ["# T12 sanity check and cost\n", nnote,
          "Sanity: the Ada fit-half GSM8K run (5-shot, Fast-dLLM settings, fp16, 2080 Ti) against the "
          "published LLaDA-8B-Instruct numbers (0.793 top-k k=1; 0.792 threshold tau=0.9); 95% Wilson "
          "intervals.\n", _md(shead, srows),
          "\nCost, report halves (zero-shot chat, 256 tokens, blocks of 32, greedy): tokens per step = "
          "256 N_ok / sum NFE over the n_ok problems that decoded (driver errors are not imputed); NFE and seconds are per-problem means over them; failures = "
          "problems the driver could not decode (scored wrong); unstopped = no stop string within 256 tokens; "
          "timeouts = programs that timed out when scored; failed_any = problems with any of the three.\n\n" + OFF_OP_MD, _md(mhead, mrows)]
    (d / "t12_sanity_cost.md").write_text("\n".join(md), encoding="utf-8")

    hhead = ["task", "decoder", "speed", "config", "n", "n_ok", "tokens_per_step", "acc", "acc_lo", "acc_hi",
             "matched_slot", "matched_tps", "matched_diff", "diff_lo", "diff_hi", "n_paired",
             "off_operating_point", "matched_off_operating_point"]
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
                hrows.append([task, dec, s, st["config"], st["n"], st["n_ok"], _fmt(st["tps"], 3), _fmt(st.get("acc")), _fmt(lo), _fmt(hi),
                              e["slot"] if e and e["matched"] else "", _fmt(e["tps"], 3) if e and e["matched"] else "",
                              _fmt(e["diff"]) if e and e["matched"] and e["diff"] is not None else "",
                              _fmt(e["lo"]) if e and e["matched"] and e["lo"] is not None else "",
                              _fmt(e["hi"]) if e and e["matched"] and e["hi"] is not None else "",
                              e["n_paired"] if e and e["matched"] else "",
                              _flag(off_op(dec, s)), _flag(e and e["matched"] and off_op(dec, e["slot"]))])
    _csv(d / "t12_speed1_speed3.csv", hhead, hrows)
    hmhead, hmrows = _md_dagger(hhead, hrows, ["off_operating_point", "matched_off_operating_point"])
    (d / "t12_speed1_speed3.md").write_text(
        "# T12 at speeds 1 and 3\n\n" + nnote + "\nPer decoder: accuracy at its own speed-s setting (95% Wilson), and, "
        "when matched, the paired accuracy difference C3a minus the decoder at the setting nearest to "
        "C3a's slot s (within 10%; matched_slot / matched_tps), 95% percentile bootstrap CI (2,000 "
        "resamples of problems, seed 20261030). A blank is '---': no setting within 10%, or KLASS "
        "unmatched at the target.\n\n" + OFF_OP_MD + _md(hmhead, hmrows), encoding="utf-8")


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


TAG_RE = re.compile(r"[a-z0-9-]+")
SENS_NOTE = ("% SENSITIVITY OUTPUT (deviation 17, PR #74): records merged from more than one machine; "
             "NOT FOR THE PAPER.\n")


def sensitivity_dir(out: Path, tag: str) -> Path:
    return out / "t12" / "sensitivity" / tag


def planned_outputs(c3_dir: Path, t12_dir: Path, macros_name: str = "macros_c3.tex") -> list[Path]:
    """Every file ``main`` writes (the figure's two formats included) for these two directories."""
    return ([c3_dir / macros_name, t12_dir / "t12_matched_speed.tex", t12_dir / "t12_sanity_cost.tex",
             t12_dir / f"{MODEL_TAG}_t12-summary.json"]
            + [t12_dir / "figures" / f"t12_matched_speed.{e}" for e in ("pdf", "png")]
            + [t12_dir / "zip" / n for n in ("t12_sanity.csv", "t12_sanity_cost.csv", "t12_sanity_cost.md",
                                             "t12_speed1_speed3.csv", "t12_speed1_speed3.md")])


def check_sensitivity_outputs(outputs, sens: Path, normal) -> None:
    """Refuse (``SystemExit``, before anything is written) if any sensitivity output resolves outside
    ``sens`` or onto one of the paper's normal output files (``normal``)."""
    root = sens.resolve()
    bad_normal = {Path(n).resolve() for n in normal}
    for o in outputs:
        r = Path(o).resolve()
        if r in bad_normal or root not in r.parents:
            raise SystemExit(f"refusing a sensitivity run: output {o} resolves to {r}, "
                             f"{'one of the paper outputs' if r in bad_normal else 'outside ' + str(root)}; "
                             "a sensitivity run writes only under <out-dir>/t12/sensitivity/<TAG>/ "
                             "(prereg/deviations.md 17)")


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
    ap.add_argument("--manifest-dir", default="artifacts",
                    help="where split_manifest_<task>.json are: the report-half ids the records must cover")
    ap.add_argument("--allow-partial", action="store_true",
                    help="accept an incomplete run (a shard missing, a time cap): report each task on the "
                         "report-half problems that have a record for every configuration of its grid "
                         "(N = their number; the other problems' records are dropped before anything is "
                         "computed; prereg/deviations.md 17) instead of refusing the run")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--allow-mixed-machines", action="store_true",
                    help="SENSITIVITY ONLY (never for the paper): accept records of a task from more than one "
                         "run / machine / output directory (deviation 17, PR #74), recording what differed in the "
                         "summary; needs --tag; every output goes under <out-dir>/t12/sensitivity/<TAG>/ and the "
                         "macros are CThreeTTwelveSens...")
    ap.add_argument("--tag", default=None, help="name of the sensitivity run, [a-z0-9-]+ (with --allow-mixed-machines)")
    a = ap.parse_args(argv)
    sens = bool(a.allow_mixed_machines)
    if sens and not a.tag:
        ap.error("--allow-mixed-machines requires --tag TAG ([a-z0-9-]+): its outputs go under "
                 "<out-dir>/t12/sensitivity/<TAG>/")
    if a.tag is not None and not sens:
        ap.error("--tag is only for --allow-mixed-machines (a sensitivity run)")
    if sens and not TAG_RE.fullmatch(a.tag):
        ap.error(f"--tag {a.tag!r} must match [a-z0-9-]+")
    if sens:
        sdir = sensitivity_dir(Path(a.out_dir), a.tag)
        check_sensitivity_outputs(
            planned_outputs(sdir, sdir, "macros_c3_sensitivity.tex"), sdir,
            planned_outputs(Path(a.out_dir) / "c3", Path(a.out_dir) / "t12")
            + planned_outputs(ROOT / "artifacts" / "c3", ROOT / "artifacts" / "t12"))

    warnings, inputs = [], {}
    recs = load_records(a.records, warnings, inputs, a.manifest_dir, allow_mixed=sens) if a.records else {}
    grid = load_grids(a.grids, warnings, inputs)
    if a.records:
        reduce_to_complete(recs, grid, warnings)          # drops the incomplete problems' records, in place
    short = recs.get("_shortfalls") or []
    if short and not a.allow_partial:
        raise SystemExit("refusing a partial T12 run (nothing written):\n  " + "\n  ".join(short)
                         + "\npass --allow-partial to report each task on the problems for which all of its "
                           "configurations finished (prereg/deviations.md 17)")
    for w in short:
        warnings.append(f"PARTIAL (--allow-partial): {w}")
    runs = {m["run"] for m in (recs.get("_meta") or {}).values()}
    canon = {r: m["run"] for m in (recs.get("_meta") or {}).values() for r in m.get("runs", ())}
    runs |= set(canon)                                    # merged runs (sensitivity mode only)
    verdicts = load_verdicts(a.verdicts, warnings, inputs, runs, canon) if a.verdicts else {}
    scores = load_scores(a.scores, warnings, inputs, runs) if a.scores else {}
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
         "primary_speed": PRIMARY_SPEED, "allow_partial": bool(a.allow_partial), "completeness_rule": COMPLETE_RULE,
         "pairing": {"rule": RULE, "seed": SEED, "n_boot": N_BOOT, "tolerance": TOL,
                     "speed_metric": "256 N / sum(NFE) over the task's report-half problems",
                     "deviation": "prereg/deviations.md entry 17"},
         "tasks": {t: build_task(t, recs, verdicts, scores, grid, warnings) for t in TASKS},
         "off_operating_point_note": OFF_OP_NOTE,
         "calibrators": cal, "sanity": sanity,
         "inputs": inputs, "missing_inputs": missing, "warnings": warnings}
    S["provenance"] = provenance()
    if sens:
        S.update({"sensitivity": True, "sensitivity_tag": a.tag, "allow_mixed_machines": True,
                  "mixed_machines": (recs.get("_mixed") or {}),
                  "sensitivity_note": "NOT FOR THE PAPER: records merged from more than one machine "
                                      "(prereg/deviations.md 17, PR #74)"})

    out = Path(a.out_dir)
    if sens:
        t12 = c3 = sensitivity_dir(out, a.tag)            # every output under one directory
        macros_path = c3 / "macros_c3_sensitivity.tex"
        macros = build_macros(S, "CThreeTTwelveSens", "CThreeTElevenSens")
        note = SENS_NOTE
    else:
        t12, c3 = out / "t12", out / "c3"
        macros_path = c3 / "macros_c3.tex"
        macros = build_macros(S)
        note = ""
    t12.mkdir(parents=True, exist_ok=True)
    write_macros(macros, macros_path)
    (t12 / "t12_matched_speed.tex").write_text(note + matched_table(S), encoding="utf-8")
    (t12 / "t12_sanity_cost.tex").write_text(note + sanity_cost_table(S), encoding="utf-8")
    write_zip_tables(S, t12 / "zip")
    figs = [] if a.no_figures else make_figure(S, t12 / "figures")
    S["outputs"] = sorted([str(macros_path), str(t12 / "t12_matched_speed.tex"),
                           str(t12 / "t12_sanity_cost.tex")] + [str(p) for p in figs])
    S["macro_names"] = sorted(macros.names)
    (t12 / f"{MODEL_TAG}_t12-summary.json").write_text(json.dumps(jsonable(S), indent=2, default=str),
                                                         encoding="utf-8")
    n = {t: S["tasks"][t]["n"] for t in TASKS}
    print(f"[c3-paper] {len(macros.names)} macros, {len(figs)} figure files -> {t12 if sens else out} "
          f"(report problems: {n['gsm8k'] or DASH} GSM8K, {n['mbpp'] or DASH} MBPP)")
    if missing:
        print(f"[c3-paper] no input for: {', '.join(missing)} (their cells are {DASH}, their macros undefined)",
              file=sys.stderr)
    for w in warnings:
        print(f"[c3-paper] warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
