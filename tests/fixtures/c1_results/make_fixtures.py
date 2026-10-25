#!/usr/bin/env python3
"""Rebuild tests/fixtures/c1_results/ -- SYNTHETIC, a schema fixture, not a result.

The files are what scripts/analyze_traces.py, analyze_e3.py, run_controls.py and
summarize_c1.py write, from a toy model on 16 documents (``ToyDiffusionLM``) and the
simulated designs of ``c3.sim.trajectories``, with the numbers of each "arm" scaled and the
bootstrap p-values set by hand so that the tables show a mix of decisions.  Nothing here
has seen a real trace.  Only needed to regenerate (the output is committed):

    # 1. the three base files (about 2 min)
    python tests/fixtures/c1_results/make_fixtures.py --gen-base /tmp/c1base
    # 2. the fixture tree from them
    python tests/fixtures/c1_results/make_fixtures.py --base /tmp/c1base
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

NOTE = "SYNTHETIC test fixture (toy model, scaled by make_fixtures.py): not a result"
# label: (scale of the effects, bootstrap p of the E1 contrast, of the E2 contrast)
ARMS = {
    "LladaBase": (1.0, 1e-4, 1e-4),
    "DreamBase": (0.8, 1e-4, 0.20),
    "LladaInstructGsm": (0.5, 0.01, 0.30),
    "LladaInstructMbpp": (0.4, 0.04, 0.60),
    "DreamInstructGsm": (0.6, 0.30, 0.70),
    "DreamInstructMbpp": (0.3, 0.50, 0.90),
    "Mdlm": (1.2, 1e-4, 1e-4),
    "LladaBaseTimeSplit": (0.9, 0.02, 0.02),
}
FLOOR = 0.01


def gen_base(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], check=True, cwd=ROOT)  # noqa: E731
    import numpy as np
    from c3.traces.collect import CollectConfig, Document, collect_shard
    from c3.traces.models import ToyDiffusionLM
    from c3.traces.shards import merge
    rng = np.random.default_rng(0)
    docs = [Document(i, rng.integers(0, 60, 100)) for i in range(16)]
    cfg = CollectConfig(topk=5, mask_rates=(0.5, 0.8), masks_per_rate=2, decoder_k=(4, 8))
    collect_shard(ToyDiffusionLM(), docs, cfg, out / "toy" / "run", 0, 1, part_docs=4,
                  progress_every=0)
    man = merge(out / "toy" / "run", out / "toy" / "run.dataset.json")
    (out / "split.json").write_text(json.dumps({str(i): i % 2 for i in range(16)}))
    run("scripts/analyze_traces.py", "--traces", man, "--split", "half", "--manifest",
        out / "split.json", "--n-boot", 40, "--eq-boot", 20, "--knots", 5, "--floor", FLOOR,
        "--no-figures", "--out", out / "c1")
    run("scripts/analyze_e3.py", "--traces", "sim-design:decoder_shift,n_docs=30,shift=0.5,ks=2+4+8",
        "--split", "half", "--half", "report", "--n-boot", 40, "--knots", 5, "--n-ell-bins", 20,
        "--reference", "both", "--no-figures", "--out", out / "e3")
    run("scripts/run_controls.py", "--traces", "sim-design:position,n_docs=40,ks=4+8", "--split",
        "half", "--half", "report", "--n-boot", 40, "--knots", 5, "--temperature", "logodds",
        "--out", out / "t8")


def scale_pair(d: dict, keys, s: float) -> None:
    for k in keys:
        if k in d and d[k] is not None:
            d[k] = [x * s for x in d[k]] if isinstance(d[k], list) else d[k] * s


def make_c1(base: dict, label: str, s: float, p1: float, p2: float) -> dict:
    res = copy.deepcopy(base)
    res["label"] = f"{NOTE}; arm {label}"
    res["input"] = {"source": f"synthetic:{label}"}
    res["provenance"]["note"] = NOTE
    for kind, r in res["by_mask_kind"].items():
        r["calibrators"] = {"trimmed": "calibrator tables dropped from the fixture"}
        r.pop("native", None)
        for row in r["delta"]:
            scale_pair(row, ("delta", "ci_low", "ci_high"), s)
        for row in r["decomposition"]:
            for n in ("delta", "delta_own", "field", "residual", "delta_ctx"):
                scale_pair(row, (n, f"{n}_ci"), s)
        for key, p in (("contrast", p1), ("field_contrast", p2)):
            c = r[key]
            scale_pair(c, ("value", "ci"), s)
            c["p_boot"] = p
            lo, hi = c["ci"]
            c["declared"] = bool((lo > 0 or hi < 0) and abs(c["value"]) > 2 * FLOOR)
            if "slope_log2m" in c:
                scale_pair(c, ("slope_log2m", "slope_ci"), s)
    return res


def variant(res: dict, tag: str) -> dict:
    out = copy.deepcopy(res)
    out["label"] += f" [{tag}]"
    for r in out["by_mask_kind"].values():
        scale_pair(r["field_contrast"], ("value", "ci"), 1.1 if tag != "clean" else 0.95)
    return out


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-base", default=None)
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    if a.gen_base:
        gen_base(Path(a.gen_base))
        return
    base = Path(a.base)
    import analyze_e3
    import analyze_traces
    import run_controls
    c1 = json.loads((base / "c1" / "results.json").read_text())
    e3 = json.loads((base / "e3" / "results.json").read_text())
    t8 = json.loads((base / "t8" / "ablation.json").read_text())
    for old in ("c1", "e3", "t8"):
        for p in sorted((HERE / old).rglob("*"), reverse=True) if (HERE / old).exists() else []:
            p.unlink() if p.is_file() else p.rmdir()

    for label, (s, p1, p2) in ARMS.items():
        res = make_c1(c1, label, s, p1, p2)
        dump(HERE / "c1" / label / "results.json", res)
        analyze_traces.write_macros(HERE / "c1" / label / "macros.tex", f"COne{label}", res)
    for label, tag, word in (("LladaBase", "knots5", "KnotsFive"), ("LladaBase", "knots13", "KnotsThirteen"),
                             ("LladaBase", "clean", "Clean"), ("DreamBase", "knots5", "KnotsFive")):
        res = variant(make_c1(c1, label, *ARMS[label]), tag)
        d = HERE / "c1" / f"{label}_{tag}"
        dump(d / "results.json", res)
        analyze_traces.write_macros(d / "macros.tex", f"COne{label}{word}", res)

    # the real summarize_c1.py, over the seven main arms
    specs = [f"{l}={HERE / 'c1' / l}" for l in list(ARMS)[:7]]
    subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_c1.py"), "--results", *specs,
                    "--primary-arms", "uniform", "decoder:topk_conf:k4", "--out",
                    str(HERE / "c1" / "summary")], check=True, cwd=ROOT)
    summ = json.loads((HERE / "c1" / "summary" / "c1_summary.json").read_text())
    summ["provenance"]["note"] = NOTE
    dump(HERE / "c1" / "summary" / "c1_summary.json", summ)

    for label, s in (("LladaBase", 1.0), ("DreamBase", 0.7), ("Mdlm", 1.3)):
        res = copy.deepcopy(e3)
        res["input"] = {"source": f"synthetic:{label}", "note": NOTE}
        res["provenance"]["note"] = NOTE
        for c in res["comparisons"].values():
            scale_pair(c["type_effect"], ("value", "ci"), s)
            c["nuisance"] = {"trimmed": "dropped from the fixture"}
        dump(HERE / "e3" / label / "results.json", res)
        analyze_e3.write_macros(HERE / "e3" / label / "macros.tex", f"EThree{label}", res)
    for label, s in (("LladaBase", 1.0), ("DreamBase", 0.7), ("Mdlm", 1.3)):
        res = copy.deepcopy(t8)
        res["input"] = {"source": f"synthetic:{label}", "note": NOTE}
        res["provenance"]["note"] = NOTE
        res["config"] = {}
        for r in res["rows"]:
            for part in ("without", "with", "difference"):
                scale_pair(r[part], ("value", "ci"), s)
        dump(HERE / "t8" / label / "ablation.json", res)
        run_controls.write_macros(HERE / "t8" / label / "macros.tex", f"TEight{label}", res["rows"],
                                  NOTE)
    # precision: fp16 vs 4-bit rows, relabelled from the position rows
    prec = copy.deepcopy(t8)
    prec["rows"] = [dict(r, control="precision", note="without = fp16, with = 4-bit, same documents")
                    for r in prec["rows"] if r["control"] == "position"
                    and r["arm"] in ("uniform", "decoder:topk_conf:k4")]
    prec["input"] = {"source": "synthetic:LladaBasePrecision", "note": NOTE}
    prec["config"] = {}
    dump(HERE / "t8" / "LladaBasePrecision" / "ablation.json", prec)
    run_controls.write_macros(HERE / "t8" / "LladaBasePrecision" / "macros.tex",
                              "TEightLladaBasePrecision", prec["rows"], NOTE)
    print("fixtures written under", HERE)


if __name__ == "__main__":
    main()
