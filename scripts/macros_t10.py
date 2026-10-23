#!/usr/bin/env python3
"""T10 for the paper: one LaTeX macro per C2 real-text and probe-calibration number.

    python scripts/macros_t10.py                  # writes artifacts/c2/macros_t10.tex

Reads committed artifacts only (no model, no GPU), from ``--in-dir``:

  t10_local-110m_fp32_train110m-v3.json   the 110M (train110m-v3, fp32): real text, and the
                                          disagreement proxy on the T9 probe for both fine-tunes
  t10_llada-8b-base_fp16_main.json        LLaDA-8B-Base (fp16): real text only

and writes ``--out``: one \\newcommand per number Sec. 5 (C2) cites, prefix CTwoTTen, letters
only, the naming scheme in the file's header (the style of ``scripts/figures_t9.py``).

The paper reads real text over mask rates 0.1-0.9 (``per_k_rates_below_1``).  At rate 1.0
every document's input is the same all-[MASK] sequence, so that one state gives every
document the same disagreement; the all-rates numbers (``per_k``, and ``contrasts``, which
are computed over all ten rates) are written as well, under names that say AllRates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c3.provenance import sha256_file  # noqa: E402

PREFIX = "CTwoTTen"
RUNS = {"": "t10_local-110m_fp32_train110m-v3.json",       # 110M: real text + probe
        "Llada": "t10_llada-8b-base_fp16_main.json"}        # LLaDA-8B-Base: real text only
PROBE_TAGS = {"ft110m-v3-visible": "", "ft110m-v3": "Alt"}
"""The main variant first: T9 chose ft110m-v3-visible (artifacts/probe/main_variant.json)."""
LOW_RATES = 0.7
"""DisMaxUpToRatePtSeven: the largest disagreement over k and the rates up to this one."""

K_WORD = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
RHO_WORD = {0.0: "Zero", 0.25: "Quarter", 0.5: "Half", 0.75: "ThreeQuarters", 1.0: "One"}
RATE_WORD = {0.1: "PtOne", 0.2: "PtTwo", 0.3: "PtThree", 0.4: "PtFour", 0.5: "PtFive",
             0.6: "PtSix", 0.7: "PtSeven", 0.8: "PtEight", 0.9: "PtNine", 1.0: "One"}
CONTRAST_WORD = {"disagreement_k4_minus_k2": "DisKFourMinusKTwo",
                 "disagreement_k8_minus_k4": "DisKEightMinusKFour",
                 "disagreement_k8_minus_k2": "DisKEightMinusKTwo",
                 "delta_k8_minus_k1": "DeltaKEightMinusKOne",
                 "ce_k8_minus_k1": "CEKEightMinusKOne"}


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def rel(path: Path) -> str:
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


# --- number formats -------------------------------------------------------------------------
# A mean that can take either sign is written with its sign (+0.007); interval bounds carry a
# minus only (as in the mid report: +0.005 [-0.009, 0.019]).  A value that rounds to zero is
# written unsigned.  Minus signs are set in math mode, so they print as minus, not hyphen.
def fixed(digits: int, sign: bool = False):
    def fmt(v: float) -> str:
        body = f"{abs(v):.{digits}f}"
        if float(body) == 0:
            return body
        if v < 0:
            return f"\\ensuremath{{-{body}}}"
        return f"\\ensuremath{{+{body}}}" if sign else body
    return fmt


def rate(v: float) -> str:
    """Disagreement on real text: a rate, mostly far below 1%, so four decimals (0 if none)."""
    return "0" if v == 0 else fixed(4)(v)


def count(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


QTY = {"disagreement": ("Dis", rate, rate),
       "delta": ("Delta", fixed(3, sign=True), fixed(3)),
       "ce": ("CE", fixed(2), fixed(2))}
"""JSON quantity -> (macro word, format of the mean, format of the interval bounds).
Delta = conf - agree (positive = overconfident); CE in nats."""
CONTRAST_FMT = {"disagreement": (fixed(4, sign=True), fixed(4)),
                "delta": (fixed(3, sign=True), fixed(3)),
                "ce": (fixed(2, sign=True), fixed(2))}
PROBE_DIS = fixed(3)
"""Disagreement on the probe: per field, three decimals."""
CORR = fixed(2)


class Macros:
    def __init__(self):
        self.lines: list[str] = []
        self.names: set[str] = set()

    def comment(self, text: str = "") -> None:
        self.lines.append(f"% {text}".rstrip())

    def add(self, name: str, body: str) -> None:
        name = PREFIX + name
        if not (name.isascii() and name.isalpha()):
            raise ValueError(f"macro name {name!r} is not letters only")
        if name in self.names:
            raise ValueError(f"macro {name!r} defined twice")
        self.names.add(name)
        self.lines.append(f"\\newcommand{{\\{name}}}{{{body}}}")

    def est(self, name: str, e: dict, fmt, bound=None, interval: bool = True) -> None:
        """The mean as ``name`` and, unless ``interval`` is False, its 95% interval as
        ``name``Lo / ``name``Hi (formatted by ``bound``, default ``fmt``)."""
        bound = bound or fmt
        self.add(name, fmt(e["mean"]))
        if interval:
            self.add(name + "Lo", bound(e["ci95"][0]))
            self.add(name + "Hi", bound(e["ci95"][1]))


# --- checks ---------------------------------------------------------------------------------
def check(runs: dict) -> None:
    for word, d in runs.items():
        name = RUNS[word]
        if d.get("task") != "T10":
            raise SystemExit(f"{name}: not a T10 output")
        rt = d["real_text"]
        ks = sorted(e["k"] for e in rt["per_k"])
        if ks != sorted(e["k"] for e in rt["per_k_rates_below_1"]) or ks != sorted(d["config"]["b_ks"]):
            raise SystemExit(f"{name}: per_k and per_k_rates_below_1 cover different k")
        if any(e["k"] == 1 and e["disagreement"]["mean"] != 0 for e in rt["per_k"]):
            raise SystemExit(f"{name}: disagreement at k = 1 is not 0")
        # MinMasked is read from the assumptions as written, not re-derived.
        floor = max(d["config"]["ks"])
        if not any(f"min_masked = max(k) = {floor}" in a for a in d["assumptions"]):
            raise SystemExit(f"{name}: the min_masked assumption is not max(k) = {floor}")
        if d["provenance"].get("git_dirty"):
            raise SystemExit(f"{name}: produced from a dirty tree")
    a, b = runs[""], runs["Llada"]
    for key in ("ks", "b_ks", "rates", "n_boot", "n_docs", "seed"):
        if a["config"].get(key) != b["config"].get(key):
            raise SystemExit(f"the two runs differ in config.{key}")
    probe = a.get("probe_calibration", {})
    if set(probe) != set(PROBE_TAGS):
        raise SystemExit(f"probe_calibration holds {sorted(probe)}, expected {sorted(PROBE_TAGS)}")
    if len({probe[t]["spearman"]["n_cells"] for t in probe}) != 1:
        raise SystemExit("the two probe correlations use different numbers of cells")
    chosen = ROOT / "artifacts" / "probe" / "main_variant.json"      # T9 (PR #12), if present
    if chosen.is_file() and load(chosen)["choice"] != next(iter(PROBE_TAGS)):
        raise SystemExit(f"{rel(chosen)} chose {load(chosen)['choice']}, not "
                         f"{next(iter(PROBE_TAGS))}")


# --- the macros -----------------------------------------------------------------------------
def macros(runs: dict, paths: dict) -> str:
    m = Macros()
    m.comment("Generated by scripts/macros_t10.py -- do not edit by hand.")
    m.comment("T10, C2 on real text: parallel-sequential disagreement (proxy for (A)), Delta and CE")
    m.comment("at the committed positions (proxies for (B)), and the disagreement proxy on the T9 probe.")
    m.comment("Sources (sha256; the run's commit and Slurm job):")
    for word, d in runs.items():
        prov = d["provenance"]
        m.comment(f"  {rel(paths[word])}  {sha256_file(paths[word])}  "
                  f"(commit {prov['git_commit'][:7]}, job {prov.get('slurm_job_id')})")
    m.comment()
    m.comment("Model word: none = the 110M (train110m-v3, fp32); Llada = LLaDA-8B-Base (fp16).")
    m.comment("Real text, mask rates 0.1-0.9 (per_k_rates_below_1; what the paper reads):")
    m.comment("  \\CTwoTTen[Llada]<Q>K<K>[Lo|Hi]   Q: Dis = disagreement (fraction of the k committed")
    m.comment("  positions where parallel and sequential tokens differ; 0 at k = 1 by construction),")
    m.comment("  Delta = conf - agree (positive = overconfident), CE = -log p(true token) in nats;")
    m.comment("  K: KOne, KTwo, KFour, KEight; Lo/Hi: 95% document-bootstrap interval")
    m.comment("All ten rates, including the shared all-[MASK] state at rate 1.0 (per_k):")
    m.comment("  \\CTwoTTen[Llada]<Q>AllRatesK<K>[Lo|Hi]")
    m.comment("Disagreement per k and mask rate (per_k_rate): \\CTwoTTen[Llada]DisK<K>Rate<R>[Lo|Hi],")
    m.comment("  R: PtOne ... PtNine = 0.1 ... 0.9, One = 1.0;  ...DisMaxUpToRatePtSeven = the largest")
    m.comment(f"  disagreement over k and the rates up to {LOW_RATES}")
    m.comment("Contrasts between k, over all ten rates (contrasts): \\CTwoTTen[Llada]Contrast<Name>[Lo|Hi]")
    m.comment("Probe (110M only): no variant word = ft110m-v3-visible (T9's main variant), Alt = ft110m-v3;")
    m.comment("  ...ProbeSpearman[Lo|Hi] = Spearman of disagreement with the exact (A) per field over the")
    m.comment("  k >= 2 confidence-rule cells; ...ProbeDisatRho<Rho>K<K> = disagreement per field in a cell")
    m.comment("  (Lo/Hi for the main variant only)")

    a = runs[""]
    m.comment()
    m.comment("Design")
    m.add("NStatesPerDoc", count(len(a["config"]["rates"])))
    m.add("NBoot", count(a["config"]["n_boot"]))
    m.add("MinMasked", count(max(a["config"]["ks"])))
    for word, d in runs.items():
        rt = d["real_text"]
        m.add(f"{word}NDocs", count(rt["n_docs"]))
        m.add(f"{word}NStates", count(rt["n_states"]))
        m.add(f"{word}LastDocId", count(rt["documents"]["last_doc_id"]))
        m.add(f"{word}NSkippedShort", count(len(rt["documents"]["skipped_short"])))

    for word, d in runs.items():
        rt = d["real_text"]
        m.comment()
        m.comment(f"{RUNS[word]}: real text")
        for e in sorted(rt["per_k_rates_below_1"], key=lambda e: e["k"]):
            for q, (short, fmt, bound) in QTY.items():
                m.est(f"{word}{short}K{K_WORD[e['k']]}", e[q], fmt, bound)
        for e in sorted(rt["per_k"], key=lambda e: e["k"]):
            for q, (short, fmt, bound) in QTY.items():
                m.est(f"{word}{short}AllRatesK{K_WORD[e['k']]}", e[q], fmt, bound)
        low = []
        for e in sorted(rt["per_k_rate"], key=lambda e: (e["k"], e["rate"])):
            if e["k"] == 1:                          # 0 by construction (checked above)
                continue
            m.est(f"{word}DisK{K_WORD[e['k']]}Rate{RATE_WORD[e['rate']]}", e["disagreement"], rate)
            if e["rate"] <= LOW_RATES + 1e-9:
                low.append(e["disagreement"]["mean"])
        m.add(f"{word}DisMaxUpToRatePtSeven", rate(max(low)))
        for key, x in rt["contrasts"].items():
            fmt, bound = CONTRAST_FMT[key.split("_k")[0]]
            m.est(f"{word}Contrast{CONTRAST_WORD[key]}", x, fmt, bound)

    probe = a["probe_calibration"]
    m.comment()
    m.comment(f"{RUNS['']}: the disagreement proxy on the T9 probe")
    first = next(iter(PROBE_TAGS))
    m.add("ProbeNCells", count(probe[first]["spearman"]["n_cells"]))
    m.add("ProbeNInstances", count(probe[first]["n_instances"]))
    for tag, word in PROBE_TAGS.items():
        p = probe[tag]
        sp = p["spearman"]
        m.est(f"{word}ProbeSpearman", {"mean": sp["rho_s"], "ci95": sp["ci95"]}, CORR)
        m.add(f"{word}ProbeStepsBitwise", count(p["alignment"]["steps_bitwise"]))
        m.add(f"{word}ProbeStepsGrid", count(p["alignment"]["steps_grid"]))
        for c in sorted(p["cells"], key=lambda c: (c["rho"], c["k"])):
            m.est(f"{word}ProbeDisatRho{RHO_WORD[c['rho']]}K{K_WORD[c['k']]}", c["disagreement"],
                  PROBE_DIS, interval=not word)
    return "\n".join(m.lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=str(ROOT / "artifacts" / "c2"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "c2" / "macros_t10.tex"))
    a = ap.parse_args()
    paths = {word: Path(a.in_dir) / name for word, name in RUNS.items()}
    runs = {word: load(p) for word, p in paths.items()}
    check(runs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(macros(runs, paths))
    print(f"[t10] macros -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
