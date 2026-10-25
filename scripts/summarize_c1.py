#!/usr/bin/env python3
"""Collect every C1 arm into one table, with Holm across the pre-registered families.

    python scripts/summarize_c1.py --results Small=artifacts/c1_Small \\
        LladaBase=artifacts/c1_LladaBase DreamBase=artifacts/c1_DreamBase \\
        --out artifacts/c1_summary

Rows are (model, mask arm) from each ``analyze_traces.py`` output.  Columns:
Delta(1), Delta(32), the E1 contrast and whether it is declared, the E2 field
contrast, beta_z with and
without state FE, gamma_2..gamma_4 (Eq. 3, with FE), ECE and the number of
documents.  Holm (``c3.stats.holm``) is applied to three families, from the
document-bootstrap p-values: the E1 contrast across rows, the E2 field contrast
across rows (plan v2 Sec. 5: models x datasets for pre-registered endpoints),
and gamma_2..gamma_4 within each row.  ``--primary-arms`` restricts the two
across-row families to the pre-registered arms; everything else is labelled
exploratory.  Each family is fixed in advance as every label x primary arm: a
member that is missing or has no finite p-value counts as p = 1, so a lost arm
cannot shrink the family.  A family member is ``claimed`` only if it is declared
(``c3.stats.declare``) AND rejected by Holm; the table stars claimed effects.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c3.provenance import collect as provenance
from c3.stats import holm


_ARM_WORDS = {"uniform": "Uni", "decoder": "Dec", "uniform_matched": "UniMatched"}
_DIGITS = dict(zip("0123456789", ("Zero", "One", "Two", "Three", "Four", "Five", "Six",
                                  "Seven", "Eight", "Nine")))


def arm_tag(arm: str) -> str:
    """Letters-only macro name for a mask arm: decoder:topk_conf:k4 -> DecTopkConfKFour."""
    head, *rest = arm.split(":")
    tag = _ARM_WORDS.get(head, "Arm")
    for w in (w for part in rest for w in part.split("_")):
        w = "".join(_DIGITS.get(ch, ch) for ch in w)
        tag += w[:1].upper() + w[1:]
    return tag


def _delta(r, m):
    row = next((x for x in r["delta"] if int(x["level"]) == m), None)
    return None if row is None else {"value": row["delta"], "ci": [row["ci_low"], row["ci_high"]]}


def _coef(r, key, name):
    c = r.get("regression", {}).get(key, {}).get(name)
    return None if not c or "value" not in c else {"value": c["value"], "ci": c["ci"],
                                                   "p_boot": c.get("p_boot")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True, help="LABEL=DIR (analyze_traces --out)")
    ap.add_argument("--primary-arms", nargs="*", default=None,
                    help="arms in the E1/E2 Holm families (default: every arm)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--macro-prefix", default="COneSum")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for spec in a.results:
        label, _, d = spec.partition("=")
        res = json.loads((Path(d) / "results.json").read_text())
        for arm, r in res["by_mask_kind"].items():
            rows.append({
                "model": label, "arm": arm, "n_docs": r["n_docs"],
                "delta_1": _delta(r, 1), "delta_32": _delta(r, 32),
                "contrast": r["contrast"], "field_contrast": r["field_contrast"],
                "beta_z_nofe": _coef(r, "eq2_nofe", "beta_z"),
                "beta_z_fe": _coef(r, "eq2_fe", "beta_z"),
                "gamma": {g: _coef(r, "eq3_fe", g) for g in ("gamma_2", "gamma_3", "gamma_4")},
                "ece": r["ece"]["ece"],
                "source": d})
    primary = [i for i, r in enumerate(rows)
               if a.primary_arms is None or r["arm"] in a.primary_arms]
    labels = [spec.partition("=")[0] for spec in a.results]
    arms = a.primary_arms if a.primary_arms is not None else sorted({r["arm"] for r in rows})
    members = [(lab, arm) for lab in labels for arm in arms]
    where = {(rows[i]["model"], rows[i]["arm"]): i for i in primary}
    missing = [f"{lab}:{arm}" for lab, arm in members if (lab, arm) not in where]
    if missing:
        print(f"[summarize] WARNING: {len(missing)} Holm-family member(s) missing, counted as "
              f"p = 1: {', '.join(missing)}", file=sys.stderr)
    for fam, key in (("E1", "contrast"), ("E2", "field_contrast")):
        p = np.array([rows[where[m]][key].get("p_boot", np.nan) if m in where else np.nan
                      for m in members], dtype=float)
        adj, rej = holm(np.where(np.isfinite(p), p, 1.0), a.alpha)
        for j, m in enumerate(members):
            if m not in where:
                continue
            c = rows[where[m]][key]
            c["holm_family"], c["holm_m"] = fam, len(members)
            c["p_holm"], c["holm_reject"] = float(adj[j]), bool(rej[j])
            c["claimed"] = bool(rej[j]) and bool(c.get("declared"))
    for i, r in enumerate(rows):
        if i not in primary:
            for key in ("contrast", "field_contrast"):
                r[key]["holm_family"] = "exploratory"
        gs = [g for g in ("gamma_2", "gamma_3", "gamma_4") if r["gamma"][g]
              and r["gamma"][g].get("p_boot") is not None]
        if gs:
            adj, rej = holm([r["gamma"][g]["p_boot"] for g in gs], a.alpha)
            for g, pa, rj in zip(gs, adj, rej):
                r["gamma"][g]["p_holm"], r["gamma"][g]["holm_reject"] = float(pa), bool(rj)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "c1_summary.json").write_text(json.dumps(
        {"rows": rows, "alpha": a.alpha, "primary_arms": a.primary_arms,
         "holm_members": [f"{lab}:{arm}" for lab, arm in members], "holm_missing": missing,
         "provenance": provenance(),
         "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")},
        indent=2, default=float))

    def f(c):
        return "--" if not c else f"${c['value']:+.3f}$"

    tex = ["% Generated by scripts/summarize_c1.py -- do not edit by hand.",
           "\\begin{tabular}{llrcccccc}", "\\toprule",
           "Model & arm & docs & $\\Delta(1)$ & $\\Delta(32)$ & E1 contrast & E2 field & "
           "$\\beta_z$ no FE & $\\beta_z$ FE \\\\", "\\midrule"]
    for r in rows:
        star = lambda c: "$^{*}$" if c.get("claimed") else ""  # noqa: E731
        tex.append(f"{r['model']} & {r['arm'].replace('_', '-')} & {r['n_docs']} & "
                   f"{f(r['delta_1'])} & {f(r['delta_32'])} & "
                   f"{f(r['contrast'])}{star(r['contrast'])} & "
                   f"{f(r['field_contrast'])}{star(r['field_contrast'])} & "
                   f"{f(r['beta_z_nofe'])} & {f(r['beta_z_fe'])} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "% *: claimed: declared (c3.stats.declare) and rejected after Holm within its family."]
    (out / "table.tex").write_text("\n".join(tex) + "\n")
    mac = ["% Generated by scripts/summarize_c1.py -- do not edit by hand."]
    for r in rows:
        tag = r["model"] + arm_tag(r["arm"])
        if not tag.isalpha():
            continue        # a model label LaTeX cannot name; its numbers are in summary.json
        for key, nm in (("contrast", "Contrast"), ("field_contrast", "Field")):
            mac.append(f"\\newcommand{{\\{a.macro_prefix}{nm}Holm{tag}}}"
                       f"{{{r[key].get('p_holm', float('nan')):.4f}}}")
    (out / "macros.tex").write_text("\n".join(mac) + "\n")
    print(f"[summary] {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
