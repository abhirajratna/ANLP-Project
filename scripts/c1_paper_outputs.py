#!/usr/bin/env python3
"""C1 for the paper: tables, figures and macros from the analysis' result files alone.

    python scripts/c1_paper_outputs.py                       # results/  ->  paper/{tables,figures}/c1/
    python scripts/c1_paper_outputs.py --results /tmp/results --main-arm DreamBase

Reads what ``prereg/analysis_commands.sh`` writes under ``--results`` (no traces, no GPU,
no model; the cluster has no matplotlib, the laptop does):

  c1/<Label>/results.json (+ macros.tex)       E1, E2, ECE, Eq. 2/3, per mask arm
  c1/<Label>_knots5|_knots13|_clean|_additive/ sensitivity variants of the same (additive: the
                                               additive context calibrator, E1/E2 only)
  c1/summary/c1_summary.json                   Holm across models x datasets, gamma_2..4 within
                                               arm, ``claimed``, ``holm_m``, ``holm_missing``
                                               (summarize_c1.py)
  e3/<Label>/results.json (+ macros.tex)       decoder vs matched-uniform / grid at equal confidence
  t8/<Label>/ablation.json (+ macros.tex)      position, token type, temperature (and precision)

and writes, under ``--out-tables`` (default paper/tables/c1) and ``--out-figures``
(default paper/figures/c1):

  c1_main.tex          main table, page width (\\input in a table*): per arm and mask arm, E1 and
                       E2 with 95% CI and the Holm decision, beta_z of Eq. 2 without (beta_z) and
                       with state FE (beta_z^FE), values only
  c1_main_col.tex      one column: E1 and E2 point estimates with the Holm / declared markers
  c1_beta.tex          beta_z with 95% CIs, no FE and FE
  c1_gamma.tex         gamma_2..gamma_4 of Eq. 3 (with state FE) and the Holm p within the arm
  c1_shared.tex        the shared-error term E_m[pi(l,s) - a] at m = 1 and 32, and the within-state
                       ICC of the residuals a - pi(l,s)
  c1_sensitivity.tex   E1 / E2 under 5, 9, 13 knots, without contaminated documents, and E2 with
                       the additive context calibrator
  c1_e3.tex            E3, type effect at equal confidence, matched-uniform and grid references
  c1_t8.tex            T8, E1 and E2 without / with position, token type, temperature controls
  c1_t8_precision.tex  T8 precision (fp16 vs 4-bit), when the run exists
  macros_c1.tex        ``\\input`` of every per-arm macros.tex (copied to macros/) plus the macros
                       the analysis does not write: Holm decisions, ``claimed`` flags and counts,
                       the family size and missing members (``\\COneHolmFootnote``), the E2
                       contrast (also under each sensitivity variant), declared flags, the
                       shared-error term and ICC, gamma_2..4 with their Holm p
  c1_sources.json      every input with its sha256, and a c3/provenance.py stamp
  c1_delta_m_all, c1_field_m_all           Delta(m), field(m) vs m, one panel per arm
  c1_delta_m_<Label>, c1_reliability_<Label>, c1_e3_<Label>   for the main arm
                       (``.pdf`` and ``.png``, ACL widths)

Missing arms, mask arms, variants or summaries are shown as ``---`` (an em dash in LaTeX), never
a crash: results land arm by arm, and the 110M and 8B arms live under two Ada accounts.

Table markers (one superscript): ``*`` claimed: declared AND Holm-rejected in its family (the
``claimed`` field of the summary; in c1_gamma.tex: Holm-rejected within the arm); ``\\dagger``
declared (CI excludes 0 and the effect exceeds 2 x floor); ``\\ddagger`` the 95% CI of the number shown (beta_z,
or in T8 the paired with-minus-without difference) excludes 0; ``\\S`` the Eq. 2 fit did not
converge.  ``TS`` in an arm name is the time-split set.  The preamble needs ``booktabs``.  Every
table is a bare ``tabular`` inside a ``\\footnotesize`` group; the caption is the paper's.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from c3.provenance import collect as provenance, sha256_file  # noqa: E402

# --------------------------------------------------------------------------- arms
# Label order of prereg/analysis_commands.sh, with the name the paper prints.
MAIN_ARMS = {
    "LladaBase": "LLaDA Base",
    "DreamBase": "Dream Base",
    "LladaInstructGsm": "LLaDA-I GSM8K",
    "LladaInstructMbpp": "LLaDA-I MBPP",
    "DreamInstructGsm": "Dream-I GSM8K",
    "DreamInstructMbpp": "Dream-I MBPP",
    "Mdlm": "MDLM-110M",
}
TIMESPLIT_ARMS = {                      # TS = the time-split set (T4), guaranteed unseen
    "LladaBaseTimeSplit": "LLaDA Base TS",
    "DreamBaseTimeSplit": "Dream Base TS",
    "MdlmTimeSplit": "MDLM-110M TS",
}
KNOWN = {**MAIN_ARMS, **TIMESPLIT_ARMS}
# The two arms of the E1 / E2 Holm families (PRIMARY_ARMS of analysis_commands.sh).
PRIMARY_MASKS = ("uniform", "decoder:topk_conf:k4")
MASK_NAME = {"uniform": "uniform", "decoder:topk_conf:k4": "dec. $k{=}4$"}
E3_KS = (2, 4, 8)
VARIANTS = {"knots5": "KnotsFive", "knots13": "KnotsThirteen", "clean": "Clean",
            "additive": "Additive"}
GAMMAS = {"gamma_2": "Two", "gamma_3": "Three", "gamma_4": "Four"}
SHARED_MS = (1, 32)                     # field sizes of the shared-error table
DASH = "---"
FIG_ARMS_PER_ROW = 4

_WORDS = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen", 32: "ThirtyTwo",
          64: "SixtyFour", 128: "OneTwentyEight", 256: "TwoFiftySix"}
_KIND_TAGS = {"uniform": "Uni", "decoder": "Dec", "uniform_matched": "UniMatched"}


def arm_tag(arm: str) -> str:
    """Digit-free macro tag of a mask arm; the same rule as scripts/analyze_traces.py."""
    kind, *rest = arm.split(":")
    tag = _KIND_TAGS.get(kind, kind.title().replace("_", ""))
    for part in rest:
        if part[:1] == "k" and part[1:].isdigit():
            tag += "K" + _WORDS[int(part[1:])]
        else:
            tag += "".join(w.title() for w in part.split("_"))
    return tag


def display(label: str) -> str:
    return KNOWN.get(label) or tex_escape(label)


def tex_escape(s: str) -> str:
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"),
                 ("#", "\\#"), ("$", "\\$"), ("{", "\\{"), ("}", "\\}")):
        s = s.replace(a, b)
    return s


# --------------------------------------------------------------------------- loading
def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"[c1-paper] unreadable {path}: {exc}", file=sys.stderr)
        return None


class Results:
    """Every result file under a results root; anything missing is simply None."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.sources: dict[str, Path] = {}      # relative name -> file actually read
        c1 = self.root / "c1"
        self.labels = [l for l in KNOWN if (c1 / l / "results.json").exists()]
        if c1.is_dir():                          # arms this script does not know, kept in the table
            extra = sorted(d.name for d in c1.iterdir()
                           if (d / "results.json").exists() and d.name != "summary"
                           and d.name not in KNOWN
                           and not any(d.name.endswith("_" + v) for v in VARIANTS))
            self.labels += extra
        self.c1 = {l: self._get(f"c1/{l}/results.json") for l in self.labels}
        self.variant = {(l, v): self._get(f"c1/{l}_{v}/results.json")
                        for l in self.labels for v in VARIANTS}
        self.summary = self._get("c1/summary/c1_summary.json")
        self.e3 = {l: self._get(f"e3/{l}/results.json") for l in KNOWN}
        self.t8 = {l: self._get(f"t8/{l}/ablation.json") for l in KNOWN}
        self.precision = self._get("t8/LladaBasePrecision/ablation.json")
        self.t8_labels = [l for l in KNOWN if self.t8.get(l)]
        self.e3_labels = [l for l in KNOWN if self.e3.get(l)]
        self._holm = {}
        for row in (self.summary or {}).get("rows", []):
            self._holm[(row["model"], row["arm"])] = row

    def _get(self, rel: str):
        p = self.root / rel
        if not p.exists():
            return None
        data = _read(p)
        if data is not None:
            self.sources[rel] = p
        return data

    def kind(self, label: str, mask: str, variant: str | None = None):
        res = self.variant.get((label, variant)) if variant else self.c1.get(label)
        return ((res or {}).get("by_mask_kind") or {}).get(mask)

    def holm(self, label: str, mask: str):
        return self._holm.get((label, mask))


# --------------------------------------------------------------------------- number cells
def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def num(x, nd: int = 3) -> str:
    """A signed number for math mode; ``---`` when missing; scientific when absurdly large
    (a quasi-separated Eq. 2 fit)."""
    if not _ok(x):
        return DASH
    if abs(x) >= 100:
        mant, exp = f"{x:.1e}".split("e")
        return f"${mant}{{\\times}}10^{{{int(exp)}}}$"
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return f"${s}$"


def _bare(x, nd=3) -> str:
    return num(x, nd).strip("$")


def est_ci(e, nd: int = 3, marks: str = "") -> str:
    """``$v$`` + marker superscript + ``{\\scriptsize[lo, hi]}`` from ``{value, ci}``; ``---``
    when absent.  ``marks`` is the output of :func:`sup`."""
    if not e or not _ok(e.get("value")):
        return DASH
    ci = e.get("ci") or [None, None]
    if not (_ok(ci[0]) and _ok(ci[1])):
        return num(e["value"], nd) + marks
    lo, hi = _bare(ci[0], nd), _bare(ci[1], nd)
    if "---" in (lo, hi):
        return num(e["value"], nd) + marks
    return f"{num(e['value'], nd)}{marks} {{\\scriptsize[${lo}$, ${hi}$]}}"


def ci_excludes_zero(e) -> bool:
    ci = (e or {}).get("ci")
    return bool(ci and _ok(ci[0]) and _ok(ci[1]) and (ci[0] > 0 or ci[1] < 0))


def pval(p) -> str:
    if not _ok(p):
        return DASH
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def sup(*syms: str) -> str:
    """Markers as one superscript (``$^{*,\\dagger}$``); empty when there are none."""
    syms = [x for x in syms if x]
    return "$^{" + ",".join(syms) + "}$" if syms else ""


def claimed(c) -> bool:
    """Declared AND Holm-rejected (the ``claimed`` field of summarize_c1.py).  A summary of the
    older schema has no such field: it is rebuilt from its two parts, which is its definition."""
    if not c:
        return False
    if "claimed" in c:
        return bool(c["claimed"])
    return bool(c.get("holm_reject")) and bool(c.get("declared"))


def claimed_marker(h, key: str) -> str:
    return "*" if claimed((h or {}).get(key)) else ""


def holm_cell(h, key: str) -> str:
    c = (h or {}).get(key)
    if not c:
        return DASH
    fam = c.get("holm_family")
    if fam == "exploratory":
        return "expl."
    if "holm_reject" not in c:
        return DASH
    return "yes" if c["holm_reject"] else "no"


def declared_marker(c) -> str:
    return "\\dagger" if c and c.get("declared") else ""


def beta(r, key: str):
    """(estimate, markers) of beta_z from a mask arm's regression block, or (None, '')."""
    b = ((r or {}).get("regression") or {}).get(key, {})
    b = b.get("beta_z") if isinstance(b, dict) else None
    if not b or not _ok(b.get("value")):
        return None, ""
    ok = b.get("point_converged", True) and not (b.get("n_nonconverged_draws") or 0)
    return b, ("" if ok else "\\S")


# --------------------------------------------------------------------------- tables
HEADER = ("% Generated by scripts/c1_paper_outputs.py from {src} -- do not edit by hand.\n"
          "% Needs booktabs.  --- is an em dash: the arm or variant is not in the results.\n")
OPEN = "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{2.5pt}\n"
CLOSE = "\\endgroup\n"


def _table(colspec: str, head: list[str], body: list[str], src: str):
    out = [HEADER.format(src=src), OPEN, f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n",
           " & ".join(head) + " \\\\\n\\midrule\n"]
    out += body
    out += ["\\bottomrule\n\\end{tabular}\n", CLOSE]
    return "".join(out)


def _arm_rows(R: Results, labels: list[str], cells, ncols: int, empty=None):
    """One group of rows per label: label in the first row, ``\\addlinespace`` between groups."""
    rows = []
    for gi, label in enumerate(labels):
        if gi:
            rows.append("\\addlinespace\n")
        for mi, mask in enumerate(PRIMARY_MASKS):
            cs = cells(label, mask)
            rows.append(" & ".join([display(label) if mi == 0 else "", MASK_NAME[mask], *cs])
                        + " \\\\\n")
    if not labels:
        rows.append(" & ".join([DASH, ""] + [DASH] * ncols) + " \\\\\n")
    return rows


def labels_for_main(R: Results) -> list[str]:
    """Every main arm, present or not (a missing one is a row of dashes), then the time-split
    and unknown arms that exist."""
    return list(MAIN_ARMS) + [l for l in R.labels if l not in MAIN_ARMS]


def point(e, nd: int, bad: str = "") -> str:
    """The value alone, with ``\\ddagger`` when its CI excludes 0 (and ``bad``: e.g. \\S)."""
    if not e or not _ok(e.get("value")):
        return DASH
    return num(e["value"], nd) + sup("\\ddagger" if ci_excludes_zero(e) else "", bad)


CLAIMED_NOTE = ("% * = claimed: declared (dagger) and rejected by Holm in its family; the family size and\n"
                "% any missing member are in \\COneHolmFootnote (macros_c1.tex).  Holm column: the decision\n"
                "% of the Holm step alone (yes / no; expl. = exploratory arm, outside the families).\n")


def main_table(R: Results) -> str:
    def cells(label, mask):
        r, h = R.kind(label, mask), R.holm(label, mask)
        if r is None:
            return [DASH] * 6
        e1, e2 = r.get("contrast"), r.get("field_contrast")
        b0, m0 = beta(r, "eq2_nofe")
        b1, m1 = beta(r, "eq2_fe")
        return [est_ci(e1, 3, sup(claimed_marker(h, "contrast"), declared_marker(e1))),
                holm_cell(h, "contrast"),
                est_ci(e2, 3, sup(claimed_marker(h, "field_contrast"), declared_marker(e2))),
                holm_cell(h, "field_contrast"),
                point(b0, 2, m0), point(b1, 2, m1)]

    head = ["Arm", "Masks", "E1", "Holm", "E2", "Holm", "$\\beta_z$", "$\\beta_z^{\\mathrm{FE}}$"]
    body = _arm_rows(R, labels_for_main(R), cells, 6)
    return _table("lllclcll", head, body,
                  "results/c1/<Label>/results.json, results/c1/summary/c1_summary.json") + CLAIMED_NOTE


def main_col_table(R: Results) -> str:
    def cells(label, mask):
        r, h = R.kind(label, mask), R.holm(label, mask)
        if r is None:
            return [DASH, DASH]
        e1, e2 = r.get("contrast"), r.get("field_contrast")
        def pt(c, key):
            if not c or not _ok(c.get("value")):
                return DASH
            return num(c["value"], 3) + sup(claimed_marker(h, key), declared_marker(c))
        return [pt(e1, "contrast"), pt(e2, "field_contrast")]

    head = ["Arm", "Masks", "E1", "E2"]
    body = _arm_rows(R, labels_for_main(R), cells, 2)
    return _table("llll", head, body, "results/c1/<Label>/results.json and the Holm summary") + CLAIMED_NOTE


def beta_table(R: Results) -> str:
    """Eq. 2 beta_z, no FE and FE, with 95% CIs (the main table gives the values only)."""
    def cells(label, mask):
        r = R.kind(label, mask)
        if r is None:
            return [DASH, DASH]
        return [est_ci(b, 2, sup(m)) for b, m in (beta(r, "eq2_nofe"), beta(r, "eq2_fe"))]

    body = _arm_rows(R, labels_for_main(R), cells, 2)
    return _table("llll", ["Arm", "Masks", "$\\beta_z$, no FE", "$\\beta_z$, FE"], body,
                  "results/c1/<Label>/results.json (regression.eq2_nofe, eq2_fe)")


def gamma_cell(g, key: str) -> list[str]:
    """[estimate with the Holm marker, Holm p] of one gamma_k from the summary row."""
    c = (g or {}).get(key)
    if not c or not _ok(c.get("value")):
        return [DASH, DASH]
    return [num(c["value"], 2) + sup("*" if c.get("holm_reject") else ""), pval(c.get("p_holm"))]


def gamma_table(R: Results) -> str:
    """gamma_2..gamma_4 of Eq. 3 with state FE (rank 2, 3, 4 against rank 1), and the Holm p
    within the arm (three tests per row, summarize_c1.py)."""
    def cells(label, mask):
        h = R.holm(label, mask)
        out = []
        for key in GAMMAS:
            out += gamma_cell((h or {}).get("gamma"), key)
        return out

    head = ["Arm", "Masks"]
    for k in (2, 3, 4):
        head += [f"$\\gamma_{k}$", f"$p_{{\\mathrm{{Holm}}}}(\\gamma_{k})$"]
    body = _arm_rows(R, labels_for_main(R), cells, 6)
    return _table("llrrrrrr", head, body, "results/c1/summary/c1_summary.json (rows[].gamma)") + (
        "% Eq. 3 with state fixed effects: rank 2, 3, 4 of the retained top ranks against rank 1.  * = Holm-\n"
        "% rejected within the row (three tests).  An arm without a summary row is ---.\n")


def shared_table(R: Results) -> str:
    """The specification check of Proposition 1: the third term E_m[pi(l,s) - a] of the gap split
    (``decomposition[].residual``) at m = 1 and m = 32, and the ICC(1) of a - pi(l,s) within
    states (``icc_residuals``).  Both are zero under independent positions."""
    def cells(label, mask):
        r = R.kind(label, mask)
        if r is None:
            return [DASH] * 3
        out = [est_ci(residual_at(r, m), 3) for m in SHARED_MS]
        icc = (r.get("icc_residuals") or {})
        out.append(est_ci({"value": icc.get("icc"), "ci": icc.get("ci")}, 3) if icc else DASH)
        return out

    head = ["Arm", "Masks"] + [f"$m{{=}}{m}$" for m in SHARED_MS] + ["ICC"]
    body = _arm_rows(R, labels_for_main(R), cells, 3)
    return _table("llccc", head, body, "results/c1/<Label>/results.json (decomposition, icc_residuals)") + (
        "% m = 1, 32: E_m[pi(l,s) - a], the shared-error term (zero under Prop. 1).  ICC: intra-class\n"
        "% correlation of the residuals a - pi(l,s) within states, 95% document-bootstrap CI.\n")


def residual_at(r, m: int):
    """``{value, ci}`` of the shared-error term at field size ``m``, or None."""
    for row in (r or {}).get("decomposition") or []:
        if row.get("m") == m and _ok(row.get("residual")):
            return {"value": row["residual"], "ci": row.get("residual_ci")}
    return None


def sensitivity_table(R: Results) -> str:
    def cells(label, mask):
        base = R.kind(label, mask)
        out = []

        def point(r, key):
            c = (r or {}).get(key)
            return num(c["value"], 3) + sup(declared_marker(c)) if c and _ok(c.get("value")) else DASH

        out.append(point(base, "contrast"))
        out.append(point(R.kind(label, mask, "clean"), "contrast"))
        out.append(point(R.kind(label, mask, "knots5"), "field_contrast"))
        out.append(point(base, "field_contrast"))
        out.append(point(R.kind(label, mask, "knots13"), "field_contrast"))
        out.append(point(R.kind(label, mask, "clean"), "field_contrast"))
        out.append(point(R.kind(label, mask, "additive"), "field_contrast"))
        return out

    head = ["Arm", "Masks", "E1", "E1 clean", "E2, 5 kn.", "E2, 9 kn.", "E2, 13 kn.", "E2 clean",
            "E2 add."]
    body = _arm_rows(R, labels_for_main(R), cells, 7)
    return _table("llccccccc", head, body,
                  "results/c1/<Label>, <Label>_knots5, <Label>_knots13, <Label>_clean, "
                  "<Label>_additive") + (
        "% E2 add. = E2 with the additive context calibrator (plan v2 T6(2)); the interacted one is\n"
        "% primary (the 9-knot column).  E1 takes no calibrator: the additive run's E1 is the primary's.\n"
        "% Markers: dagger = declared.\n")


def _e3_arms(R: Results) -> list[tuple[str, int]]:
    found = set()
    for res in R.e3.values():
        for key in (res or {}).get("comparisons", {}):
            stem = key.rsplit("_vs_", 1)[0]
            parts = stem.split(":")
            if len(parts) == 3 and parts[2][1:].isdigit():
                found.add((parts[1], int(parts[2][1:])))
    return sorted(found | {("topk_conf", k) for k in E3_KS})


def e3_table(R: Results) -> str:
    def side(res, key):
        c = ((res or {}).get("comparisons") or {}).get(key)
        if not c or "error" in c or "type_effect" not in c:
            return [DASH, DASH]
        te = c["type_effect"]
        return [est_ci(te, 3), pval((c.get("type_test") or {}).get("p"))]

    body = []
    labels = list(MAIN_ARMS) + [l for l in R.e3_labels if l not in MAIN_ARMS]
    for gi, label in enumerate(labels):
        if gi:
            body.append("\\addlinespace\n")
        res = R.e3.get(label)
        if res is None:               # no E3 for this arm yet: one row, not three
            body.append(" & ".join([display(label), DASH] + [DASH] * 4) + " \\\\\n")
            continue
        for ai, (pol, k) in enumerate(_e3_arms(R)):
            cells = side(res, f"decoder:{pol}:k{k}_vs_matched") + side(res, f"decoder:{pol}:k{k}_vs_grid")
            name = f"${k}$" if pol == "topk_conf" else f"{tex_escape(pol)}, ${k}$"
            body.append(" & ".join([display(label) if ai == 0 else "", name, *cells]) + " \\\\\n")
    head = ["Arm", "$k$", "Matched uniform", "$p$", "Uniform grid", "$p$"]
    return _table("lllrlr", head, body, "results/e3/<Label>/results.json") + (
        "% type effect = pi_ref(l) - pi_dec(l) averaged over decoder positions; > 0: at equal\n"
        "% confidence the decoder's tokens are right less often.  p: sup-t test of the whole curve.\n")


T8_CONTROLS = (("position", "E1_contrast", "E2_field_contrast"),
               ("token_type", "E1_contrast", "E2_field_contrast"),
               ("temperature_logodds", "E1_contrast", None))


def _t8_row(res, mask: str, effect: str | None, control: str):
    """(without, with, difference) of one T8 row, or Nones."""
    if effect is None:
        return None, None, None
    for r in (res or {}).get("rows", []):
        if r["arm"] == mask and r["control"] == control and r["effect"] == effect:
            return r["without"], r["with"], r["difference"]
    return None, None, None


def t8_table(R: Results) -> str:
    def cells(label, mask):
        res = R.t8.get(label)
        out = []
        for col in (1, 2):                    # 1: E1, 2: E2
            base = None
            vals = []
            for ctrl in T8_CONTROLS:
                wo, wi, diff = _t8_row(res, mask, ctrl[col], ctrl[0])
                if base is None and wo is not None:
                    base = wo
                vals.append((wi, diff))
            out.append(num(base["value"], 3) if base else DASH)
            for wi, diff in vals:
                out.append(num(wi["value"], 3) + sup("\\ddagger" if ci_excludes_zero(diff) else "")
                           if wi else DASH)
        return out

    head = ["Arm", "Masks", "E1", "+pos.", "+type", "+temp.", "E2", "+pos.", "+type", "+temp."]
    body = _arm_rows(R, labels_for_main(R), cells, 8)
    return _table("llrrrrrrrr", head, body, "results/t8/<Label>/ablation.json") + (
        "% E1 = Delta(32)-Delta(1), E2 = field(32)-field(1), without control and with it.  There is\n"
        "% no E2 row for temperature (it recalibrates Delta only).  Markers: \\ddagger = the paired\n"
        "% difference with-minus-without has a 95% CI excluding 0.\n")


def point_only(e) -> str:
    return num(e["value"], 3) if e and _ok(e.get("value")) else DASH


def precision_table(R: Results) -> str:
    rows = [r for r in (R.precision or {}).get("rows", []) if r["control"] == "precision"]
    body = []
    for r in rows:
        eff = {"E1_contrast": "E1", "E2_field_contrast": "E2", "beta_z_noFE": "$\\beta_z$, no FE"}.get(
            r["effect"], tex_escape(r["effect"]))
        mask = MASK_NAME.get(r["arm"], tex_escape(r["arm"]))
        body.append(" & ".join([mask, eff, point_only(r["without"]), point_only(r["with"]),
                                est_ci(r["difference"], 3, sup("\\ddagger" if ci_excludes_zero(r["difference"]) else ""))]) + " \\\\\n")
    if not body:
        body = [" & ".join([DASH] * 5) + " \\\\\n"]
    head = ["Masks", "Effect", "fp16", "4-bit", "Difference"]
    return _table("llrrr", head, body, "results/t8/LladaBasePrecision/ablation.json")


# --------------------------------------------------------------------------- macros
class Macros:
    def __init__(self):
        self.lines: list[str] = []
        self.names: set[str] = set()

    def comment(self, text: str = "") -> None:
        self.lines.append(f"% {text}".rstrip())

    def add(self, name: str, body: str) -> None:
        if not (name.isascii() and name.isalpha()):
            raise ValueError(f"macro name {name!r} is not letters only")
        if name in self.names:
            raise ValueError(f"macro {name!r} defined twice")
        self.names.add(name)
        self.lines.append(f"\\newcommand{{\\{name}}}{{{body}}}")


def _yn(b) -> str:
    return "yes" if b else "no"


def _est_macros(m: Macros, name: str, e, nd: int = 3) -> None:
    """``name`` (value), ``name``Lo and ``name``Hi (the 95% CI) from ``{value, ci}``."""
    if not e or not _ok(e.get("value")):
        return
    m.add(name, f"{e['value']:.{nd}f}")
    ci = e.get("ci") or []
    if len(ci) == 2 and _ok(ci[0]) and _ok(ci[1]):
        m.add(name + "Lo", f"{ci[0]:.{nd}f}")
        m.add(name + "Hi", f"{ci[1]:.{nd}f}")


def sensitivity_macros(R: Results, m: Macros) -> None:
    """E2 under each sensitivity variant (the analysis writes E1 and the field(m) curve only), and
    the additive calibrator against the primary run, cell by cell."""
    p = "COne"
    m.comment("E2 contrast of the sensitivity runs (knots, clean, additive): "
              "COneFieldContrast<Label><Variant><Tag>[Lo|Hi], COneFieldDeclared...")
    cells = agree = 0
    runs = set()
    k_cells = k_agree = 0

    def same_as(fc, prim) -> bool:
        return (fc["value"] > 0) == (prim["value"] > 0) and \
            bool(fc.get("declared")) == bool(prim.get("declared"))

    for label in R.labels:
        for mask in PRIMARY_MASKS:
            tag = label + arm_tag(mask)
            prim = (R.kind(label, mask) or {}).get("field_contrast")
            knots = [(R.kind(label, mask, v) or {}).get("field_contrast") for v in ("knots5", "knots13")]
            if prim and _ok(prim.get("value")) and all(k and _ok(k.get("value")) for k in knots):
                k_cells += 1
                k_agree += all(same_as(k, prim) for k in knots)
            for v, word in VARIANTS.items():
                fc = (R.kind(label, mask, v) or {}).get("field_contrast")
                if not fc or not _ok(fc.get("value")):
                    continue
                vtag = f"{label}{word}{arm_tag(mask)}"
                _est_macros(m, f"{p}FieldContrast{vtag}", fc)
                if "declared" in fc:
                    m.add(f"{p}FieldDeclared{vtag}", _yn(fc["declared"]))
                if v != "additive":
                    continue
                runs.add(label)
                if not prim or not _ok(prim.get("value")):
                    continue
                same = same_as(fc, prim)
                cells += 1
                agree += same
                m.add(f"{p}FieldAdditiveShift{tag}", f"{fc['value'] - prim['value']:.3f}")
                m.add(f"{p}FieldAdditiveAgree{tag}", _yn(same))
    if k_cells:
        m.comment("5 and 13 knots vs the primary 9: cells with both runs, and the cells where E2 keeps "
                  "its sign and its declared flag at both")
        m.add(f"{p}KnotsCells", str(k_cells))
        m.add(f"{p}KnotsAgree", str(k_agree))
    if runs:
        m.comment("additive vs interacted calibrator: runs present, cells compared, and the cells where "
                  "E2 keeps its sign and its declared flag")
        m.add(f"{p}AdditiveRuns", str(len(runs)))
        m.add(f"{p}AdditiveCells", str(cells))
        m.add(f"{p}AdditiveAgree", str(agree))


def shared_error_macros(R: Results, m: Macros) -> None:
    """The third term of the gap split, E_m[pi(l,s) - a] (``decomposition[].residual``), and the
    within-state ICC of the residuals (``icc_residuals``): COne<Label>Residual<m><Tag>[Lo|Hi],
    COne<Label>ICC<Tag>[Lo|Hi]."""
    p = "COne"
    m.comment("shared-error term E_m[pi(l,s) - a] and the within-state ICC of residuals a - pi(l,s) "
              "(zero under Prop. 1)")
    for label in R.labels:
        for mask in PRIMARY_MASKS:
            r = R.kind(label, mask)
            if r is None:
                continue
            tag = arm_tag(mask)
            for row in r.get("decomposition") or []:
                if row.get("m") not in _WORDS:
                    continue
                _est_macros(m, f"{p}{label}Residual{_WORDS[row['m']]}{tag}",
                            residual_at(r, row["m"]))
            icc = r.get("icc_residuals") or {}
            _est_macros(m, f"{p}{label}ICC{tag}", {"value": icc.get("icc"), "ci": icc.get("ci")})


def holm_footnote_macros(R: Results, m: Macros) -> None:
    """The Holm family size and its missing members, for the footnote of the starred tables."""
    p = "COne"
    summ = R.summary or {}
    sizes = [(row.get(k) or {}).get("holm_m") for row in summ.get("rows", [])
             for k in ("contrast", "field_contrast")]
    sizes = [int(x) for x in sizes if _ok(x)]
    size = max(sizes) if sizes else (len(summ["holm_members"]) if summ.get("holm_members") else None)
    missing = summ.get("holm_missing")
    m.comment("footnote of the starred tables: family size per contrast (holm_m) and the family members "
              "without results, counted as p = 1 (holm_missing)")
    if size:
        m.add(f"{p}HolmM", str(size))
    if missing is not None:
        names = []
        for entry in missing:
            lab, _, arm = str(entry).partition(":")
            names.append(f"{display(lab)} ({MASK_NAME.get(arm, tex_escape(arm))})")
        m.add(f"{p}HolmMissingCount", str(len(names)))
        m.add(f"{p}HolmMissing", ", ".join(names) if names else "none")
    text = ("$^{*}$ Claimed: declared (interval excludes 0 and effect above twice the floor) and rejected "
            "by Holm" + (f" at $\\alpha={summ['alpha']:g}$" if _ok(summ.get("alpha")) else "")
            + " within its family"
            + (f" of ${size}$ cells per contrast" if size else "")
            + "; a cell without results counts as $p=1$"
            + (f". Missing: {', '.join(names) if names else 'none'}" if missing is not None else "") + ".")
    m.add(f"{p}HolmFootnote", text)


def gamma_macros(R: Results, m: Macros) -> None:
    """gamma_2..gamma_4 of Eq. 3 (with FE) of the primary rows, their Holm p within the row, and the
    count of rejections."""
    p = "COne"
    m.comment("gamma_2..4 of Eq. 3 with state FE; Holm within each row (three tests): "
              "COne<Label>Gamma<k><Tag>[Lo|Hi], COneHolmPGamma<k>.., COneHolmDecisionGamma<k>..")
    family = reject = rows_rej = 0
    for (label, arm), row in sorted(R._holm.items()):
        if (row.get("contrast") or {}).get("holm_family") not in ("E1", "E2"):
            continue                                  # exploratory arms are not in the count
        if not (label + arm_tag(arm)).isalpha():
            continue
        tag = arm_tag(arm)
        any_rej = False
        for key, word in GAMMAS.items():
            g = (row.get("gamma") or {}).get(key)
            if not g or not _ok(g.get("value")):
                continue
            if label in R.labels:                 # values only for arms whose own results are here
                _est_macros(m, f"{p}{label}Gamma{word}{tag}", g)
            if _ok(g.get("p_holm")):
                m.add(f"{p}HolmPGamma{word}{label}{tag}", f"{g['p_holm']:.4f}")
                m.add(f"{p}HolmDecisionGamma{word}{label}{tag}", _yn(g.get("holm_reject")))
                family += 1
                reject += bool(g.get("holm_reject"))
                any_rej = any_rej or bool(g.get("holm_reject"))
        rows_rej += any_rej
    m.add(f"{p}HolmGammaFamily", str(family))
    m.add(f"{p}HolmGammaReject", str(reject))
    m.add(f"{p}HolmGammaRows", str(rows_rej))


def summary_macros(R: Results, m: Macros) -> None:
    """The headline numbers the per-arm macros.tex files lack."""
    p = "COne"
    # E2 contrast and the declared flags: per (label, primary mask arm).
    m.comment("E2 contrast and declared flags (the per-arm macros.tex have field(m) but not these)")
    for label in R.labels:
        for mask in PRIMARY_MASKS:
            r = R.kind(label, mask)
            if r is None:
                continue
            tag = label + arm_tag(mask)
            fc = r.get("field_contrast") or {}
            if _ok(fc.get("value")):
                m.add(f"{p}FieldContrast{tag}", f"{fc['value']:.3f}")
                if len(fc.get("ci") or []) == 2:
                    m.add(f"{p}FieldContrast{tag}Lo", f"{fc['ci'][0]:.3f}")
                    m.add(f"{p}FieldContrast{tag}Hi", f"{fc['ci'][1]:.3f}")
                m.add(f"{p}FieldDeclared{tag}", _yn(fc.get("declared")))
            if "declared" in (r.get("contrast") or {}):
                m.add(f"{p}ContrastDeclared{tag}", _yn(r["contrast"]["declared"]))
    sensitivity_macros(R, m)
    shared_error_macros(R, m)
    if not R.summary:
        m.comment("no results/c1/summary/c1_summary.json yet: no Holm macros")
        return
    m.comment("Holm decisions from results/c1/summary (scripts/summarize_c1.py), per family")
    fams = {"contrast": "EOne", "field_contrast": "ETwo"}
    count = {k: {"family": 0, "reject": 0, "declared": 0, "both": 0} for k in fams}
    for (label, arm), row in sorted(R._holm.items()):
        tag = label + arm_tag(arm)
        for key, fam in fams.items():
            c = row.get(key) or {}
            if c.get("holm_family") not in ("E1", "E2"):
                continue
            n = count[key]
            n["family"] += 1
            n["reject"] += bool(c.get("holm_reject"))
            n["declared"] += bool(c.get("declared"))
            n["both"] += claimed(c)
            if _ok(c.get("p_holm")):
                m.add(f"{p}HolmP{fam}{tag}", f"{c['p_holm']:.4f}")
            m.add(f"{p}HolmDecision{fam}{tag}", _yn(c.get("holm_reject")))
            m.add(f"{p}Claimed{fam}{tag}", _yn(claimed(c)))
    m.comment("counts over the family members; 'Confirmed' = claimed = Holm-rejected and declared "
              "(CI excludes 0, effect > 2 x floor)")
    for key, fam in fams.items():
        n = count[key]
        m.add(f"{p}Holm{fam}Family", str(n["family"]))
        m.add(f"{p}Holm{fam}Reject", str(n["reject"]))
        m.add(f"{p}Declared{fam}", str(n["declared"]))
        m.add(f"{p}Confirmed{fam}", str(n["both"]))
    if _ok(R.summary.get("alpha")):
        m.add(f"{p}HolmAlpha", f"{R.summary['alpha']:g}")
    m.add(f"{p}ArmsPresent", str(sum(l in MAIN_ARMS for l in R.labels)))
    m.add(f"{p}ArmsExpected", str(len(MAIN_ARMS)))
    m.add(f"{p}TimeSplitPresent", str(sum(l in TIMESPLIT_ARMS for l in R.labels)))
    holm_footnote_macros(R, m)
    gamma_macros(R, m)


def macro_inputs(R: Results, out_tables: Path, tex_root: Path) -> list[tuple[str, str]]:
    """Copy each per-arm macros.tex next to the paper's tables; returns (what, \\input path)."""
    dest = out_tables / "macros"
    items = []
    todo = []
    for label in R.labels:
        todo.append((f"c1-{label}", R.root / "c1" / label / "macros.tex"))
        for v in VARIANTS:
            todo.append((f"c1-{label}-{v}", R.root / "c1" / f"{label}_{v}" / "macros.tex"))
    todo += [(f"e3-{l}", R.root / "e3" / l / "macros.tex") for l in R.e3_labels]
    todo += [(f"t8-{l}", R.root / "t8" / l / "macros.tex") for l in R.t8_labels]
    if R.precision:
        todo.append(("t8-LladaBasePrecision", R.root / "t8" / "LladaBasePrecision" / "macros.tex"))
    for name, src in todo:
        if not src.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        dst = dest / f"{name}.tex"
        shutil.copyfile(src, dst)
        R.sources[f"macros/{name}"] = src
        items.append((name, Path(os.path.relpath(dst, tex_root)).with_suffix("").as_posix()))
    return items


def write_macros(R: Results, out_tables: Path, tex_root: Path) -> list[str]:
    items = macro_inputs(R, out_tables, tex_root)
    m = Macros()
    m.comment("Generated by scripts/c1_paper_outputs.py -- do not edit by hand.")
    m.comment("Prefix COne.  \\input this once; the per-arm files are copies of the analysis' own")
    m.comment("macros.tex (results/c1/<Label>/, e3/, t8/), written with --macro-prefix COne<Label>,")
    m.comment("EThree<Label>, TEight<Label>.  Paths are relative to the paper's directory.")
    missing = [l for l in MAIN_ARMS if l not in R.labels]
    m.comment("arms without results: " + (", ".join(missing) if missing else "none"))
    head = list(m.lines)
    inputs = [f"\\input{{{path}}}" for _, path in items]
    m.lines = []
    summary_macros(R, m)
    text = "\n".join(head + inputs + [""] + m.lines) + "\n"
    (out_tables / "macros_c1.tex").write_text(text, encoding="utf-8")
    return [name for name, _ in items]


# --------------------------------------------------------------------------- figures
def _levels(r, key="delta", x="level", y="delta", lo="ci_low", hi="ci_high"):
    rows = (r or {}).get(key) or []
    rows = [w for w in rows if _ok(w.get(y))]
    return ([w[x] for w in rows], [w[y] for w in rows],
            [w[lo] if _ok(w.get(lo)) else w[y] for w in rows],
            [w[hi] if _ok(w.get(hi)) else w[y] for w in rows])


def _field(r):
    rows = [w for w in (r or {}).get("decomposition") or [] if _ok(w.get("field"))]
    return ([w["m"] for w in rows], [w["field"] for w in rows],
            [w["field_ci"][0] for w in rows], [w["field_ci"][1] for w in rows])


def _curve(ax, xs, ys, lo, hi, color, marker, ls="-", label=None, band=True):
    if not xs:
        return
    if band:
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=1)
    ax.plot(xs, ys, color=color, linewidth=1.2, linestyle=ls, marker=marker, markersize=3.0,
            markeredgecolor="white", markeredgewidth=0.4, zorder=3, label=label)


def _mask_style(F):
    return {"uniform": (F.COLOR["A"], "o", "uniform"),
            "decoder:topk_conf:k4": (F.COLOR["B"], "s", "decoder $k{=}4$")}


def _m_axis(ax, ms):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms, [str(int(v)) for v in ms])
    ax.minorticks_off()
    ax.axhline(0, color="#c3c2b7", linewidth=0.6, zorder=0)


def figure_by_arm(R: Results, F, plt, out: Path, stem: str, which: str) -> list[Path]:
    """Delta(m) or field(m) against m, one panel per arm present; uniform vs decoder k = 4."""
    labels = [l for l in R.labels if any(R.kind(l, k) for k in PRIMARY_MASKS)]
    if not labels:
        return []
    style = _mask_style(F)
    n = len(labels)
    cols = min(FIG_ARMS_PER_ROW, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.9, 1.55 * rows + 0.45), squeeze=False,
                             sharex=True, layout="constrained")
    ms = [1, 2, 4, 8, 16, 32]
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_visible(False)
            continue
        F._style(ax)
        label = labels[i]
        for mask, (color, marker, name) in style.items():
            r = R.kind(label, mask)
            xs, ys, lo, hi = _levels(r) if which == "delta" else _field(r)
            _curve(ax, xs, ys, lo, hi, color, marker, label=name)
            ms = xs or ms
        _m_axis(ax, ms)
        ax.set_title(display(label), fontsize=7)
        if i % cols == 0:
            ax.set_ylabel("confidence $-$ accuracy" if which == "delta"
                          else "field term", fontsize=7)
        if i // cols == rows - 1 or i + cols >= n:
            ax.set_xlabel("field size $m$", fontsize=7)
    h, l = axes.flat[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False, handlelength=2.0)
    paths = F._save(fig, out, stem, f"C1 {'Delta(m)' if which == 'delta' else 'field(m)'} by arm")
    plt.close(fig)
    return paths


def figure_main_arm(R: Results, F, plt, out: Path, label: str) -> list[Path]:
    """Column width: Delta(m) (solid, with 95% CI) and the field term of Prop. 2 (dashed)."""
    if not any(R.kind(label, k) for k in PRIMARY_MASKS):
        return []
    fig, ax = plt.subplots(figsize=(F.COLUMN_IN, 2.15), layout="constrained")
    F._style(ax)
    ms = [1, 2, 4, 8, 16, 32]
    for mask, (color, marker, name) in _mask_style(F).items():
        r = R.kind(label, mask)
        xs, ys, lo, hi = _levels(r)
        _curve(ax, xs, ys, lo, hi, color, marker, label=f"{name}, $\\Delta(m)$")
        fx, fy, *_ = _field(r)
        _curve(ax, fx, fy, None, None, color, None, ls=(0, (3, 1.5)), band=False,
               label=f"{name}, field term")
        ms = xs or ms
    _m_axis(ax, ms)
    ax.set_xlabel("field size $m$")
    ax.set_ylabel("confidence $-$ accuracy")
    ax.set_title(display(label))
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2,
               frameon=False, handlelength=2.2, columnspacing=1.0, fontsize=6)
    ax.set_title("")
    paths = F._save(fig, out, f"c1_delta_m_{label}", f"C1 Delta(m), {label}")
    plt.close(fig)
    return paths


def figure_reliability(R: Results, F, plt, out: Path, label: str) -> list[Path]:
    """Equal-mass reliability bins over all masked positions (``ece.bins``): point values only."""
    have = {k: R.kind(label, k) for k in PRIMARY_MASKS if (R.kind(label, k) or {}).get("ece")}
    if not have:
        return []
    fig, ax = plt.subplots(figsize=(F.COLUMN_IN, 2.35), layout="constrained")
    F._style(ax, grid_y=False)
    ax.plot([0, 1], [0, 1], color=F.MUTED, linewidth=0.8, linestyle="--", zorder=0)
    for mask, r in have.items():
        color, marker, name = _mask_style(F)[mask]
        e = r["ece"]
        bins = [b for b in e.get("bins", []) if _ok(b.get("conf")) and _ok(b.get("acc"))]
        ci = "" if not (_ok(e.get("ci_low")) and _ok(e.get("ci_high"))) else \
            f" [{e['ci_low']:.3f}, {e['ci_high']:.3f}]"
        ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], color=color, marker=marker,
                markersize=3.0, markeredgecolor="white", markeredgewidth=0.4, linewidth=1.0,
                label=f"{name}: ECE {e['ece']:.3f}{ci}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.legend(frameon=False, loc="upper left", fontsize=6, handlelength=1.8)
    paths = F._save(fig, out, f"c1_reliability_{label}", f"C1 reliability, {label}")
    plt.close(fig)
    return paths


def figure_e3(R: Results, F, plt, out: Path, label: str, key="decoder:topk_conf:k4_vs_matched"):
    """E3: reliability of decoder vs matched-uniform states (with bin intervals) and the
    difference curve (pointwise and simultaneous bands)."""
    c = ((R.e3.get(label) or {}).get("comparisons") or {}).get(key)
    if not c or "error" in c or "reliability" not in c or "difference_curve" not in c:
        return []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.3), layout="constrained")
    for ax in (a1, a2):
        F._style(ax, grid_y=False)
    a1.plot([0, 1], [0, 1], color=F.MUTED, linewidth=0.8, linestyle="--", zorder=0)
    for name, color, marker, lab in (("reference", F.COLOR["A"], "o", "matched uniform"),
                                     ("decoder", F.COLOR["B"], "s", "decoder")):
        rel = c["reliability"].get(name) or {}
        bins = [b for b in rel.get("bins", []) if _ok(b.get("conf")) and _ok(b.get("acc"))]
        if not bins:
            continue
        err = [[b["acc"] - b["acc_ci"][0] for b in bins], [b["acc_ci"][1] - b["acc"] for b in bins]]
        wece = rel.get("weighted_ece")
        a1.errorbar([b["conf"] for b in bins], [b["acc"] for b in bins], yerr=err, color=color,
                    marker=marker, markersize=3.0, linewidth=0.9, elinewidth=0.6, capsize=1.2,
                    label=lab + (f" (wECE {wece:.3f})" if _ok(wece) else ""))
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.set_xlabel("confidence")
    a1.set_ylabel("accuracy")
    a1.legend(frameon=False, loc="upper left", fontsize=6)
    dc = c["difference_curve"]
    x, v = dc["conf"], dc["value"]
    ci = dc["ci"]
    a2.fill_between(x, [w[0] for w in ci], [w[1] for w in ci], color=F.COLOR["A"], alpha=0.2,
                    linewidth=0, label="pointwise 95%")
    band = (dc.get("simultaneous") or {}).get("band")
    if band:
        a2.plot(x, [w[0] for w in band], color=F.COLOR["A"], linewidth=0.7, linestyle=":",
                label="simultaneous 95%")
        a2.plot(x, [w[1] for w in band], color=F.COLOR["A"], linewidth=0.7, linestyle=":")
    a2.plot(x, v, color=F.COLOR["A"], linewidth=1.2)
    a2.axhline(0, color="#c3c2b7", linewidth=0.6)
    a2.set_xlabel("confidence")
    a2.set_ylabel("accuracy: matched $-$ decoder")
    te = c["type_effect"]
    a2.set_title(f"type effect {te['value']:+.3f} [{te['ci'][0]:+.3f}, {te['ci'][1]:+.3f}]",
                 fontsize=6.5)
    a2.legend(frameon=False, loc="best", fontsize=6)
    paths = F._save(fig, out, f"c1_e3_{label}", f"C1 E3, {label}")
    plt.close(fig)
    return paths


def make_figures(R: Results, out: Path, main_arm: str) -> list[Path]:
    try:
        import figures_t9 as F          # style of the T9 figures: palette, ACL width, _style, _save
        plt = F._plt()
    except ImportError as exc:
        print(f"[c1-paper] figures skipped: {exc}", file=sys.stderr)
        return []
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    paths += figure_by_arm(R, F, plt, out, "c1_delta_m_all", "delta")
    paths += figure_by_arm(R, F, plt, out, "c1_field_m_all", "field")
    for fn in (figure_main_arm, figure_reliability, figure_e3):
        paths += fn(R, F, plt, out, main_arm)
    return paths


# --------------------------------------------------------------------------- driver
def pick_main(R: Results, wanted: str | None) -> str:
    if wanted:
        return wanted
    return "LladaBase" if "LladaBase" in R.labels else (R.labels[0] if R.labels else "LladaBase")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="results root (c1/, e3/, t8/ below it)")
    ap.add_argument("--out-tables", default="paper/tables/c1")
    ap.add_argument("--out-figures", default="paper/figures/c1")
    ap.add_argument("--tex-root", default=None,
                    help="directory the paper's \\input paths are relative to "
                         "(default: two levels above --out-tables, i.e. paper/)")
    ap.add_argument("--main-arm", default=None,
                    help="label of the arm for the single-arm figures (default LladaBase)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.results)
    if not root.is_dir():
        raise SystemExit(f"{root}: no such results directory")
    R = Results(root)
    tables = Path(a.out_tables)
    tables.mkdir(parents=True, exist_ok=True)
    tex_root = Path(a.tex_root) if a.tex_root else tables.parent.parent
    main_arm = pick_main(R, a.main_arm)

    written = []
    for name, fn in (("c1_main", main_table), ("c1_main_col", main_col_table), ("c1_beta", beta_table),
                     ("c1_gamma", gamma_table), ("c1_shared", shared_table),
                     ("c1_sensitivity", sensitivity_table), ("c1_e3", e3_table),
                     ("c1_t8", t8_table), ("c1_t8_precision", precision_table)):
        p = tables / f"{name}.tex"
        p.write_text(fn(R), encoding="utf-8")
        written.append(p)
    macro_files = write_macros(R, tables, tex_root)
    written.append(tables / "macros_c1.tex")
    if not a.no_figures:
        written += make_figures(R, Path(a.out_figures), main_arm)

    manifest = {"results_root": str(root), "main_arm": main_arm, "arms_present": R.labels,
                "arms_missing": [l for l in MAIN_ARMS if l not in R.labels],
                "summary_present": R.summary is not None, "macro_files": macro_files,
                "sources": {k: {"path": str(p), "sha256": sha256_file(p)}
                            for k, p in sorted(R.sources.items())},
                "outputs": sorted(p.name for p in written),
                "provenance": provenance()}
    (tables / "c1_sources.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[c1-paper] {len(R.labels)} arms ({', '.join(R.labels) or 'none'}); "
          f"{len(written)} files -> {tables}"
          + ("" if a.no_figures else f", {a.out_figures}"))
    miss = manifest["arms_missing"]
    if miss:
        print(f"[c1-paper] no results for: {', '.join(miss)} (their rows are {DASH})")
    no_add = [l for l in R.labels if l in MAIN_ARMS and R.variant.get((l, "additive")) is None]
    if no_add:
        print(f"[c1-paper] no additive-calibrator run (c1/<Label>_additive) for: {', '.join(no_add)} "
              f"(their E2 add. cells are {DASH})", file=sys.stderr)
    if R.summary is None:
        print(f"[c1-paper] no c1/summary/c1_summary.json: no Holm decisions, claimed stars, "
              f"gamma rows or footnote ({DASH})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
