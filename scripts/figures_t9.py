#!/usr/bin/env python3
"""T9 for the paper: the main variant by the pre-set rule, the crossover figures, the C2 macros.

    python scripts/figures_t9.py                  # writes under artifacts/probe/

Reads committed artifacts only (no model, no GPU), from ``--results-dir``:

  rho_check_<tag>.json   how far the rho marker moves q(f(leader)) at rho = 1 (the 0.5 bar)
  val_rescored.json      validation NELBO of both fine-tunes under one protocol
  history_<tag>.jsonl    the fine-tunes' training logs (a consistency check, token counts)
  grid_<tag>.json        (A) and (B) per field for rho x k x rule, the crossover, the identity

and writes, under ``--out-dir``:

  main_variant.json      the rule as written, both candidates' numbers, the choice, the
                         source files with their sha256, and a c3/provenance.py stamp
  figures/c2_crossover_<tag>.{pdf,png}
                         (A) and (B) in nats per field against k, one panel per rho (columns)
                         and selection rule (rows), at ACL column width; shaded: the k
                         interval in which (A) overtakes (B) (hatched if the 95% interval of
                         (A)-(B) still holds 0 there).  One for each fine-tune: the main
                         variant (main text) and the other (appendix)
  figures/c2_untuned_vs_tuned.{pdf,png}
                         (B) at k = 1 and the (A) share at k = 8 against rho, for the
                         untuned model and both fine-tunes (confidence top-k)
  macros_t9.tex          one \\newcommand per number the C2 section cites (prefix CTwo;
                         the naming scheme is in the file's header)

``--no-figures`` skips the figures (for an environment without matplotlib).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c3.provenance import collect as provenance, sha256_file  # noqa: E402

# --- the pre-set rule ---------------------------------------------------------------------
CANDIDATES = ("ft110m-v3", "ft110m-v3-visible")   # the two T9 fine-tunes of train110m-v3
UNTUNED = "untuned-v3"
MARKER_BAR = 0.5
WRITTEN = [
    {"where": "artifacts/README.md, the probe/results row (since commit fc38138)",
     "text": "`rho_check_<tag>.json`: how far the rho marker moves q(f(leader)) at rho=1 "
             "(+0.014 and +0.251; the pre-set bar for the main result was 0.5, "
             "**not met by either**)"},
    {"where": "commit fc38138 (14 Oct), message",
     "text": "neither reaches the 0.5 bar fixed for the main result, and (B) still grows with "
             "rho in both. [...] Main-result choice left to the user."},
    {"where": "documents/final-plan.md, Ojas item 3",
     "text": "**T9:** pick the main variant by the pre-set rule. That is `ft110m-v3-visible`, "
             "whose val NELBO with the marker visible is 0.1751 against 0.1892 "
             "(`val_rescored.json`). Then make the crossover figure."},
]
RULE = {
    "bar": "a fine-tune qualifies for the main result if its rho marker moves q(f(leader)) at "
           f"rho = 1 by at least {MARKER_BAR} (rho_check_<tag>.json: "
           "marker_effect_on_q_copy_at_rho1, own marker minus the marker swapped to rho = 0)",
    "fallback": "if neither qualifies: the lower validation NELBO, both scored under one protocol "
                "(val_rescored.json: runs.<tag>.ema.marker_visible.val_nelbo; EMA weights, as in "
                "the probe grid, with the marker visible, as at evaluation)",
    "both_qualify": "not covered by the rule as written; the script stops",
    "written": WRITTEN,
    "note": "The 0.5 bar is written down (artifacts/README.md, commit fc38138). The fallback, "
            "lower validation NELBO, is not stated as a rule in any committed file; "
            "documents/final-plan.md applies it with these numbers. It is recorded here as "
            "applied. The training logs' NELBO is not comparable between the two: it masks and "
            "scores the marker, which ft110m-v3-visible never saw masked (2.21 vs 0.19).",
}

# --- figure style: dataviz reference palette, slots 1-2 (validated on white: CVD dE 24.7) ---
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#e4e3dc"
COLOR = {"A": "#2a78d6", "B": "#eb6834"}
MARKER = {"A": "o", "B": "s"}                     # shape as well as hue, for grayscale print
TERM = {"A": "(A) dependency", "B": "(B) estimation"}
RULES = ("confidence", "random")
RULE_LABEL = {"confidence": "confidence top-k", "random": "random positions"}
COLUMN_IN = 3.3                                    # ACL column width, inches

# --- macro names: letters only ------------------------------------------------------------
PREFIX = "CTwo"
RHO_WORD = {0.0: "Zero", 0.25: "Quarter", 0.5: "Half", 0.75: "ThreeQuarters", 1.0: "One"}
K_WORD = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
RULE_WORD = {"confidence": "Conf", "random": "Rand"}


def load(path: Path):
    return json.loads(Path(path).read_text())


def rel(path: Path) -> str:
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def last_val(history: Path) -> dict:
    rows = [json.loads(ln) for ln in Path(history).read_text().splitlines() if ln.strip()]
    vals = [r for r in rows if "val_nelbo" in r]
    if not vals:
        raise SystemExit(f"{history}: no validation record")
    return vals[-1]


# --- 1. the decision ----------------------------------------------------------------------
def decide(results: Path) -> dict:
    files = {}

    def read(name):
        files[name] = results / name
        return load(results / name) if name.endswith(".json") else results / name

    val = read("val_rescored.json")
    cand = {}
    for tag in CANDIDATES:
        rc, grid = read(f"rho_check_{tag}.json"), read(f"grid_{tag}.json")
        hist = last_val(read(f"history_{tag}.jsonl"))
        run = val["runs"][tag]
        # The three files must describe the same weights, or the rule compares nothing.
        if rc["checkpoint_sha256"] != grid["model"]["checkpoint_sha256"]:
            raise SystemExit(f"{tag}: rho check and grid ran on different checkpoints")
        for w, key in (("raw", "val_nelbo"), ("ema", "ema_val_nelbo")):
            if abs(run[w]["marker_masked"]["val_nelbo"] - hist[key]) > 1e-9:
                raise SystemExit(f"{tag}: val_rescored ({w}, marker masked) does not reproduce "
                                 f"the training log's last validation: not the final weights")
        effect = rc["marker_effect_on_q_copy_at_rho1"]
        prefix = run["trained_visible_prefix"]
        cand[tag] = {
            "fine_tune": (f"rho marker never masked (visible prefix of {prefix} tokens)" if prefix
                          else "rho marker masked like any token"),
            "marker": "visible" if prefix else "masked",
            "fine_tune_tokens": hist["tokens"],
            "slurm_job": grid["provenance"].get("slurm_job_id"),
            "checkpoint_sha256": grid["model"]["checkpoint_sha256"],
            "probe_weights": grid["model"]["weights"],
            "marker_effect_on_q_copy_at_rho1": effect,
            "reaches_bar": effect >= MARKER_BAR,
            "val_nelbo_ema_marker_visible": run["ema"]["marker_visible"]["val_nelbo"],
            "val_nelbo_raw_marker_visible": run["raw"]["marker_visible"]["val_nelbo"],
            "val_nelbo_ema_marker_masked_training_log_protocol":
                run["ema"]["marker_masked"]["val_nelbo"],
        }
    passing = [t for t in CANDIDATES if cand[t]["reaches_bar"]]
    if len(passing) > 1:
        raise SystemExit("both fine-tunes reach the bar; the pre-set rule does not say "
                         "which to take")
    if passing:
        choice, step = passing[0], f"bar: only {passing[0]} reaches {MARKER_BAR}"
    else:
        choice = min(CANDIDATES, key=lambda t: cand[t]["val_nelbo_ema_marker_visible"])
        step = (f"fallback: neither reaches the {MARKER_BAR} bar, so the lower validation NELBO "
                "(EMA weights, marker visible)")
    alt = next(t for t in CANDIDATES if t != choice)
    c, a = cand[choice], cand[alt]
    return {
        "task": "T9: which fine-tune is the main variant of the C2 probe results",
        "rule": RULE, "candidates": cand, "choice": choice, "other": alt, "decided_by": step,
        "summary": (f"{choice}: marker effect {c['marker_effect_on_q_copy_at_rho1']:+.3f} vs "
                    f"{a['marker_effect_on_q_copy_at_rho1']:+.3f} ({alt}), bar {MARKER_BAR}; "
                    f"validation NELBO {c['val_nelbo_ema_marker_visible']:.4f} vs "
                    f"{a['val_nelbo_ema_marker_visible']:.4f} nats/token"),
        "checks": {"rho_check_and_grid_same_checkpoint": True,
                   "val_rescored_marker_masked_reproduces_training_log": True},
        "sources": {rel(p): sha256_file(p) for p in files.values()},
        "provenance": provenance(),
    }


# --- 2. the figures -----------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.titlepad": 3,
        "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "hatch.color": MUTED, "hatch.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # TrueType, not Type 3 (venue PDF checks)
    })
    return plt


def _style(ax, grid_y=True) -> None:
    if grid_y:
        ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=1.5)


def _save(fig, out: Path, stem: str, title: str) -> list[Path]:
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        meta = {"Title": title}
        if ext == "pdf":
            meta["CreationDate"] = None              # byte-stable output
        fig.savefig(p, dpi=300, metadata=meta)
        paths.append(p)
    return paths


def cells(grid: dict) -> dict:
    return {(c["rule"], c["k"], c["rho"]): c for c in grid["cells"]}


def crossover_figure(grid: dict, out: Path, stem: str, title: str) -> list[Path]:
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    c = cells(grid)
    ks = sorted(grid["ks"])
    rhos = sorted({key[2] for key in c})
    rules = [r for r in RULES if r in grid["rules"]]
    top = max(c[key][f"{t}_per_field"]["ci95"][1] for key in c for t in "AB")
    top = math.ceil(top / 0.05) * 0.05 + 0.03
    fig, axes = plt.subplots(len(rules), len(rhos), figsize=(COLUMN_IN, 0.9 * len(rules) + 0.75),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, wspace=0.03, hspace=0.04)
    unsure = False
    for r, rule in enumerate(rules):
        for j, rho in enumerate(rhos):
            ax = axes[r][j]
            _style(ax)
            x = grid["crossover"][f"{rule}/rho={rho}"]
            first, sure = x["first_k_A_exceeds_B"], x["first_k_A_exceeds_B_ci"]
            if first is not None:
                lo = ks[max(ks.index(first) - 1, 0)]
                if sure == first:
                    ax.axvspan(lo, first, facecolor=BAND, edgecolor="none", zorder=0)
                else:                     # A > B in mean only: the interval of A - B holds 0
                    unsure = True
                    ax.axvspan(lo, first, facecolor="none", edgecolor=MUTED, hatch="//////",
                               linewidth=0, zorder=0)
            for t in ("A", "B"):
                f = [c[(rule, k, rho)][f"{t}_per_field"] for k in ks]
                m = [max(v["mean"], 0.0) for v in f]          # TC, KL >= 0; -1e-17 is rounding
                ax.fill_between(ks, [v["ci95"][0] for v in f], [v["ci95"][1] for v in f],
                                color=COLOR[t], alpha=0.16, linewidth=0, zorder=1)
                ax.plot(ks, m, color=COLOR[t], linewidth=1.2, marker=MARKER[t],
                        markersize=3.3 if t == "A" else 2.9, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=3, solid_capstyle="round")
            ax.set_xscale("log", base=2)
            ax.set_xticks(ks, [str(k) for k in ks])
            ax.minorticks_off()
            ax.set_xlim(ks[0] / 1.18, ks[-1] * 1.18)
            ax.set_ylim(-0.02 * top, top)
            ax.set_yticks([0, 0.2, 0.4] if top > 0.3 else [0, 0.1, 0.2])
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
            if r == 0:
                ax.set_title(f"ρ = {rho:g}")
        side = axes[r][-1]
        side.yaxis.set_label_position("right")
        side.set_ylabel(RULE_LABEL[rule].replace(" ", "\n", 1), rotation=270, va="bottom",
                        labelpad=1, fontsize=6.5)
    fig.supxlabel("tokens per step k", fontsize=7, color=INK2)
    fig.supylabel("nats per field", fontsize=7, color=INK2)
    handles = [Line2D([], [], color=COLOR[t], linewidth=1.2, marker=MARKER[t], markersize=3.3,
                      markeredgecolor="white", markeredgewidth=0.5, label=TERM[t]) for t in "AB"]
    handles.append(Patch(facecolor=BAND, edgecolor="none", label="(A) overtakes (B)"))
    if unsure:
        handles.append(Patch(facecolor="none", edgecolor=MUTED, hatch="//////", linewidth=0,
                             label="(A) > (B) in mean only"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3 if not unsure else 2,
               frameon=False, handlelength=1.8, columnspacing=1.0, handletextpad=0.5)
    paths = _save(fig, out, stem, title)
    plt.close(fig)
    return paths


def comparison_figure(grids: dict, order: list, label: dict, out: Path, stem: str,
                      rule: str = "confidence") -> list[Path]:
    """(B) at k = 1 (log scale) and the (A) share at k = 8 against rho, one line per model."""
    plt = _plt()
    style = dict(zip(order, ({"color": INK, "marker": "o", "linestyle": "-"},
                             {"color": INK2, "marker": "s", "linestyle": (0, (3, 1.5))},
                             {"color": MUTED, "marker": "^", "linestyle": (0, (1, 1.2))})))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.75), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.08)
    for tag in order:
        c = cells(grids[tag])
        rhos = sorted({key[2] for key in c})
        kw = dict(color=style[tag]["color"], marker=style[tag]["marker"],
                  linestyle=style[tag]["linestyle"], linewidth=1.1, markersize=3.2,
                  markeredgecolor="white", markeredgewidth=0.5, label=label[tag])
        ax1.plot(rhos, [c[(rule, 1, r)]["B_per_field"]["mean"] for r in rhos], **kw)
        ax2.plot(rhos, [100 * max(c[(rule, 8, r)]["share_A"], 0.0) for r in rhos], **kw)
    for ax in (ax1, ax2):
        _style(ax)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
        ax.set_xlabel("dependency ρ", fontsize=7)
        ax.set_xlim(-0.06, 1.06)
    ax1.set_yscale("log")
    ax1.set_yticks([0.03, 0.1, 0.3, 1, 3, 10], ["0.03", "0.1", "0.3", "1", "3", "10"])
    ax1.minorticks_off()
    ax1.set_ylim(0.02, 14)
    ax1.set_title("(B) at k = 1", fontsize=7)
    ax1.set_ylabel("nats per field", fontsize=7)
    ax2.set_ylim(-3, 103)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.axhline(50, color=MUTED, linewidth=0.8, zorder=0)
    ax2.text(1.04, 47, "crossover", fontsize=5.8, color=INK2, ha="right", va="top")
    ax2.set_title("(A) share at k = 8", fontsize=7)
    ax2.set_ylabel("% of (A)+(B)", fontsize=7)
    fig.legend(*ax1.get_legend_handles_labels(), loc="outside upper center", ncol=2,
               frameon=False, handlelength=2.6, columnspacing=1.2, handletextpad=0.5)
    paths = _save(fig, out, stem, f"T9: untuned vs fine-tuned, {RULE_LABEL[rule]}")
    plt.close(fig)
    return paths


# --- 3. the macros ------------------------------------------------------------------------
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


def nats(v: float) -> str:
    return f"{0.0 if -1e-9 < v < 0 else v:.3f}"


def pct(v: float) -> str:
    return f"{100 * (0.0 if -1e-9 < v < 0 else v):.0f}"


def count(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def sci(v: float) -> str:
    mant, exp = f"{v:.1e}".split("e")
    return f"\\ensuremath{{{mant}\\times 10^{{{int(exp)}}}}}"


def cross_k(k) -> str:
    return "--" if k is None else str(k)


def macros(dec: dict, grids: dict, main: str, alt: str) -> str:
    variants = {main: "", alt: "Alt", UNTUNED: "Untuned"}
    m = Macros()
    m.comment("Generated by scripts/figures_t9.py -- do not edit by hand.")
    m.comment(f"T9, C2 on the synthetic JSON probe.  commit {dec['provenance']['git_commit']}")
    m.comment(f"Main variant {main} by the pre-set rule (artifacts/probe/main_variant.json); "
              f"Alt = {alt}; Untuned = {UNTUNED}.")
    m.comment("Sources: artifacts/probe/results/grid_<tag>.json, rho_check_<tag>.json, "
              "val_rescored.json, history_<tag>.jsonl.")
    m.comment()
    m.comment("Grid cells: \\CTwo[Alt|Untuned]<Rule><Q>at<Rho><K>[Lo|Hi], e.g. "
              "\\CTwoConfAatRhoOneKEight")
    m.comment("  no variant word = the main variant; Rule: Conf = confidence top-k, "
              "Rand = random positions")
    m.comment("  Q: A = (A) dependency, B = (B) estimation, nats per field (mean over the "
              "instances at that rho);")
    m.comment("     ShareA = (A) as a percentage of (A)+(B)")
    m.comment("  Rho: RhoZero, RhoQuarter, RhoHalf, RhoThreeQuarters, RhoOne;  "
              "K: KOne, KTwo, KFour, KEight")
    m.comment("  Lo/Hi (main variant only): 95% bootstrap interval, instances resampled")
    m.comment("Crossover: \\CTwo[Alt|Untuned]<Rule>CrossKat<Rho> = the first k at which mean (A) "
              "> mean (B);")
    m.comment("  ...CrossKCIat<Rho> = the first k at which the 95% interval of (A)-(B) is above 0;")
    m.comment("  -- = at no k <= 8.  ...CrossRhoMin = the smallest rho with a crossover, "
              "...ShareAMax = the largest ShareA.")

    for tag, word in variants.items():
        g = grids[tag]
        c = cells(g)
        ks = sorted(g["ks"])
        rhos = sorted({key[2] for key in c})
        m.comment()
        m.comment(f"{tag}{' (main variant)' if tag == main else ''}: "
                  f"{g['n_instances']} instances, {g['n_boot']} resamples")
        for rule in RULES:
            rw = RULE_WORD[rule]
            for rho in rhos:
                for k in ks:
                    cell = c[(rule, k, rho)]
                    at = f"at{'Rho' + RHO_WORD[rho]}K{K_WORD[k]}"
                    for q in ("A", "B"):
                        est = cell[f"{q}_per_field"]
                        m.add(f"{word}{rw}{q}{at}", nats(est["mean"]))
                        if tag == main:
                            m.add(f"{word}{rw}{q}{at}Lo", nats(est["ci95"][0]))
                            m.add(f"{word}{rw}{q}{at}Hi", nats(est["ci95"][1]))
                    m.add(f"{word}{rw}ShareA{at}", pct(cell["share_A"]))
            crossed = []
            for rho in rhos:
                x = g["crossover"][f"{rule}/rho={rho}"]
                m.add(f"{word}{rw}CrossKatRho{RHO_WORD[rho]}", cross_k(x["first_k_A_exceeds_B"]))
                m.add(f"{word}{rw}CrossKCIatRho{RHO_WORD[rho]}",
                      cross_k(x["first_k_A_exceeds_B_ci"]))
                if x["first_k_A_exceeds_B"] is not None:
                    crossed.append(rho)
            m.add(f"{word}{rw}CrossRhoMin", f"{min(crossed):g}" if crossed else "--")
            m.add(f"{word}{rw}ShareAMax", pct(max(x["share_A"] for x in g["cells"]
                                                  if x["rule"] == rule)))
        ident = g["identity"]
        m.add(f"{word}IdentityMaxErr", sci(ident["max_abs_err_nats"]))
        m.add(f"{word}IdentityStepsChecked", count(ident["steps_checked"]))
        m.add(f"{word}IdentityStepsTotal", count(ident["steps_total"]))

    m.comment()
    m.comment("Design and identity across the three plan-v2 runs")
    g = grids[main]
    per_rho = {cell["n_instances"] for cell in g["cells"]}
    m.add("ProbeNInstances", count(g["n_instances"]))
    m.add("ProbeNInstancesPerRho", count(per_rho.pop()) if len(per_rho) == 1 else "--")
    m.add("ProbeNBoot", count(g["n_boot"]))
    m.add("IdentityMaxErrAll", sci(max(grids[t]["identity"]["max_abs_err_nats"]
                                       for t in variants)))
    m.add("IdentityTol", "\\ensuremath{10^{-6}}")

    m.comment()
    m.comment("The main-variant decision: marker bar, marker effect at rho = 1, validation NELBO")
    m.comment("(nats/token, EMA weights, marker visible), fine-tuning tokens (millions)")
    cand = dec["candidates"]
    m.add("MarkerBar", f"{MARKER_BAR:g}")
    for tag, word in ((main, ""), (alt, "Alt")):
        m.add(f"{word}MarkerEffect", f"{cand[tag]['marker_effect_on_q_copy_at_rho1']:.3f}")
        m.add(f"{word}ValNelbo", f"{cand[tag]['val_nelbo_ema_marker_visible']:.4f}")
        m.add(f"{word}FtTokensM", f"{cand[tag]['fine_tune_tokens'] / 1e6:.0f}")
        m.add(f"{word}Variant", tag)
    return "\n".join(m.lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "artifacts/probe/results"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/probe"))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    results, out = Path(a.results_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dec = decide(results)
    main_tag, alt_tag = dec["choice"], dec["other"]
    (out / "main_variant.json").write_text(json.dumps(dec, indent=1) + "\n")
    print(f"[t9] main variant: {main_tag} ({dec['decided_by']}): {dec['summary']}")

    grids = {t: load(results / f"grid_{t}.json") for t in (main_tag, alt_tag, UNTUNED)}
    (out / "macros_t9.tex").write_text(macros(dec, grids, main_tag, alt_tag))
    print(f"[t9] macros -> {out / 'macros_t9.tex'}")

    if not a.no_figures:
        figs = out / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        written = []
        for tag, role in ((main_tag, "main variant"), (alt_tag, "other fine-tune")):
            written += crossover_figure(grids[tag], figs, f"c2_crossover_{tag}",
                                        f"T9 crossover, {tag} ({role})")
        cand = dec["candidates"]
        label = {t: f"fine-tuned, marker {cand[t]['marker']}" for t in (main_tag, alt_tag)}
        label[UNTUNED] = "untuned"
        written += comparison_figure(grids, [main_tag, alt_tag, UNTUNED], label, figs,
                                     "c2_untuned_vs_tuned")
        print(f"[t9] figures -> {', '.join(p.name for p in written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
