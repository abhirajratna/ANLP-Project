"""Figures for the mid-evaluation slides, drawn from the committed artefacts.

    .venv/bin/python documents/mid-eval-slides/make_figures.py

Every number plotted here is read from a JSON file under artifacts/; nothing
is typed in by hand. Figures are drawn at the size they appear on the slide
(16:9 Beamer, 144 mm text width), so the fonts match the slide text.
"""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
OUT = Path(__file__).resolve().parent / "figures"

# Ink and series colours (validated: adjacent and all-pairs CVD separation for
# the three categorical slots; the blue ramp passes the ordinal checks).
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

MM = 1 / 25.4
FULL_W = 144 * MM


def _setup_fonts() -> None:
    """Use Fira Sans (the slide face) when tectonic has cached it."""
    pattern = os.path.expanduser("~/.cache/tectonic/**/FiraSans-*.otf")
    files = glob.glob(pattern, recursive=True)
    family = "DejaVu Sans"
    if files:
        for f in files:
            font_manager.fontManager.addfont(f)
        family = "Fira Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.weight": "normal",
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.titleweight": "medium",
            "axes.titlecolor": INK,
            "axes.labelsize": 7.5,
            "axes.labelcolor": INK2,
            "axes.edgecolor": BASE,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": BASE,
            "ytick.color": BASE,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "xtick.major.size": 2.5,
            "ytick.major.size": 0,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "pdf.fonttype": 42 if family == "DejaVu Sans" else 3,
            "savefig.transparent": True,
        }
    )


def _load(rel: str) -> dict:
    return json.loads((ART / rel).read_text())


def _num(v: float, fmt: str = "+.1f") -> str:
    """Format with a real minus sign."""
    return format(v, fmt).replace("-", "\u2212")


def _pp(x):
    """Probability units to percentage points."""
    if isinstance(x, (list, tuple)):
        return [_pp(v) for v in x]
    return 100.0 * x


def _log2_m_axis(ax, ms=(1, 2, 4, 8, 16, 32)) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.minorticks_off()
    ax.set_xlim(ms[0] / 1.25, ms[-1] * 1.25)


def _zero(ax) -> None:
    ax.axhline(0, color=BASE, lw=0.8, zorder=1)


def _series(ax, m, y, color, marker, ci=None, z=3, label=None, ms=4.2):
    if ci is not None:
        lo = [c[0] for c in ci]
        hi = [c[1] for c in ci]
        ax.fill_between(m, lo, hi, color=color, alpha=0.12, lw=0, zorder=z - 1)
    ax.plot(
        m, y, color=color, marker=marker, ms=ms, mec=SURFACE, mew=0.9,
        zorder=z, label=label,
    )


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


GAP_LEGEND = [
    Line2D([], [], color=BLUE, marker="o", ms=4.2, mec=SURFACE, mew=0.9,
           label=r"observed $\Delta(m)$, 95% CI"),
    Line2D([], [], color=ORANGE, marker="s", ms=4, mec=SURFACE, mew=0.9,
           label=r"own-confidence prediction $\hat\pi(\ell)$"),
    Line2D([], [], color=AQUA, marker="^", ms=4.4, mec=SURFACE, mew=0.9,
           label=r"context-aware prediction $\hat\pi(\ell, s)$"),
    Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label="ECE"),
]


def _gap_panel(ax, m, delta, delta_ci, own, ctx, ece_pp, title):
    _zero(ax)
    ax.axhline(ece_pp, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    _series(ax, m, own, ORANGE, "s", z=3, ms=4)
    _series(ax, m, ctx, AQUA, "^", z=4, ms=4.4)
    _series(ax, m, delta, BLUE, "o", ci=delta_ci, z=5)
    _log2_m_axis(ax)
    ax.set_title(title, loc="left", pad=4)


# --------------------------------------------------------------------------
# T0: four simulated scenarios
# --------------------------------------------------------------------------

T0_TITLES = {
    "calibrated": "Calibrated\nin every context",
    "uniform": "Same miscalibration\neverywhere",
    "context": "Miscalibration\ndepends on context",
    "misreading": "Errors shared\nwithin a state",
}


def fig_t0_scenarios() -> None:
    t0 = _load("t0/table.json")
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 35 * MM), sharey=True)
    for ax, key in zip(axes, T0_TITLES):
        sc = t0["scenarios"][key]
        s = sc["series"]
        m = s["delta"]["m"]
        _gap_panel(
            ax, m,
            _pp(s["delta"]["point"]), _pp(s["delta"]["ci"]),
            _pp(s["delta_own"]["point"]), _pp(s["delta_ctx"]["point"]),
            _pp(sc["ece"]["value"]), T0_TITLES[key],
        )
        ax.set_xlabel("field size $m$")
        ax.title.set_fontsize(7.4)
        if key == "context":
            i8 = m.index(8)
            v = _pp(s["delta"]["point"][i8])
            ax.annotate(
                f"{v:.1f} pp at $m$ = 8", xy=(8, v), xytext=(1.05, 14.2),
                fontsize=6.6, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                shrinkA=0, shrinkB=3),
            )
    axes[0].set_ylabel("confidence − accuracy (pp)")
    axes[0].set_ylim(-1.5, 15.2)
    fig.subplots_adjust(wspace=0.12, top=0.80)
    fig.legend(handles=GAP_LEGEND, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.94), handlelength=2.0, columnspacing=1.6)
    _save(fig, "t0_scenarios")


def fig_t0_field_growth() -> None:
    t0 = _load("t0/table.json")
    fg = t0["field_growth"]
    keys = sorted(fg, key=float)
    fig, axes = plt.subplots(1, len(keys), figsize=(168 * MM, 28 * MM), sharey=True)
    for ax, k in zip(axes, keys):
        g = fg[k]
        _zero(ax)
        _series(ax, g["m"], _pp(g["field"]), BLUE, "o", ci=_pp(g["ci"]))
        _log2_m_axis(ax)
        ax.set_title(f"noise $\\sigma$ = {float(k):g}", loc="left", pad=3)
        ax.set_xlabel("field size $m$")
        peak = max(range(len(g["m"])), key=lambda i: g["field"][i])
        ax.annotate(
            f"peak {_num(_pp(g['field'][peak]))}", xy=(g["m"][peak], _pp(g["field"][peak])),
            xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.2,
            color=INK2,
        )
    axes[0].set_ylabel("true field term (pp)")
    axes[0].set_ylim(-11, 6.5)
    fig.subplots_adjust(wspace=0.1)
    _save(fig, "t0_field_growth")


# --------------------------------------------------------------------------
# Estimator validation (proposal Eq. 2 on known truth)
# --------------------------------------------------------------------------

def fig_estimator() -> None:
    """Spurious beta_z under a heteroskedastic null, by spline knots."""
    ev = _load("estimator_validation.json")["checks"]
    fig, a2 = plt.subplots(figsize=(80 * MM, 47 * MM))
    grid = ev["heteroskedastic_grid"]
    taus = sorted(grid, key=float)
    a2.axvspan(8.2, 9.8, color=BLUE, alpha=0.08, lw=0, zorder=0)
    a2.text(9, 0.131, "fixed: 9 knots", ha="center", va="bottom", fontsize=6.4,
            color=INK2)
    _zero(a2)
    for tau, col in zip(taus, RAMP4):
        ks = sorted(grid[tau], key=int)
        a2.plot([int(k) for k in ks], [grid[tau][k] for k in ks], color=col,
                marker="o", ms=4, mec=SURFACE, mew=0.9, label=f"τ = {float(tau):g}",
                zorder=3)
    a2.set_xticks([5, 7, 9, 13, 17])
    a2.set_xlim(4.2, 17.8)
    a2.set_ylim(-0.065, 0.15)
    a2.set_xlabel("spline knots in the control $f(\\ell)$")
    a2.set_ylabel(r"spurious $\beta_z$ (true effect is 0)")
    a2.legend(title="noise dispersion", title_fontsize=6.8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), handlelength=1.6, borderaxespad=0.2,
              alignment="left")
    worst = ev["worst_spurious_beta_z"]
    a2.annotate(_num(worst, "+.2f"), xy=(5, worst), xytext=(6, 0), textcoords="offset points",
                fontsize=6.4, color=INK, va="center")
    _save(fig, "estimator_validation")


# --------------------------------------------------------------------------
# Contamination screen (T4)
# --------------------------------------------------------------------------

def fig_contamination() -> None:
    fig, ax = plt.subplots(figsize=(62 * MM, 29 * MM))
    thr = None
    for key, name, col, mk in [
        ("llada-8b-base", "LLaDA-8B-Base", BLUE, "o"),
        ("dream-7b-base", "Dream-7B-Base", ORANGE, "s"),
    ]:
        c = _load(f"contamination_{key}.json")
        thr = c["threshold"]
        runs = [d["run_length"] for d in c["docs"].values()]
        n = len(runs)
        rs = list(range(0, c["continuation_tokens"] + 1))
        share = [sum(r >= x for r in runs) / n for x in rs]
        xs = [x for x, s in zip(rs, share) if s > 0]
        ys = [s for s in share if s > 0]
        ax.step(xs, ys, where="post", color=col, lw=1.4,
                label=f"{name}: {c['n_flagged']} / {n:,} flagged")
    ax.axvline(thr, color=INK2, lw=0.8, zorder=1)
    ax.text(thr + 0.8, 0.012, f"flag at ≥ {thr}\nof 50 tokens", fontsize=6.4,
            color=INK2, va="bottom", ha="left")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 1.3)
    ax.set_xlim(0, 50)
    ax.set_yticks([1, 0.1, 0.01, 0.001])
    ax.set_yticklabels(["100%", "10%", "1%", "0.1%"])
    ax.minorticks_off()
    ax.set_xlabel("longest verbatim run x (tokens of 50)")
    ax.set_ylabel("documents with a run ≥ x")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 0.98), handlelength=1.2,
              borderaxespad=0.1, ncol=1, labelspacing=0.25)
    _save(fig, "contamination")


# --------------------------------------------------------------------------
# T2 pilot, LLaDA-8B-Base fp16
# --------------------------------------------------------------------------

def _t2():
    return _load("t2_pilot/analysis/results.json")["by_mask_kind"]


def fig_t2_gap() -> None:
    t2 = _t2()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 34 * MM), sharey=True)
    for ax, key, title in [
        (axes[0], "decoder", "Decoder-induced masks"),
        (axes[1], "uniform", "Uniform masks"),
    ]:
        a = t2[key]
        d = a["decomposition"]
        m = [r["m"] for r in d]
        _gap_panel(
            ax, m,
            _pp([r["delta"] for r in d]), _pp([r["delta_ci"] for r in d]),
            _pp([r["delta_own"] for r in d]), _pp([r["delta_ctx"] for r in d]),
            _pp(a["ece"]["ece"]),
            f"{title} ({a['n_states']:,} states)",
        )
        ax.set_xlabel("field size $m$")
        c = a["contrast"]
        ax.text(
            0.03, 0.97,
            f"$\\Delta(32)-\\Delta(1)$ = {_num(_pp(c['value']))} pp\n"
            f"95% CI [{_num(_pp(c['ci'][0]))}, {_num(_pp(c['ci'][1]))}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6, color=INK,
            linespacing=1.4,
        )
    axes[0].set_ylabel("confidence − accuracy (pp)")
    axes[0].set_ylim(-1.3, 2.9)
    fig.subplots_adjust(wspace=0.08, top=0.86)
    fig.legend(handles=GAP_LEGEND, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.95), handlelength=2.0, columnspacing=1.6)
    _save(fig, "t2_gap")


def fig_t2_reliability() -> None:
    t2 = _t2()
    fig, ax = plt.subplots(figsize=(56 * MM, 50 * MM))
    lo, hi = 0.02, 0.999
    ax.plot([lo, hi], [lo, hi], color=BASE, lw=0.8, zorder=1)
    for key, name, col, mk in [
        ("uniform", "uniform masks", ORANGE, "s"),
        ("decoder", "decoder masks", BLUE, "o"),
    ]:
        bins = t2[key]["ece"]["bins"]
        ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], color=col, lw=1.0,
                marker=mk, ms=4.2 if mk == "o" else 3.8, mec=SURFACE, mew=0.8,
                label=f"{name} (ECE {t2[key]['ece']['ece']:.3f})", zorder=3)
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    ticks = [0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
    labels = ["0.05", "0.2", "0.5", "0.8", "0.95", "0.99"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.minorticks_off()
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(axis="both")
    ax.set_xlabel("mean confidence in bin (log-odds scale)")
    ax.set_ylabel("accuracy in bin")
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_[::-1], loc="upper left", handlelength=1.6,
              borderaxespad=0.1)
    low = sum(b["conf"] < 0.1 for b in t2["decoder"]["ece"]["bins"])
    ax.annotate(f"{low} of 15 decoder bins\nsit below 0.1", xy=(0.06, 0.045),
                xytext=(0.3, 0.06), fontsize=6.2, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=4))

    _save(fig, "t2_reliability")


def fig_t2_forest() -> None:
    t2 = _t2()
    rows = [
        ("eq2", "beta_z", r"$\beta_z$  (order statistic)"),
        ("eq3", "gamma_2", r"$\gamma_2$  (rank 2 vs winner)"),
        ("eq3", "gamma_3", r"$\gamma_3$  (rank 3)"),
        ("eq3", "gamma_4", r"$\gamma_4$  (rank 4)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 29 * MM), sharex=True, sharey=True)
    for ax, fe, title in [(axes[0], "nofe", "Without state fixed effects"),
                          (axes[1], "fe", "With state fixed effects")]:
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        for j, (eq, coef, _) in enumerate(rows):
            y = len(rows) - 1 - j
            for key, col, mk, off in [("decoder", BLUE, "o", 0.14),
                                      ("uniform", ORANGE, "s", -0.14)]:
                r = t2[key]["regression"][f"{eq}_{fe}"][coef]
                ax.plot(r["ci"], [y + off] * 2, color=col, lw=1.5, zorder=2)
                ax.plot([r["value"]], [y + off], ls="none", marker=mk,
                        ms=4.6 if mk == "o" else 4.2, color=col, mec=SURFACE, mew=0.9,
                        zorder=3)
        ax.set_title(title, loc="left", pad=4)
        ax.set_xlabel("coefficient (log-odds)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[2] for r in rows][::-1])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    axes[0].set_xlim(-0.17, 0.17)
    axes[0].set_xticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
    handles = [
        Line2D([], [], color=BLUE, marker="o", ms=4.6, mec=SURFACE, mew=0.9,
               label="decoder masks"),
        Line2D([], [], color=ORANGE, marker="s", ms=4.2, mec=SURFACE, mew=0.9,
               label="uniform masks"),
    ]
    fig.subplots_adjust(wspace=0.06, top=0.86)
    fig.legend(handles=handles, loc="lower right", ncol=2, bbox_to_anchor=(0.9, 0.93),
               handlelength=2.0)
    _save(fig, "t2_forest")


def fig_precision() -> None:
    """Pool winners under decoder masks: proposal-era 4-bit pilot vs T2 fp16."""
    s0 = _load("stage0_gate_decoder.json")
    t2 = _t2()["decoder"]
    fig, ax = plt.subplots(figsize=(68 * MM, 42 * MM))
    _zero(ax)
    # The Stage 0 file uses agree - conf; flip to conf - agree.
    g = s0["commitment_gap"]
    m4 = [r["m"] for r in g]
    y4 = [-_pp(r["gap"]) for r in g]
    ci4 = [[-_pp(r["ci_high"]), -_pp(r["ci_low"])] for r in g]
    _series(ax, m4, y4, ORANGE, "s", ci=ci4, z=3, ms=4)
    d = t2["decomposition"]
    _series(ax, [r["m"] for r in d], _pp([r["delta"] for r in d]), BLUE, "o",
            ci=_pp([r["delta_ci"] for r in d]), z=4)
    _log2_m_axis(ax)
    ax.set_xlabel("field size $m$")
    ax.set_ylabel("winner's confidence − accuracy (pp)")
    ax.set_ylim(-3.5, 10.5)
    handles = [
        Line2D([], [], color=ORANGE, marker="s", ms=4, mec=SURFACE, mew=0.9,
               label=f"4-bit NF4, Stage 0 pilot ({s0['n_documents']} docs)"),
        Line2D([], [], color=BLUE, marker="o", ms=4.2, mec=SURFACE, mew=0.9,
               label=f"fp16, T2 pilot ({t2['n_docs']} other docs)"),
    ]
    ax.legend(handles=handles, loc="upper left", handlelength=1.8, borderaxespad=0.1)
    _save(fig, "precision")


def fig_power() -> None:
    g1 = _load("t2_pilot/gate_g1.json")
    dmin = g1["config"]["delta_min"]
    fig, ax = plt.subplots(figsize=(70 * MM, 43 * MM))
    ns = [10 * (2000 / 10) ** (i / 200) for i in range(201)]
    k = math.sqrt(7.85)  # plan v2: n = 7.85 sigma_d^2 / delta^2 (80% power, 5% two-sided)
    ax.axhline(_pp(dmin), color=INK2, lw=0.8, zorder=1)
    ax.text(2000, _pp(dmin) + 0.08, r"$\delta_{\min}$ = 2 pp, fixed in advance",
            ha="right", va="bottom", fontsize=6.4, color=INK2)
    for key, name, col in [("decoder", "decoder masks", BLUE),
                           ("uniform", "uniform masks", ORANGE)]:
        sd = g1["arms"][key]["delta_contrast"]["sigma_d"]
        ax.plot(ns, [_pp(k * sd / math.sqrt(n)) for n in ns], color=col,
                label=f"{name.split()[0]} ($\\sigma_d$ = {sd:.3f})", zorder=3)
    sd = g1["arms"]["decoder"]["delta_contrast"]["sigma_d"]
    for n, txt, dx, dy, ha in [
        (20, "pilot, 20 docs", 6, 4, "left"),
        (1000, "report half, 1,000 docs", 0, 8, "center"),
    ]:
        v = _pp(k * sd / math.sqrt(n))
        ax.plot([n], [v], marker="o", ms=5, color=BLUE, mec=SURFACE, mew=0.9, zorder=4)
        ax.annotate(f"{txt}\n{v:.2f} pp", xy=(n, v), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="bottom", fontsize=6.4,
                    color=INK, linespacing=1.3)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000])
    ax.set_xticklabels(["10", "20", "50", "100", "200", "500", "1k", "2k"])
    ax.minorticks_off()
    ax.set_xlim(10, 2000)
    ax.set_ylim(0, 3.4)
    ax.set_xlabel("documents in the report half")
    ax.set_ylabel(r"smallest detectable $\Delta(32)-\Delta(1)$ (pp)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, handlelength=1.6,
              borderaxespad=0.1, columnspacing=1.2)
    _save(fig, "power")


def main() -> None:
    _setup_fonts()
    fig_t0_scenarios()
    fig_t0_field_growth()
    fig_estimator()
    fig_contamination()
    fig_t2_gap()
    fig_t2_reliability()
    fig_t2_forest()
    fig_precision()
    fig_power()
    print(f"wrote {len(list(OUT.glob('*.pdf')))} figures to {OUT}")


if __name__ == "__main__":
    main()
