#!/usr/bin/env python3
"""The T9 crossover figures from scripts/run_probe.py results.

    python scripts/plot_probe.py artifacts/probe/results/grid_untuned-v2.json \\
        artifacts/probe/results/grid_ft110m-v3.json --out-dir artifacts/probe/figures

Per result: ``crossover_<tag>.{pdf,png}``, small multiples (rows: selection rule,
columns: rho) of (A) dependency and (B) estimation per field against k, with
their 95% document-bootstrap bands -- the crossover is where the lines cross.
Across results: ``crossover_map.{pdf,png}``, the share of (A) in (A)+(B) for
every rho x k cell, one panel per result and rule, on a diverging scale whose
gray midpoint (50%) is the crossover.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Reference palette (dataviz skill), light mode: slot 1 for (A), slot 2 for (B);
# the pair passes the validator (adjacent CVD dE 24.7, normal-vision dE 33.6).
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
COLOR = {"A": "#2a78d6", "B": "#eb6834"}
NAME = {"A": "(A) dependency: TC", "B": "(B) estimation: Σ KL"}
RULE_NAME = {"random": "random positions", "confidence": "confidence top-k"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.titlesize": 8, "axes.titlecolor": INK,
})


def cells(res: dict) -> dict:
    return {(c["rule"], c["k"], c["rho"]): c for c in res["cells"]}


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def lines(res: dict, out: Path) -> None:
    c = cells(res)
    rules = [r for r in ("random", "confidence") if r in res["rules"]]
    rhos = sorted({k[2] for k in c})
    ks = sorted({k[1] for k in c})
    fig, axes = plt.subplots(len(rules), len(rhos), figsize=(1.55 * len(rhos) + 0.6, 1.6 * len(rules) + 0.9),
                             sharex=True, sharey=True, squeeze=False)
    for r, rule in enumerate(rules):
        for j, rho in enumerate(rhos):
            ax = axes[r][j]
            style(ax)
            for term in ("A", "B"):
                key = f"{term}_per_field"
                m = np.array([c[(rule, k, rho)][key]["mean"] for k in ks])
                lo = np.array([c[(rule, k, rho)][key]["ci95"][0] for k in ks])
                hi = np.array([c[(rule, k, rho)][key]["ci95"][1] for k in ks])
                ax.fill_between(ks, lo, hi, color=COLOR[term], alpha=0.12, linewidth=0)
                ax.plot(ks, m, color=COLOR[term], linewidth=1.5, solid_capstyle="round",
                        marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
            first = res["crossover"].get(f"{rule}/rho={rho}", {}).get("first_k_A_exceeds_B")
            ax.text(0.04, 0.95, f"A > B from k = {first}" if first else "B dominates",
                    transform=ax.transAxes, va="top", fontsize=6.5, color=INK2)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ks, [str(k) for k in ks])
            ax.minorticks_off()
            if r == 0:
                ax.set_title(f"ρ = {rho:g}")
            if j == 0:
                ax.set_ylabel(f"{RULE_NAME[rule]}\nnats per field")
            if r == len(rules) - 1:
                ax.set_xlabel("tokens per step k")
    handles = [Line2D([], [], color=COLOR[t], linewidth=1.5, marker="o", markersize=4.5,
                      markeredgecolor=SURFACE, label=NAME[t]) for t in ("A", "B")]
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.suptitle(f"T9 probe, {res['tag']}: per-field KL of a k-token parallel commit "
                 f"(mean over {res['n_instances'] // max(1, len(rhos))} instances per ρ, 95% CI)",
                 y=0.985, fontsize=8.5, color=INK)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.945), fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"crossover_{res['tag']}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def share_map(results: list[dict], out: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("b_to_a", [COLOR["B"], NEUTRAL, COLOR["A"]])
    panels = [(res, rule) for res in results for rule in ("random", "confidence")
              if rule in res["rules"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels) + 0.8, 2.3), squeeze=False,
                             sharey=True)
    im = None
    for ax, (res, rule) in zip(axes[0], panels):
        c = cells(res)
        rhos = sorted({k[2] for k in c})
        ks = sorted({k[1] for k in c})
        grid = np.array([[c[(rule, k, rho)]["share_A"] for k in ks] for rho in rhos])
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
        for y in range(len(rhos)):
            for x in range(len(ks)):
                v = max(0.0, grid[y, x])         # (A) >= 0; rounding can leave -1e-17
                ax.text(x, y, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5,
                        color=SURFACE if abs(v - 0.5) > 0.35 else INK)
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_yticks(range(len(rhos)), [f"{r:g}" for r in rhos])
        ax.set_xlabel("tokens per step k")
        ax.set_title(f"{res['tag']}\n{RULE_NAME[rule]}")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0][0].set_ylabel("dependency ρ")
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cb.set_label("(A) share of (A)+(B), %  (50 = crossover)", color=INK2)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "25", "50", "75", "100"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"crossover_map.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="grid_<tag>.json files")
    ap.add_argument("--out-dir", default="artifacts/probe/figures")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = [json.loads(Path(p).read_text()) for p in a.results]
    for res in results:
        lines(res, out)
    share_map(results, out)
    print(f"[plot] {len(results)} result(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
