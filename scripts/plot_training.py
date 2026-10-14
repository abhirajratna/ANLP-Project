#!/usr/bin/env python3
"""The 110M validation curve, with gate G2's tail window (plan v2 T3).

    python scripts/plot_training.py --history artifacts/history_train110m-v3.jsonl \\
        --gate artifacts/gate_g2_train110m-v3.json --out artifacts/figures/train110m-v3_curve

Left: validation NELBO (EMA and raw weights) against training tokens.  Right: the
last 20% of steps, the G2 tail window shaded and the line check_g2.py fitted to it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
EMA, RAW = "#2a78d6", "#eb6834"          # reference palette slots 1-2 (validated pair)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.titlesize": 8})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out", required=True, help="path stem; .pdf and .png are written")
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.history).read_text().splitlines() if l.strip()]
    gate = json.loads(Path(a.gate).read_text())
    by_step = {}
    for r in rows:
        if "val_nelbo" in r:
            by_step[r["step"]] = r              # a requeue re-logs steps: keep the last
    pts = [by_step[s] for s in sorted(by_step)]
    step = np.array([p["step"] for p in pts], float)
    tok = np.array([p["tokens"] for p in pts], float) / 1e9
    ema = np.array([p.get("ema_val_nelbo", np.nan) for p in pts])
    raw = np.array([p["val_nelbo"] for p in pts])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, linewidth=0.75)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    for ax, sel in ((ax1, slice(None)), (ax2, step >= step.max() - 0.2 * (step.max() - step.min()))):
        ax.plot(tok[sel], raw[sel], color=RAW, linewidth=1.5, label="raw weights")
        ax.plot(tok[sel], ema[sel], color=EMA, linewidth=1.5, label="EMA weights (G2 metric)")
        ax.set_xlabel("training tokens (billions)")
    ax1.set_ylabel("validation NELBO, nats/token")
    ax1.set_ylim(min(np.nanmin(ema), raw.min()) - 0.2, 8.0)
    ax1.set_title("whole run")
    cut = step.max() - gate["tail_fraction"] * (step.max() - step.min())
    tail = step >= cut
    ax2.axvspan(tok[tail].min(), tok[tail].max(), color=GRID, alpha=0.6, linewidth=0)
    m, c = np.polyfit(step[tail], ema[tail], 1)
    ax2.plot(tok[tail], m * step[tail] + c, color=INK2, linewidth=1, linestyle=(0, (3, 2)))
    ax2.set_title(f"last 20%: G2 {'PASS' if gate['pass'] else 'FAIL'}, tail change "
                  f"{100 * gate['relative_change']:.2f}% (threshold {100 * gate['threshold']:.1f}%)")
    ax2.text(0.02, 0.05, "shaded: G2 window (last 10% of steps);\ndashed: the line fitted to it",
             transform=ax2.transAxes, fontsize=6.5, color=INK2)
    ax1.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    print(f"[plot] {len(pts)} validation points -> {out}.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
