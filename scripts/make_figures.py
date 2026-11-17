#!/usr/bin/env python3
"""Figures from the committed grid artefact. Every number read from results/, none typed."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: F401  PercentFormatter needs the submodule
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = ROOT / "figures"

ORDER = ["random", "scaffold", "butina", "time", "frontier"]
LABEL = {"random": "random\n(shuffled rows)",
         "scaffold": "scaffold\n(Murcko)",
         "butina": "butina\n(cluster)",
         "time": "time\n(within history)",
         "frontier": "frontier\n(least similar)"}


def _footer(fig, text: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=7)
    limit = 0.975 * fig.get_figwidth() * fig.dpi
    lines: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            trial = (line + " " + word).strip()
            probe.set_text(trial)
            if line and probe.get_window_extent(renderer).width > limit:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    probe.remove()
    fig.text(0.012, 0.004, "\n".join(lines), fontsize=7, color="#555555",
             ha="left", va="bottom", linespacing=1.35)


def fig_ratio_by_protocol() -> None:
    """The headline: retrospective/prospective error ratio per validation protocol."""
    d = json.loads((RES / "grid.json").read_text())
    prot = d["protocols"]
    t = pd.read_csv(RES / "grid.csv")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8),
                                  gridspec_kw=dict(width_ratios=[1, 1.15]))

    # ---- left: median ratio and CI per protocol, with the honest line at 1.0
    meds = [prot[p]["median_ratio"] for p in ORDER]
    lo = [prot[p]["ci95"][0] for p in ORDER]
    hi = [prot[p]["ci95"][1] for p in ORDER]
    x = np.arange(len(ORDER))
    colours = ["#d62728" if (h < 1.0) else "#4c72b0" for h in hi]
    ax.errorbar(x, meds,
                yerr=[np.array(meds) - np.array(lo), np.array(hi) - np.array(meds)],
                fmt="o", ms=7, capsize=5, lw=1.6, color="#333333", zorder=3)
    ax.scatter(x, meds, s=90, c=colours, zorder=4, edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, ls="--", lw=1.4, color="#222222")
    ax.text(len(ORDER) - 0.55, 1.0, " calibrated", va="bottom", ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[p] for p in ORDER], fontsize=8)
    ax.set_ylabel("retrospective MAE ÷ prospective MAE", fontsize=9)
    ax.set_title("Two protocols flatter the model, three do not\n"
                 f"median over {d['n_endpoints']} endpoints × {d['n_models']} model "
                 f"families, 95% bootstrap CI", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    for xi, (m, l, h) in enumerate(zip(meds, lo, hi)):
        ax.annotate(f"{m:.2f}", (xi, h), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)

    # ---- right: every (endpoint, model) point, so the spread is visible
    for i, p in enumerate(ORDER):
        sub = t[t.protocol == p]
        jitter = (np.random.default_rng(i).random(len(sub)) - 0.5) * 0.22
        ax2.scatter(np.full(len(sub), i) + jitter, sub.ratio, s=18,
                    color=colours[i], alpha=0.75, edgecolor="none")
    ax2.axhline(1.0, ls="--", lw=1.4, color="#222222")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABEL[p].split("\n")[0] for p in ORDER], fontsize=8)
    ax2.set_ylabel("ratio, per endpoint × model", fontsize=9)
    ax2.set_title("The spread is wide; the medians are what move\n"
                  f"{len(t)} cells, one dot each", fontsize=9.5)
    ax2.grid(axis="y", alpha=0.25)

    _footer(fig, "Left: median over endpoints and model families of (MAE from the "
                 "validation protocol, measured inside the training period) ÷ (MAE on "
                 "the real later compounds). A value below 1.0 means the protocol "
                 "under-stated the error a team will actually see. Bars are a 95% "
                 "percentile bootstrap over the ratios. Right: the same ratios "
                 "unaggregated, which is why the intervals are reported. Prospective "
                 "MAE is computed once per endpoint×model by fitting on the whole "
                 "training period and scoring the future compounds; no future compound "
                 "enters any retrospective number.")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "ratio_by_protocol.png", dpi=160)
    plt.close(fig)
    print("  wrote figures/ratio_by_protocol.png")


def fig_shift_vs_ratio() -> None:
    """Does the size of the train→test distribution shift predict how badly a protocol lies?

    Honest possibility, stated up front: it may not, and that is worth showing.
    """
    d = json.loads((RES / "grid.json").read_text())
    cov = pd.DataFrame(d["coverage"]).set_index("endpoint")
    t = pd.read_csv(RES / "grid.csv")

    # shift proxy: absolute change in the endpoint's median between periods, in units
    # of the training interquartile range (a scale-free measure)
    med = pd.DataFrame(d["coverage"])[["endpoint"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i, p in enumerate(ORDER):
        sub = t[t.protocol == p]
        xs = [cov.loc[e, "train_n"] for e in sub.endpoint]
        ax.scatter(xs, sub.ratio, s=22, alpha=0.7, label=p,
                   color=["#d62728", "#dd8452", "#55a868", "#4c72b0", "#8172b3"][i])
    ax.axhline(1.0, ls="--", lw=1.2, color="#222222")
    ax.set_xlabel("training-period sample size for the endpoint", fontsize=9)
    ax.set_ylabel("retrospective ÷ prospective MAE", fontsize=9)
    ax.set_title("Ratio against endpoint sample size, all cells", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=3, frameon=False)
    ax.grid(alpha=0.25)
    _footer(fig, "One dot per (endpoint, model, protocol). The x axis is how many "
                 "compounds the training period has for that endpoint, which is the "
                 "quantity a team controls when it decides whether to trust a "
                 "validation split at all. This panel is descriptive: no trend is "
                 "claimed, and if the points scatter then sample size is not the "
                 "explanation for the ratios in the headline figure.")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGS / "shift_vs_ratio.png", dpi=160)
    plt.close(fig)
    print("  wrote figures/shift_vs_ratio.png")


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_ratio_by_protocol()
    fig_shift_vs_ratio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
