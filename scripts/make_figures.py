"""Paper figures (specs confirmed by Mao, 2026-10-11).

collapse: 2x3 panels, G against n r^d (log x), colour = radius r (viridis), grey fitted stretched
exponential, annotation d and out-of-sample R2.
themes: two panels, theme mass coverage against n with the Good-Turing expectation per corpus
(left) and coverage at the shared radius r = 0.34 (right), Okabe-Ito colours.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper/figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
})
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def collapse():
    """Points: one draw per cell (results/eureka1_surface_*.csv). Line: the v2 fit on cells with
    n <= 1,000 (results/collapse_v2.csv, a and b). Annotation: d and full out of sample R2 of the
    paper's Table 1 (v2, GTR rows averaged over three samples)."""
    v2 = pd.read_csv(RES / "collapse_v2.csv").set_index("corpus")
    panels = [("manifold_d2", "manifold_d2", ["manifold_d2"], "Manifold d = 2"),
              ("manifold_d8", "manifold_d8", ["manifold_d8"], "Manifold d = 8"),
              ("gtr_askdocs", "askdocs_gtr_s42", ["askdocs_gtr_s42", "askdocs_gtr_s43", "askdocs_gtr_s44"], "AskDocs"),
              ("gtr_tifu", "tifu_gtr_s42", ["tifu_gtr_s42", "tifu_gtr_s43", "tifu_gtr_s44"], "TIFU"),
              ("gtr_trump", "trump_gtr_s42", ["trump_gtr_s42", "trump_gtr_s43", "trump_gtr_s44"], "Trump"),
              ("zipf_k200", "zipf_k200", ["zipf_k200"], "Zipf, 200 topics")]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharey=True)
    fig.subplots_adjust(hspace=0.55, wspace=0.12)
    for ax, (surf, fitrow, annrows, label) in zip(axes.ravel(), panels):
        S = pd.read_csv(RES / f"eureka1_surface_{surf}.csv", index_col=0)
        n_grid = S.index.to_numpy(float); r_grid = S.columns.to_numpy(float); G = S.to_numpy()
        d = float(v2.loc[annrows, "d_twonn"].mean()); r2 = float(v2.loc[annrows, "fit1000_r2_out"].mean())
        a, b = float(v2.loc[fitrow, "fit1000_a"]), float(v2.loc[fitrow, "fit1000_b"])
        dfit = float(v2.loc[fitrow, "d_twonn"])
        X = np.outer(n_grid, r_grid ** dfit)
        norm = mpl.colors.LogNorm(r_grid.min(), r_grid.max())
        for j, r in enumerate(r_grid):
            ax.scatter(X[:, j], G[:, j], s=7, c=[plt.cm.viridis(norm(r))] * len(n_grid), linewidths=0)
        xs = np.geomspace(X.min(), X.max(), 200)
        ax.plot(xs, np.exp(-np.exp(a + b * np.log(xs))), color="0.45", lw=0.9, zorder=0)
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_title(label, fontsize=7.5, loc="left")
        corner = (0.04, 0.05, "left", "bottom") if label == "Trump" else (0.97, 0.97, "right", "top")
        ax.text(corner[0], corner[1], f"d = {d:.1f}, $R^2$ = {r2:.2f}", transform=ax.transAxes, ha=corner[2], va=corner[3], fontsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel("$n\\,r^{d}$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Isolation rate $G$")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(0, 1)); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Radius $r$ (small to large)"); cb.set_ticks([])
    fig.savefig(FIG / "collapse.pdf", bbox_inches="tight")
    fig.savefig(FIG / "collapse.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def themes():
    df = pd.read_csv(RES / "theme_recall.csv")
    corpora = ["random", "citizens", "trump", "covid", "illegal", "sv_askdocs", "sv_trump", "sv_tifu"]
    labels = {"random": "Census random", "citizens": "Census citizens", "trump": "Census trump", "covid": "Census covid",
              "illegal": "Census illegal", "sv_askdocs": "AskDocs", "sv_trump": "Trump", "sv_tifu": "TIFU"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ns = np.geomspace(20, 450, 100)
    for c, col in zip(corpora, OKABE):
        g = df[df.corpus == c].sort_values("n")
        m = np.array([t["mass"] for t in json.load((ROOT / f"data/themes/{c}_themes.json").open())["themes"]])
        exp = [(m * (1 - (1 - m) ** n)).sum() for n in ns]
        ax1.plot(ns, exp, color=col, lw=0.7, alpha=0.9)
        ax1.scatter(g.n, g.label_cov, color=col, s=12, zorder=3, label=labels[c], linewidths=0)
        ax2.plot(g.n, g["cov_q0.5"], color=col, lw=0.8, marker="o", ms=3, label=labels[c])
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks([25, 50, 100, 200, 400])
        ax.set_xticklabels(["25", "50", "100", "200", "400"])
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlabel("Sample size $n$")
    ax1.set_ylim(0.5, 1.02)
    ax1.set_ylabel("Theme mass covered")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Coverage at $r$ = 0.34")
    ax1.legend(loc="lower right", fontsize=5.5, ncol=2, handletextpad=0.3, columnspacing=0.8)
    ax1.set_title("a", loc="left", fontweight="bold")
    ax2.set_title("b", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "themes.pdf", bbox_inches="tight")
    fig.savefig(FIG / "themes.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    collapse()
    themes()
    print("written", sorted(p.name for p in FIG.iterdir()))
