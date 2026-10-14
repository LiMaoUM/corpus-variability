"""Sample size that covers half of the finite corpus at the calibrated radius, measured and predicted
(ARS re-reviews of v3, N2, and of v4, V1 to V5).

For each Qwen3 corpus with a calibrated radius and each judge, at r_same and at both ends of its 90%
interval (results/rsame_boot.csv):
- measured: G(n, r) on a grid that extends to 0.7 N (three draws per n), the corpus missing mass
  M(n) = G(n) (1 - n/N), and the n where M crosses 0.5 (log interpolation). Since M(n) <= 1 - n/N,
  the share n/N is at most one half by construction.
- predicted: the collapse law fitted inside one 2,000-document pilot exactly as in
  scripts/revision_round4.py (same seeds, d the mean TwoNN over five pilots, fit on n <= 1,000),
  G(n, r) = exp(-exp(a + b log(n r^d))), and the n where G (1 - n/N) reaches 0.5.
Writes results/half_cover.csv (one row per corpus, judge and radius end).
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from collapse_v2 import corpora, crossing, surface, fit_cells, N_GRID  # noqa: E402
from corpus_variability.collapse import twonn_dimension  # noqa: E402
from nstar_rsame import RCAL_NAME  # noqa: E402
from revision_round4 import PILOT_GRID  # noqa: E402

PASSING = {"askdocs_qwen3", "tifu_qwen3", "ts_qwen3", "census_random_qwen3", "bsky_qwen3"}
GRID = sorted(set(N_GRID) | {10000, 15000, 20000, 25000, 35000})


def nn_dist(X):
    """Cosine distance from each row of unit-norm X to its nearest other row."""
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), 2048):
        S = X[s:s + 2048] @ X.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -np.inf
        out[s:s + 2048] = 1 - S.max(axis=1)
    return np.clip(out, 0, None)


def pilot_law(X, metric):
    N = X.shape[0]
    rng = np.random.default_rng(11)
    pilots = [X[rng.choice(N, 2000, replace=False)] for _ in range(5)]
    d = float(np.mean([twonn_dimension(p, metric) for p in pilots]))
    d1 = NearestNeighbors(n_neighbors=2, metric=metric).fit(pilots[0]).kneighbors(pilots[0])[0][:, 1]
    r_grid = np.geomspace(np.percentile(d1, 5), np.percentile(d1, 99), 12)
    Gp, _ = surface(pilots[0], PILOT_GRID, r_grid, metric, np.random.default_rng(29))
    a, b = fit_cells(Gp, PILOT_GRID, r_grid, d, 1000)
    return a, b, d


def pilot_law_one(X, metric):
    """As pilot_law, with d from the first pilot alone (ARS re-review of v6, X1)."""
    N = X.shape[0]
    pilot = X[np.random.default_rng(11).choice(N, 2000, replace=False)]
    d = float(twonn_dimension(pilot, metric))
    d1 = NearestNeighbors(n_neighbors=2, metric=metric).fit(pilot).kneighbors(pilot)[0][:, 1]
    r_grid = np.geomspace(np.percentile(d1, 5), np.percentile(d1, 99), 12)
    Gp, _ = surface(pilot, PILOT_GRID, r_grid, metric, np.random.default_rng(29))
    a, b = fit_cells(Gp, PILOT_GRID, r_grid, d, 1000)
    return a, b, d


def half_share(M, n_grid):
    if M[0] <= 0.5:
        return np.nan, n_grid[0]  # below the smallest sample
    return crossing(M, n_grid, 0.5), None


def main():
    rb = pd.read_csv(ROOT / "results/rsame_boot.csv").set_index(["corpus", "judge"])
    rows = []
    for name, X, metric, _ in corpora():
        if name not in RCAL_NAME:
            continue
        N = X.shape[0]
        a, b, d = pilot_law(X, metric)
        rng = np.random.default_rng(13)
        n_grid = [n for n in GRID if n <= 0.7 * N]
        nn = {n: [nn_dist(X[rng.choice(N, n, replace=False)]) for _ in range(3)] for n in n_grid}
        n_fine = np.geomspace(10, N / 2, 4000)
        for judge in ("qwen", "gemma"):
            r_pt, r_lo, r_hi = rb.loc[(RCAL_NAME[name], judge), ["r_same", "r_lo", "r_hi"]]
            if not np.isfinite(r_pt):
                continue
            for end, r in (("point", r_pt), ("r_lo", r_lo), ("r_hi", r_hi)):
                G = np.array([np.mean([(ds > r).mean() for ds in nn[n]]) for n in n_grid])
                M = G * (1 - np.array(n_grid) / N)
                n_half, below = half_share(M, n_grid)
                G_pred = np.exp(-np.exp(a + b * np.log(n_fine * r ** d)))
                hit = np.flatnonzero(G_pred * (1 - n_fine / N) <= 0.5)
                n_pred = float(n_fine[hit[0]]) if len(hit) else N / 2
                G_pred_grid = np.exp(-np.exp(a + b * np.log(np.array(n_grid) * r ** d)))
                rows.append(dict(corpus=name, judge=judge, end=end, passes=name in PASSING, N=N, r=round(r, 3),
                                 d=round(d, 2), a=round(a, 3), b=round(b, 3), G_100=round(G[0], 3),
                                 G_nmax=round(G[-1], 3), n_max=n_grid[-1],
                                 n_half=round(n_half) if np.isfinite(n_half) else np.nan, below_n=below,
                                 share=round(n_half / N, 3) if np.isfinite(n_half) else np.nan,
                                 n_pred=round(n_pred), share_pred=round(n_pred / N, 3),
                                 G_mae=round(float(np.abs(G_pred_grid - G).mean()), 3)))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/half_cover.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
