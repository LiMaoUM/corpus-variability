"""Eureka 1, prediction test: fit the one-variable collapse log(-log G) = a + b log(n r^d) with
d = TwoNN on samples of at most N_FIT documents, then predict the isolation rate G at the
largest n in the surface for every r, and compare with the observed G.

Reads results/eureka1_surface_<name>.csv and results/eureka1_collapse.csv.
Writes results/eureka1_predict.csv and prints per-corpus MAE and max error of predicted G.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_FIT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


def main():
    summ = pd.read_csv(ROOT / "results/eureka1_collapse.csv").set_index("corpus")
    rows = []
    for f in sorted((ROOT / "results").glob("eureka1_surface_*.csv")):
        name = f.stem.replace("eureka1_surface_", "")
        S = pd.read_csv(f, index_col=0)
        n_grid = S.index.to_numpy(dtype=float)
        r_grid = S.columns.to_numpy(dtype=float)
        G = S.to_numpy()
        d = summ.loc[name, "d_twonn"]
        A = [(np.log(n) + d * np.log(r), np.log(-np.log(G[i, j])))
             for i, n in enumerate(n_grid) if n <= N_FIT
             for j, r in enumerate(r_grid) if 0.02 < G[i, j] < 0.98]
        A = np.array(A)
        Xd = np.column_stack([np.ones(A.shape[0]), A[:, 0]])
        (a, b), *_ = np.linalg.lstsq(Xd, A[:, 1], rcond=None)
        i = len(n_grid) - 1
        n_max = n_grid[i]
        z = np.log(n_max) + d * np.log(r_grid)
        G_pred = np.exp(-np.exp(a + b * z))
        G_obs = G[i]
        err = G_pred - G_obs
        rows.append(dict(corpus=name, d_twonn=d, n_fit_max=N_FIT, n_target=int(n_max),
                         mae=round(float(np.abs(err).mean()), 4), max_abs=round(float(np.abs(err).max()), 4),
                         bias=round(float(err.mean()), 4),
                         obs_range=f"{G_obs.min():.3f}..{G_obs.max():.3f}"))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/eureka1_predict.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
