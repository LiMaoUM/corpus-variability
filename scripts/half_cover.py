"""Sample size that covers half of the finite corpus at the calibrated radius (ARS re-review of v3, N2).

For each Qwen3 corpus with a calibrated radius and each judge, estimate G(n, r_same) on the
collapse_v2 grid (three draws per n), form the corpus missing mass M(n) = G(n) (1 - n/N), and find
the n where M crosses 0.5 (log interpolation). If the grid ends first, report the bracket
[N (1 - 0.5 / G(n_max)), N / 2]: G only falls with n, and G <= 1 bounds n by N / 2.
Writes results/half_cover.csv.
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
from collapse_v2 import corpora, crossing, N_GRID  # noqa: E402
from nstar_rsame import RCAL_NAME  # noqa: E402


def main():
    rb = pd.read_csv(ROOT / "results/rsame_boot.csv").set_index(["corpus", "judge"])
    rows = []
    for name, X, metric, _ in corpora():
        if name not in RCAL_NAME:
            continue
        N = X.shape[0]
        rng = np.random.default_rng(13)
        n_grid = [n for n in N_GRID if n <= 0.7 * N]
        nn = {n: [NearestNeighbors(n_neighbors=2, metric=metric).fit(X[i]).kneighbors(X[i])[0][:, 1]
                  for i in (rng.choice(N, n, replace=False) for _ in range(3))] for n in n_grid}
        for judge in ("qwen", "gemma"):
            rs = rb.loc[(RCAL_NAME[name], judge), "r_same"]
            if not np.isfinite(rs):
                continue
            G = np.array([np.mean([(ds > rs).mean() for ds in nn[n]]) for n in n_grid])
            M = G * (1 - np.array(n_grid) / N)
            n_half = crossing(M, n_grid, 0.5)
            if M[0] <= 0.5:
                lo, hi = 0, n_grid[0]
            elif np.isfinite(n_half):
                lo = hi = n_half
            else:
                lo, hi = N * (1 - 0.5 / G[-1]), N / 2
            rows.append(dict(corpus=name, judge=judge, N=N, r_same=rs, G_100=round(G[0], 3), G_nmax=round(G[-1], 3),
                             n_max=n_grid[-1], n_half_lo=round(lo), n_half_hi=round(hi),
                             frac_lo=round(lo / N, 3), frac_hi=round(hi / N, 3)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/half_cover.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
