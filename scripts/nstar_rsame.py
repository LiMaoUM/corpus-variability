"""n* at the calibrated radius under each judge (cross-judge follow-up, 2026-10-13).

The radius r_same is where half of the judged pairs are the same content (results/rcal_crossjudge.csv,
one value per judge). The prediction rescales the round-two fit exactly, since a, b and d do not
depend on r: n*(r') = n*(r) (r'/r)^(-d). The empirical crossing reuses the round-two protocol
(three draws per grid size, G at the radius, log interpolation).
Writes results/nstar_rsame.csv.
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

RCAL_NAME = {"askdocs_qwen3": "sv_askdocs", "tifu_qwen3": "sv_tifu", "trump_qwen3": "sv_trump", "ts_qwen3": "ts",
             "census_random_qwen3": "random", "census_citizens_qwen3": "citizens", "census_trump_qwen3": "trump",
             "census_covid_qwen3": "covid", "census_illegal_qwen3": "illegal"}


def main():
    r2 = pd.read_csv(ROOT / "results/revision_round2.csv").set_index("corpus")
    cj = pd.read_csv(ROOT / "results/rcal_crossjudge.csv").set_index("corpus")
    rows = []
    for name, X, metric, _ in corpora():
        if name not in RCAL_NAME:
            continue
        N = X.shape[0]
        rng = np.random.default_rng(11)
        n_grid = [n for n in N_GRID if n <= 0.7 * N]
        base = r2.loc[name]
        nn = []
        for n in n_grid:
            for _ in range(3):
                idx = rng.choice(N, n, replace=False)
                nn.append((n, NearestNeighbors(n_neighbors=2, metric=metric).fit(X[idx]).kneighbors(X[idx])[0][:, 1]))
        for judge in ("qwen", "gemma"):
            rs = cj.loc[RCAL_NAME[name], f"r_same_{judge}"]
            row = dict(corpus=name, judge=judge, N=N, d=base["d"], r_same=rs)
            for eps in (0.5, 0.2):
                if not np.isfinite(rs):
                    continue
                pred = base[f"nstar_eps{eps}"] * (rs / base["r_same"]) ** (-base["d"])
                G = np.array([np.mean([(ds > rs).mean() for m, ds in nn if m == n]) for n in n_grid])
                emp = crossing(G, n_grid, eps)
                row[f"nstar_eps{eps}"] = float(f"{pred:.3g}")
                row[f"nemp_eps{eps}"] = round(emp) if np.isfinite(emp) else np.nan
                row["G_at_nmax"] = round(float(G[-1]), 3)
            rows.append(row)
            print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/nstar_rsame.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
