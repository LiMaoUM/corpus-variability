"""Second-round revision experiments (panel re-review, 2026-10-13).

For every corpus and encoder pair of collapse_v2:
  1. Per radius baseline in n alone: at each radius, fit log(-log G) = a_r + b_r log n on cells
     with n <= 1,000 and predict G at the largest n. Compare MAE with the collapse law.
  2. Pilot-only diagnostic: from one pilot of 2,000 documents, draw nested subsamples
     n in {100, 200, 350, 500, 750, 1000, 1500, 2000} (five draws), fit the collapse on n <= 750,
     score R2 on n in {1000, 1500, 2000}. Compare with the full diagnostic.
  3. TwoNN-free dimension: ratio of the log r and log n coefficients of the two-regressor fit.
  4. n* at the calibrated radius (r where half of pairs are the same content, Table 4) for
     eps = 0.5 and 0.2, from the collapse fit on n <= 1,000, and the empirical crossing if the grid
     reaches it.
Writes results/revision_round2.csv.
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
from collapse_v2 import corpora, surface, fit_cells, r2_on, crossing, N_GRID  # noqa: E402
from corpus_variability.collapse import twonn_dimension  # noqa: E402

R_SAME = {"askdocs_qwen3": 0.146, "tifu_qwen3": 0.050, "trump_qwen3": 0.240, "census_random_qwen3": 0.284,
          "census_citizens_qwen3": 0.225, "census_trump_qwen3": 0.319, "census_covid_qwen3": 0.286,
          "census_illegal_qwen3": 0.300, "ts_qwen3": 0.162}
PILOT_GRID = [100, 200, 350, 500, 750, 1000, 1500, 2000]


def two_regressor_ratio(G, n_grid, r_grid, n_max):
    rows = [(np.log(n), np.log(r), np.log(-np.log(G[i, j, k])))
            for i, n in enumerate(n_grid) if n <= n_max for j, r in enumerate(r_grid)
            for k in range(G.shape[2]) if 0.02 < G[i, j, k] < 0.98]
    A = np.array(rows)
    X = np.column_stack([np.ones(len(A)), A[:, 0], A[:, 1]])
    coef, *_ = np.linalg.lstsq(X, A[:, 2], rcond=None)
    return coef[2] / coef[1]


def per_radius_baseline(G, n_grid, r_grid, n_fit):
    Gm = G.mean(axis=2)
    i_last = len(n_grid) - 1
    errs = []
    for j in range(len(r_grid)):
        pts = [(np.log(n), np.log(-np.log(G[i, j, k]))) for i, n in enumerate(n_grid) if n <= n_fit
               for k in range(G.shape[2]) if 0.02 < G[i, j, k] < 0.98]
        if len(pts) < 4:
            continue
        A = np.array(pts)
        (a, b), *_ = np.linalg.lstsq(np.column_stack([np.ones(len(A)), A[:, 0]]), A[:, 1], rcond=None)
        pred = np.exp(-np.exp(a + b * np.log(n_grid[i_last])))
        errs.append(abs(pred - Gm[i_last, j]))
    return float(np.mean(errs)) if errs else np.nan, len(errs)


def main():
    rows = []
    for name, X, metric, enc in corpora():
        N = X.shape[0]
        rng = np.random.default_rng(11)
        n_grid = [n for n in N_GRID if n <= 0.7 * N]
        pilots = [X[rng.choice(N, 2000, replace=False)] for _ in range(5)]
        d = float(np.mean([twonn_dimension(p, metric) for p in pilots]))
        d1 = NearestNeighbors(n_neighbors=2, metric=metric).fit(pilots[0]).kneighbors(pilots[0])[0][:, 1]
        r_grid = np.geomspace(np.percentile(d1, 5), np.percentile(d1, 99), 12)
        G, _ = surface(X, n_grid, r_grid, metric, rng)
        Gm = G.mean(axis=2)
        a, b = fit_cells(G, n_grid, r_grid, d, 1000)
        z = np.log(n_grid[-1]) + d * np.log(r_grid)
        mae_collapse = float(np.abs(np.exp(-np.exp(a + b * z)) - Gm[-1]).mean())
        mae_perr, k_perr = per_radius_baseline(G, n_grid, r_grid, 1000)
        ratio_d = two_regressor_ratio(G, n_grid, r_grid, 1000)
        # pilot-only diagnostic on pilot 0
        P = pilots[0]
        Gp, _ = surface(P, PILOT_GRID, r_grid, metric, rng)
        ap, bp = fit_cells(Gp, PILOT_GRID, r_grid, d, 750)
        r2_pilot = r2_on(Gp, PILOT_GRID, r_grid, d, ap, bp, 751)
        row = dict(corpus=name, encoder=enc, d=round(d, 2), d_ratio=round(ratio_d, 2),
                   mae_collapse=round(mae_collapse, 4), mae_per_radius=round(mae_perr, 4), radii_per_radius=k_perr,
                   r2_out_pilot=round(r2_pilot, 3), r2_out_full=round(r2_on(G, n_grid, r_grid, d, a, b, 1001), 3))
        # n* at the calibrated radius
        rs = R_SAME.get(name)
        if rs:
            G_rs = []
            for n in n_grid:
                gs = []
                for _ in range(3):
                    idx = rng.choice(N, n, replace=False)
                    ds = NearestNeighbors(n_neighbors=2, metric=metric).fit(X[idx]).kneighbors(X[idx])[0][:, 1]
                    gs.append(float((ds > rs).mean()))
                G_rs.append(np.mean(gs))
            row.update(r_same=rs, G_at_rsame_nmax=round(G_rs[-1], 3), n_max=n_grid[-1])
            for eps in (0.5, 0.2):
                n_pred = (np.log(1 / eps) / np.exp(a)) ** (1 / b) / rs ** d
                n_emp = crossing(np.array(G_rs), n_grid, eps)
                row[f"nstar_eps{eps}"] = float(f"{n_pred:.3g}")
                row[f"nemp_eps{eps}"] = round(n_emp) if np.isfinite(n_emp) else np.nan
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/revision_round2.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
