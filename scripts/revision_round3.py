"""Third set of revision checks (SAC re-review items 3 and 5, 2026-10-13).

For every corpus and encoder pair:
  A. Joint bootstrap interval for n*: resample B = 50 pilots of 2,000 documents; in each,
     re-estimate d by TwoNN and fit a, b on a fresh (n <= 1,000, five draws) grid; take the 5th and
     95th percentiles of n* per (eps, r) case; report coverage of the empirical crossing (from the
     full-corpus surface of collapse_v2's protocol) and median relative width.
  B. Chao and Jost within its range: predict G at n = 2,000 from n = 1,000 (a doubling) at each
     radius, against the collapse fit on n <= 1,000; MAE against the observed G at 2,000.
Writes results/revision_round3.csv and results/nstar_joint_boot.csv.
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
from collapse_v2 import corpora, surface, fit_cells, crossing, N_GRID, EPS  # noqa: E402
from corpus_variability.collapse import twonn_dimension  # noqa: E402
from corpus_variability.coverage import chao_extrapolated_coverage  # noqa: E402

B = 50
FIT_GRID = [100, 200, 350, 500, 750, 1000]


def main():
    rows, cases = [], []
    for name, X, metric, enc in corpora():
        N = X.shape[0]
        rng = np.random.default_rng(23)
        n_grid = [n for n in N_GRID if n <= 0.7 * N]
        pilot0 = X[rng.choice(N, 2000, replace=False)]
        d0 = twonn_dimension(pilot0, metric)
        d1 = NearestNeighbors(n_neighbors=2, metric=metric).fit(pilot0).kneighbors(pilot0)[0][:, 1]
        r_grid = np.geomspace(np.percentile(d1, 5), np.percentile(d1, 99), 12)
        G, F2 = surface(X, n_grid, r_grid, metric, rng)
        Gm = G.mean(axis=2)
        a0, b0 = fit_cells(G, n_grid, r_grid, d0, 1000)
        # B. Chao and Jost at a doubling
        i1, i2 = n_grid.index(1000), n_grid.index(2000)
        cj, co = [], []
        for j, r in enumerate(r_grid):
            f1 = int(round(Gm[i1, j] * 1000)); f2 = int(round(F2[i1, j].mean() * 1000))
            cj.append(abs((1 - chao_extrapolated_coverage(1000, f1, f2, 1000)) - Gm[i2, j]))
            co.append(abs(np.exp(-np.exp(a0 + b0 * (np.log(2000) + d0 * np.log(r)))) - Gm[i2, j]))
        # A. joint bootstrap over pilots
        boots = []
        for _ in range(B):
            P = X[rng.choice(N, 2000, replace=False)]
            db = twonn_dimension(P, metric)
            Gb, _ = surface(X, FIT_GRID, r_grid, metric, rng)
            try:
                ab, bb = fit_cells(Gb, FIT_GRID, r_grid, db, 1000)
                boots.append((db, ab, bb))
            except Exception:
                pass
        for eps in EPS:
            for j, r in enumerate(r_grid):
                n_emp = crossing(Gm[:, j], n_grid, eps)
                if not np.isfinite(n_emp) or n_emp <= 1000:
                    continue
                n_pred = (np.log(1 / eps) / np.exp(a0)) ** (1 / b0) / r ** d0
                bp = np.array([(np.log(1 / eps) / np.exp(aa)) ** (1 / bb) / r ** dd for dd, aa, bb in boots])
                lo, hi = np.nanpercentile(bp, [5, 95])
                cases.append(dict(corpus=name, eps=eps, r=round(float(r), 4), n_pred=n_pred, lo=lo, hi=hi, n_emp=n_emp,
                                  covered=bool(lo <= n_emp <= hi), rel_width=(hi - lo) / n_pred))
        cs = pd.DataFrame([c for c in cases if c["corpus"] == name])
        rows.append(dict(corpus=name, encoder=enc, mae_cj_2x=round(float(np.mean(cj)), 4), mae_collapse_2x=round(float(np.mean(co)), 4),
                         boot_n=len(boots), cases=len(cs), coverage=round(cs.covered.mean(), 2) if len(cs) else np.nan,
                         median_rel_width=round(cs.rel_width.median(), 2) if len(cs) else np.nan))
        print(rows[-1], flush=True)
    pd.DataFrame(cases).to_csv(ROOT / "results/nstar_joint_boot.csv", index=False)
    df = pd.DataFrame(rows); df.to_csv(ROOT / "results/revision_round3.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
