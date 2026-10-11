"""Direct test of the sample-size formula: fit Equation (collapse) on cells with n <= 1000, predict
n* for isolation rate eps at each radius where the surface reaches eps within the grid, and compare
with the log-interpolated empirical crossing. Reads results/eureka1_surface_*.csv and
results/eureka1_collapse.csv; prints and writes results/nstar_test.csv."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
summ = pd.read_csv(ROOT / "results/eureka1_collapse.csv").set_index("corpus")
rows = []
for name in ["manifold_d2", "manifold_d4", "manifold_d8", "gtr_askdocs", "gtr_tifu", "tweets_encoder768", "gtr_trump", "zipf_k200"]:
    S = pd.read_csv(ROOT / f"results/eureka1_surface_{name}.csv", index_col=0)
    n_grid = S.index.to_numpy(float); r_grid = S.columns.to_numpy(float); G = S.to_numpy()
    d = float(summ.loc[name, "d_twonn"])
    cells = [(np.log(n) + d * np.log(r), np.log(-np.log(G[i, j]))) for i, n in enumerate(n_grid) if n <= 1000
             for j, r in enumerate(r_grid) if 0.02 < G[i, j] < 0.98]
    A = np.array(cells); X = np.column_stack([np.ones(len(A)), A[:, 0]])
    (a, b), *_ = np.linalg.lstsq(X, A[:, 1], rcond=None)
    for eps in (0.3, 0.2, 0.1):
        for j, r in enumerate(r_grid):
            col = G[:, j]
            # empirical crossing must occur beyond the fitting range (n > 1000) and within the grid
            idx = [i for i in range(1, len(n_grid)) if col[i - 1] > eps >= col[i]]
            if not idx or n_grid[idx[0]] <= 1000:
                continue
            i = idx[0]
            w = (np.log(col[i - 1]) - np.log(eps)) / (np.log(col[i - 1]) - np.log(max(col[i], 1e-6)))
            n_emp = np.exp(np.log(n_grid[i - 1]) + w * (np.log(n_grid[i]) - np.log(n_grid[i - 1])))
            n_pred = (np.log(1 / eps) / np.exp(a)) ** (1 / b) / r ** d
            rows.append(dict(corpus=name, eps=eps, r=round(r, 4), n_pred=round(n_pred), n_emp=round(n_emp), ratio=round(n_pred / n_emp, 2)))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/nstar_test.csv", index=False)
print(df.to_string(index=False))
print("\nper corpus: cells, median ratio n_pred/n_emp, share within factor 1.5, within 2:")
for c, g in df.groupby("corpus"):
    print(f"  {c:18s} k={len(g):2d} median={g.ratio.median():.2f} in1.5={((g.ratio>1/1.5)&(g.ratio<1.5)).mean():.2f} in2={((g.ratio>0.5)&(g.ratio<2)).mean():.2f}")
