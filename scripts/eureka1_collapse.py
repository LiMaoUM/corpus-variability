"""Eureka 1 test: does the isolation surface G(n, r) collapse on n r^d, with d the intrinsic
dimension, and does the fitted law predict the required n at scales it was not fitted on?

Corpora:
  manifold_d{2,4,8}  : Gaussian on a d-dimensional subspace of R^64 (true d known), euclidean
  zipf_k200          : the Zipf topic mixture from the smoke test (no single d), cosine
  tweets_encoder768  : 7.8k 2024 tweets, 768-d encoder embeddings from from-windows data, cosine
  gtr_{askdocs,trump,tifu}: SummEval GTR-T5-base 10k caches, cosine

For each corpus: TwoNN d on 2000 docs; G over n in N_GRID and r in a geometric grid; fit of
log(-log G) = a + b log n + d_fit log r on cells with n <= N_FIT; R^2 of that fit on held-out
cells n > N_FIT; required n for isolation rate 0.05 at the median r, predicted from the fit
versus empirical first crossing.

Writes results/eureka1_collapse.csv (one row per corpus) and results/eureka1_surface_<name>.csv.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corpus_variability.collapse import fit_collapse, isolation_surface, required_n, twonn_dimension  # noqa: E402
from corpus_variability.synthetic import make_corpus  # noqa: E402

N_GRID = [100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000, 7000]
N_FIT = 1000
EPS = 0.05
SUMMEVAL_CACHE = Path("/home/maolee/projects/SummEval/revision/experiments/cache")
OLD = ROOT / "from-windows-2026-07-26/data/random_embeddings_encoder.csv"


def manifold(n, d, D, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D, d)))
    Z = rng.normal(size=(n, d))
    return (Z @ Q.T).astype(np.float32)


def corpora():
    for d in (2, 4, 8):
        yield f"manifold_d{d}", manifold(20000, d, 64, d), "euclidean", d
    X, _ = make_corpus(20000, 200, 64, 1.0, 0.6, 0)
    yield "zipf_k200", X, "cosine", np.nan
    if OLD.exists():
        X = pd.read_csv(OLD, header=None).to_numpy(dtype=np.float32)
        if not np.issubdtype(X.dtype, np.floating) or X.shape[1] < 100:
            X = pd.read_csv(OLD).select_dtypes("number").to_numpy(dtype=np.float32)
        yield "tweets_encoder768", X, "cosine", np.nan
    for ds in ("askdocs", "trump", "tifu"):
        f = SUMMEVAL_CACHE / f"gtr_{ds}_42_10000.npy"
        if f.exists():
            yield f"gtr_{ds}", np.load(f, mmap_mode="r").astype(np.float32), "cosine", np.nan


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for name, X, metric, d_true in corpora():
        N = X.shape[0]
        n_grid = [n for n in N_GRID if n <= N]
        rng = np.random.default_rng(7)
        order = rng.permutation(N)
        pilot = X[order[:2000]]
        d_hat = twonn_dimension(pilot, metric)
        # radius grid from pilot NN distances: 5th to 95th percentile of 1-NN distance at n=2000
        from sklearn.neighbors import NearestNeighbors
        d1 = NearestNeighbors(n_neighbors=2, metric=metric).fit(pilot).kneighbors(pilot)[0][:, 1]
        r_grid = np.geomspace(np.percentile(d1, 5), np.percentile(d1, 99), 12)
        G = isolation_surface(X, n_grid, r_grid, order, metric)
        surf = pd.DataFrame(G, index=n_grid, columns=np.round(r_grid, 5))
        surf.to_csv(out / f"eureka1_surface_{name}.csv")
        fit_idx = [i for i, n in enumerate(n_grid) if n <= N_FIT]
        fit = fit_collapse(G[fit_idx], [n_grid[i] for i in fit_idx], r_grid)
        # held-out R^2 on n > N_FIT
        ho = [(np.log(n), np.log(r), np.log(-np.log(G[i, j])))
              for i, n in enumerate(n_grid) if n > N_FIT
              for j, r in enumerate(r_grid) if 0.02 < G[i, j] < 0.98]
        if ho and not np.isnan(fit["d"]):
            H = np.array(ho)
            pred = fit["a"] + fit["b"] * H[:, 0] + fit["d"] * H[:, 1]
            r2_ho = 1 - ((H[:, 2] - pred) ** 2).sum() / ((H[:, 2] - H[:, 2].mean()) ** 2).sum()
        else:
            r2_ho = np.nan
        # one-variable collapse with d fixed at TwoNN: log(-log G) = a + b log(n r^d_hat),
        # against the alternatives "n alone" and "r alone", all cells 0.02 < G < 0.98
        def r2_single(z):
            A = np.array([(z(n, r), np.log(-np.log(G[i, j])))
                          for i, n in enumerate(n_grid) for j, r in enumerate(r_grid)
                          if 0.02 < G[i, j] < 0.98])
            Xd = np.column_stack([np.ones(A.shape[0]), A[:, 0]])
            coef, *_ = np.linalg.lstsq(Xd, A[:, 1], rcond=None)
            res = A[:, 1] - Xd @ coef
            return 1 - (res ** 2).sum() / ((A[:, 1] - A[:, 1].mean()) ** 2).sum(), coef[1]
        r2_nrd, b_nrd = r2_single(lambda n, r: np.log(n) + d_hat * np.log(r))
        r2_n, _ = r2_single(lambda n, r: np.log(n))
        r2_r, _ = r2_single(lambda n, r: np.log(r))
        # required n: at the largest r with G(N_FIT, r) >= 0.25, predict n for G = EPS from the
        # fit on n <= N_FIT and compare with the log-interpolated empirical crossing at n > N_FIT
        i_fit = n_grid.index(N_FIT)
        js = [j for j in range(len(r_grid)) if G[i_fit, j] >= 0.25]
        j = js[-1] if js else len(r_grid) // 2
        r_mid = r_grid[j]
        n_pred = required_n(fit, EPS, r_mid) if not np.isnan(fit["d"]) else np.nan
        col = G[:, j]
        n_emp = np.inf
        for i in range(1, len(n_grid)):
            if col[i - 1] > EPS >= col[i]:
                w = (np.log(col[i - 1]) - np.log(EPS)) / (np.log(col[i - 1]) - np.log(max(col[i], 1e-6)))
                n_emp = float(np.exp(np.log(n_grid[i - 1]) + w * (np.log(n_grid[i]) - np.log(n_grid[i - 1]))))
                break
        rows.append(dict(corpus=name, N=N, metric=metric, d_true=d_true, d_twonn=round(d_hat, 2),
                         b_logn=round(fit["b"], 3), d_fit=round(fit["d"], 3),
                         d_ratio=round(fit["d"] / fit["b"], 2),
                         r2_fit=round(fit["r2"], 3), r2_heldout=round(r2_ho, 3), cells=fit["cells"],
                         r2_nrd=round(r2_nrd, 3), b_nrd=round(b_nrd, 3), r2_n_only=round(r2_n, 3),
                         r2_r_only=round(r2_r, 3),
                         r_star=round(r_mid, 4), n_pred=round(n_pred), n_emp=round(n_emp) if np.isfinite(n_emp) else np.inf))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "eureka1_collapse.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
