"""Collapse law and sample size test, revision version (DECISIONS.md 2026-10-11, group 1 and 3).

Per corpus: TwoNN d with a bootstrap sd over pilots; an (n, r) grid with D independent draws per
n; the one-variable fit of log(-log G) on log(n r^d) on cells with n <= N_FIT, in-sample and
out-of-sample R2, mean absolute error of predicted G at the largest n; the same with N_FIT = 300;
the n* test at eps in {0.3, 0.2, 0.1} on every radius whose crossing lies beyond N_FIT, with
bootstrap intervals for the predicted and the empirical n*; and a per-radius Chao and Jost
extrapolation baseline from f1 and f2 at n = N_FIT.

Writes results/collapse_v2.csv (one row per corpus) and results/nstar_v2.csv (one row per case).
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
sys.path.insert(0, str(ROOT / "src"))
from corpus_variability.collapse import twonn_dimension  # noqa: E402
from corpus_variability.coverage import chao_extrapolated_coverage  # noqa: E402
from corpus_variability.synthetic import make_corpus  # noqa: E402

EMB = ROOT / "data/emb"
CACHE = Path("/home/maolee/projects/SummEval/revision/experiments/cache")
N_GRID = [100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000, 7000]
D_DRAWS = 5
EPS = (0.3, 0.2, 0.1)
BOOT = 200


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def manifold(n, d, D, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D, d)))
    return (rng.normal(size=(n, d)) @ Q.T).astype(np.float32)


def corpora():
    for d in (2, 4, 8):
        yield f"manifold_d{d}", manifold(20000, d, 64, d), "euclidean", "none"
    X, _ = make_corpus(20000, 200, 64, 1.0, 0.6, 0)
    yield "zipf_k200", unit(X), "cosine", "none"
    for ds in ("askdocs", "trump", "tifu"):
        for seed in (42, 43, 44):
            yield f"{ds}_gtr_s{seed}", unit(np.load(CACHE / f"gtr_{ds}_{seed}_10000.npy").astype(np.float32)), "cosine", "GTR"
        yield f"{ds}_qwen3", unit(np.load(EMB / f"summeval_{ds}_qwen3e8b.npy").astype(np.float32)), "cosine", "Qwen3"
    for c in ("random", "citizens", "trump", "covid", "illegal"):
        yield f"census_{c}_qwen3", unit(np.load(EMB / f"{c}_qwen3e8b.npy").astype(np.float32)), "cosine", "Qwen3"
    for c in ("ts", "bsky"):
        yield f"{c}_qwen3", unit(np.load(EMB / f"{c}_qwen3e8b.npy").astype(np.float32)), "cosine", "Qwen3"


def surface(X, n_grid, r_grid, metric, rng):
    """G[n, r, draw] and (f1, f2)[n, r, draw] with independent draws per n."""
    N = X.shape[0]
    G = np.full((len(n_grid), len(r_grid), D_DRAWS), np.nan)
    F2 = np.full_like(G, np.nan)
    for i, n in enumerate(n_grid):
        for k in range(D_DRAWS):
            idx = rng.choice(N, n, replace=False)
            Xs = X[idx]
            dist = NearestNeighbors(n_neighbors=3, metric=metric).fit(Xs).kneighbors(Xs)[0][:, 1:]
            for j, r in enumerate(r_grid):
                within = (dist <= r).sum(axis=1)
                G[i, j, k] = (within == 0).mean()
                F2[i, j, k] = (within == 1).mean()
    return G, F2


def fit_cells(G, n_grid, r_grid, d, n_max_fit, draws=None):
    rows = []
    for i, n in enumerate(n_grid):
        if n > n_max_fit:
            continue
        for j, r in enumerate(r_grid):
            for k in (range(G.shape[2]) if draws is None else draws):
                g = G[i, j, k]
                if 0.02 < g < 0.98:
                    rows.append((np.log(n) + d * np.log(r), np.log(-np.log(g))))
    A = np.array(rows)
    Xd = np.column_stack([np.ones(len(A)), A[:, 0]])
    (a, b), *_ = np.linalg.lstsq(Xd, A[:, 1], rcond=None)
    return a, b


def r2_on(G, n_grid, r_grid, d, a, b, n_min):
    ys, ps = [], []
    for i, n in enumerate(n_grid):
        if n < n_min:
            continue
        for j, r in enumerate(r_grid):
            for k in range(G.shape[2]):
                g = G[i, j, k]
                if 0.02 < g < 0.98:
                    ys.append(np.log(-np.log(g))); ps.append(a + b * (np.log(n) + d * np.log(r)))
    ys, ps = np.array(ys), np.array(ps)
    return 1 - ((ys - ps) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()


def crossing(col, n_grid, eps):
    for i in range(1, len(n_grid)):
        if col[i - 1] > eps >= col[i]:
            w = (np.log(col[i - 1]) - np.log(eps)) / (np.log(col[i - 1]) - np.log(max(col[i], 1e-6)))
            return float(np.exp(np.log(n_grid[i - 1]) + w * (np.log(n_grid[i]) - np.log(n_grid[i - 1]))))
    return np.nan


def main():
    rows, cases = [], []
    for name, X, metric, enc in corpora():
        N = X.shape[0]
        rng = np.random.default_rng(11)
        n_grid = [n for n in N_GRID if n <= 0.7 * N]
        pilots = [X[rng.choice(N, 2000, replace=False)] for _ in range(5)]
        ds = [twonn_dimension(p, metric) for p in pilots]
        d, d_sd = float(np.mean(ds)), float(np.std(ds))
        d1 = NearestNeighbors(n_neighbors=2, metric=metric).fit(pilots[0]).kneighbors(pilots[0])[0][:, 1]
        r_grid = np.geomspace(np.percentile(d1, 5), np.percentile(d1, 99), 12)
        G, F2 = surface(X, n_grid, r_grid, metric, rng)
        Gm = G.mean(axis=2)
        i_last = len(n_grid) - 1
        row = dict(corpus=name, encoder=enc, N=N, n_max=n_grid[i_last], d_twonn=round(d, 2), d_sd=round(d_sd, 2))
        for n_fit in (1000, 300):
            a, b = fit_cells(G, n_grid, r_grid, d, n_fit)
            r2_in = r2_on(G, n_grid, r_grid, d, a, b, 0)
            r2_out = r2_on(G, n_grid, r_grid, d, a, b, n_fit + 1)
            z = np.log(n_grid[i_last]) + d * np.log(r_grid)
            pred = np.exp(-np.exp(a + b * z))
            mae = float(np.abs(pred - Gm[i_last]).mean())
            tag = f"fit{n_fit}"
            row.update({f"{tag}_a": round(a, 3), f"{tag}_b": round(b, 3), f"{tag}_r2_in": round(r2_in, 3),
                        f"{tag}_r2_out": round(r2_out, 3), f"{tag}_mae_Gmax": round(mae, 4)})
            # Chao and Jost per-radius baseline from draw-mean f1, f2 at n = n_fit
            i_fit = n_grid.index(n_fit) if n_fit in n_grid else None
            if i_fit is not None:
                cj = []
                for j in range(len(r_grid)):
                    f1 = int(round(Gm[i_fit, j] * n_fit)); f2 = int(round(F2[i_fit, j].mean() * n_fit))
                    cj.append(1 - chao_extrapolated_coverage(n_fit, f1, f2, n_grid[i_last] - n_fit))
                row[f"{tag}_mae_Gmax_chaojost"] = round(float(np.abs(np.array(cj) - Gm[i_last]).mean()), 4)
            # n* test
            boots = []
            for _ in range(BOOT):
                bd = rng.choice(D_DRAWS, D_DRAWS, replace=True)
                boots.append(fit_cells(G, n_grid, r_grid, d, n_fit, draws=bd))
            for eps in EPS:
                for j, r in enumerate(r_grid):
                    n_emp = crossing(Gm[:, j], n_grid, eps)
                    if not np.isfinite(n_emp) or n_emp <= n_fit:
                        continue
                    n_emp_draws = [crossing(G[:, j, k], n_grid, eps) for k in range(D_DRAWS)]
                    n_pred = (np.log(1 / eps) / np.exp(a)) ** (1 / b) / r ** d
                    bp = [(np.log(1 / eps) / np.exp(aa)) ** (1 / bb) / r ** d for aa, bb in boots]
                    cases.append(dict(corpus=name, encoder=enc, n_fit=n_fit, eps=eps, r=round(float(r), 4),
                                      n_pred=round(n_pred), n_pred_lo=round(float(np.nanpercentile(bp, 5))),
                                      n_pred_hi=round(float(np.nanpercentile(bp, 95))), n_emp=round(n_emp),
                                      n_emp_lo=round(float(np.nanmin(n_emp_draws))), n_emp_hi=round(float(np.nanmax(n_emp_draws))),
                                      ratio=round(n_pred / n_emp, 2)))
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows); cs = pd.DataFrame(cases)
    df.to_csv(ROOT / "results/collapse_v2.csv", index=False)
    cs.to_csv(ROOT / "results/nstar_v2.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("\nn* test by corpus and fit range: cases, median ratio, within 1.5, within 2")
    for (c, nf), g in cs.groupby(["corpus", "n_fit"]):
        print(f"  {c:22s} fit<={nf:4d} k={len(g):2d} median={g.ratio.median():.2f} in1.5={((g.ratio > 1/1.5) & (g.ratio < 1.5)).sum():2d} in2={((g.ratio > 0.5) & (g.ratio < 2)).sum():2d}")


if __name__ == "__main__":
    main()
