"""Replicate claims 1 and 2 on the SummEval GTR caches for seeds 42, 43 and 44, and probe why
Trump (and the synthetic Zipf mixture) fail the collapse test.

Claim 1 per (corpus, seed): TwoNN d, one-variable collapse R2 on n r^d, held-out R2
(fit n <= 1000), MAE of predicted G at n = 7000.
Claim 2 per (corpus, seed): MAE of G and of G (1 - n/N) against the finite-corpus missing
mass over n in {100, 500, 2000, 5000} and three radii.
Probe per corpus (seed 42): near-duplicate share (1-NN cosine distance < 0.02 in a 5,000-doc
sample), spread of TwoNN d across 20 k-means clusters, and collapse R2 after removing
near-duplicates (greedy dedupe at 0.02).

Writes results/replicate_seeds.csv and results/collapse_probe.csv; prints both.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corpus_variability.collapse import isolation_surface, twonn_dimension  # noqa: E402
from corpus_variability.synthetic import make_corpus  # noqa: E402

CACHE = Path("/home/maolee/projects/SummEval/revision/experiments/cache")
N_GRID = [100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000, 7000]
N_FIT = 1000


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def collapse_stats(X, rng):
    N = X.shape[0]
    order = rng.permutation(N)
    pilot = X[order[:2000]]
    d = twonn_dimension(pilot)
    d1 = NearestNeighbors(n_neighbors=2, metric="cosine").fit(pilot).kneighbors(pilot)[0][:, 1]
    r_grid = np.geomspace(np.percentile(d1, 5), np.percentile(d1, 99), 12)
    n_grid = [n for n in N_GRID if n <= N]
    G = isolation_surface(X, n_grid, r_grid, order)
    cells = [(np.log(n) + d * np.log(r), np.log(-np.log(G[i, j])), n)
             for i, n in enumerate(n_grid) for j, r in enumerate(r_grid) if 0.02 < G[i, j] < 0.98]
    A = np.array([(z, y) for z, y, _ in cells])
    fit_mask = np.array([n <= N_FIT for *_, n in cells])

    def r2(mask, coef=None):
        Xd = np.column_stack([np.ones(mask.sum()), A[mask, 0]])
        if coef is None:
            coef, *_ = np.linalg.lstsq(Xd, A[mask, 1], rcond=None)
        res = A[mask, 1] - Xd @ coef
        return 1 - (res ** 2).sum() / ((A[mask, 1] - A[mask, 1].mean()) ** 2).sum(), coef

    r2_all, _ = r2(np.ones(len(cells), bool))
    _, coef_fit = r2(fit_mask)
    r2_ho, _ = r2(~fit_mask, coef_fit)
    i_last = len(n_grid) - 1
    z = np.log(n_grid[i_last]) + d * np.log(r_grid)
    G_pred = np.exp(-np.exp(coef_fit[0] + coef_fit[1] * z))
    mae = float(np.abs(G_pred - G[i_last]).mean())
    return dict(d_twonn=round(d, 2), r2_nrd=round(r2_all, 3), r2_heldout=round(r2_ho, 3),
                b=round(float(coef_fit[1]), 3), mae_pred_G=round(mae, 4), n_target=n_grid[i_last]), order, r_grid


def fpc_stats(X, order, r_grid):
    N = X.shape[0]
    errs_raw, errs_fpc = [], []
    for n in [100, 500, 2000, 5000]:
        Xs = X[order[:n]]
        nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xs)
        d_s = nn.kneighbors(Xs)[0][:, 1]
        d_c = nn.kneighbors(X, n_neighbors=1)[0][:, 0]
        for r in r_grid[[3, 6, 9]]:
            G = float((d_s > r).mean())
            M = float((d_c > r).mean())
            errs_raw.append(abs(G - M))
            errs_fpc.append(abs(G * (1 - n / N) - M))
    return dict(mae_G_vs_M=round(float(np.mean(errs_raw)), 4), mae_Gfpc_vs_M=round(float(np.mean(errs_fpc)), 4))


def probe(name, X, rng):
    N = X.shape[0]
    idx = rng.choice(N, min(5000, N), replace=False)
    P = X[idx]
    d1 = NearestNeighbors(n_neighbors=2, metric="cosine").fit(P).kneighbors(P)[0][:, 1]
    near_dup = float((d1 < 0.02).mean())
    labels = MiniBatchKMeans(20, random_state=0, n_init=3).fit_predict(P)
    ds = [twonn_dimension(P[labels == k]) for k in range(20) if (labels == k).sum() >= 50]
    # greedy dedupe at 0.02 on the full corpus
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X)
    dist, ind = nn.kneighbors(X)
    keep = np.ones(N, bool)
    for i in range(N):
        j = ind[i, 1]
        if keep[i] and keep[j] and dist[i, 1] < 0.02 and j > i:
            keep[j] = False
    Xd = X[keep]
    st, _, _ = collapse_stats(Xd, np.random.default_rng(7))
    return dict(corpus=name, N=N, near_dup_share=round(near_dup, 4), local_d_min=round(min(ds), 2),
                local_d_max=round(max(ds), 2), local_d_cv=round(float(np.std(ds) / np.mean(ds)), 3),
                N_dedup=int(keep.sum()), r2_nrd_dedup=st["r2_nrd"], r2_heldout_dedup=st["r2_heldout"])


def main():
    rows = []
    for ds in ("askdocs", "trump", "tifu"):
        for seed in (42, 43, 44):
            X = unit(np.load(CACHE / f"gtr_{ds}_{seed}_10000.npy").astype(np.float32))
            rng = np.random.default_rng(7)
            st, order, r_grid = collapse_stats(X, rng)
            st.update(fpc_stats(X, order, r_grid))
            rows.append(dict(corpus=ds, seed=seed, **st))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/replicate_seeds.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("\nMean over seeds:")
    print(df.groupby("corpus")[["d_twonn", "r2_nrd", "r2_heldout", "mae_pred_G", "mae_G_vs_M", "mae_Gfpc_vs_M"]].agg(["mean", "std"]).round(3).to_string())

    probes = []
    for ds in ("askdocs", "trump", "tifu"):
        X = unit(np.load(CACHE / f"gtr_{ds}_42_10000.npy").astype(np.float32))
        probes.append(probe(ds, X, np.random.default_rng(1)))
        print(probes[-1], flush=True)
    Xz, _ = make_corpus(20000, 200, 64, 1.0, 0.6, 0)
    probes.append(probe("zipf_k200", unit(Xz), np.random.default_rng(1)))
    print(probes[-1], flush=True)
    pdf = pd.DataFrame(probes)
    pdf.to_csv(ROOT / "results/collapse_probe.csv", index=False)
    print("\n" + pdf.to_string(index=False))


if __name__ == "__main__":
    main()
