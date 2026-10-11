"""Greedy k-center covering as a baseline for the sample size formula (DECISIONS.md 2026-10-11,
group 3). For each corpus, greedy farthest-point k-center on the full embeddings up to K_MAX
centers; at each of the radii used in collapse_v2 record the missing mass M(k, r) of the center
set, and compare the k at which M reaches eps with the random-sample n at which M reaches eps
(mean over five draws). Writes results/kcenter_baseline.csv.
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
from summary_n_curve import unit  # noqa: E402

EMB = ROOT / "data/emb"
CACHE = Path("/home/maolee/projects/SummEval/revision/experiments/cache")
K_MAX = 5000
CHECK = [100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000]
EPS = (0.3, 0.2, 0.1)


def corpora():
    for ds in ("askdocs", "tifu", "trump"):
        yield f"{ds}_gtr_s42", unit(np.load(CACHE / f"gtr_{ds}_42_10000.npy").astype(np.float32))
    for c in ("random", "citizens"):
        yield f"census_{c}_qwen3", unit(np.load(EMB / f"{c}_qwen3e8b.npy").astype(np.float32))
    for c in ("ts", "bsky"):
        yield f"{c}_qwen3", unit(np.load(EMB / f"{c}_qwen3e8b.npy").astype(np.float32))


def greedy_kcenter(X, k_max, rng):
    N = X.shape[0]
    centers = [int(rng.integers(N))]
    mind = 1 - X @ X[centers[0]]
    snapshots = {}
    for k in range(2, k_max + 1):
        c = int(np.argmax(mind))
        centers.append(c)
        mind = np.minimum(mind, 1 - X @ X[c])
        if k in CHECK:
            snapshots[k] = mind.copy()
    return snapshots


def main():
    rows = []
    for name, X in corpora():
        N = X.shape[0]
        rng = np.random.default_rng(3)
        pilot = X[rng.choice(N, 2000, replace=False)]
        d1 = NearestNeighbors(n_neighbors=2, metric="cosine").fit(pilot).kneighbors(pilot)[0][:, 1]
        r_grid = np.geomspace(np.percentile(d1, 5), np.percentile(d1, 99), 12)
        snaps = greedy_kcenter(X, min(K_MAX, int(0.7 * N)), rng)
        ks = sorted(snaps)
        M_k = {r: [float((snaps[k] > r).mean()) for k in ks] for r in r_grid}
        # random sample missing mass at the same sizes
        M_n = {r: [] for r in r_grid}
        for n in ks:
            vals = {r: [] for r in r_grid}
            for _ in range(5):
                idx = rng.choice(N, n, replace=False)
                dc = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X[idx]).kneighbors(X)[0][:, 0]
                for r in r_grid:
                    vals[r].append(float((dc > r).mean()))
            for r in r_grid:
                M_n[r].append(float(np.mean(vals[r])))

        def cross(vals, eps):
            for i in range(1, len(ks)):
                if vals[i - 1] > eps >= vals[i]:
                    w = (np.log(vals[i - 1]) - np.log(eps)) / (np.log(vals[i - 1]) - np.log(max(vals[i], 1e-6)))
                    return float(np.exp(np.log(ks[i - 1]) + w * (np.log(ks[i]) - np.log(ks[i - 1]))))
            return np.nan
        for r in r_grid:
            for eps in EPS:
                k_star, n_star = cross(M_k[r], eps), cross(M_n[r], eps)
                rows.append(dict(corpus=name, r=round(float(r), 4), eps=eps, k_center=round(k_star) if np.isfinite(k_star) else np.nan,
                                 n_random=round(n_star) if np.isfinite(n_star) else np.nan,
                                 ratio_random_over_kcenter=round(n_star / k_star, 2) if np.isfinite(k_star) and np.isfinite(n_star) else np.nan))
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/kcenter_baseline.csv", index=False)
    print(df.dropna().groupby("corpus").ratio_random_over_kcenter.describe()[["count", "50%", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
