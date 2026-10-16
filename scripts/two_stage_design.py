"""Choosing documents when only a random sample of the corpus can be embedded (Mao, 2026-10-16).

Fallback to the k-means design of resolution_and_design.py: draw a random pool of m documents, embed only
the pool, choose n documents within it by k-means, and measure the missing mass of the chosen documents
in the pool (what the researcher can compute) and in the whole corpus (the truth). Compared with k-means
on the whole corpus and with random samples of n, from results/designed_samples.csv.
Writes results/two_stage_design.csv.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collapse_v2 import corpora  # noqa: E402
from resolution_and_design import kmeans_design, nearest_to_set  # noqa: E402

POOLS = [1000, 5000]
NS = [15, 100]
DRAWS = 5


def main():
    des = pd.read_csv(ROOT / "results/designed_samples.csv")
    radii = des.drop_duplicates(["corpus", "level"])[["corpus", "level", "r"]]
    ref = des.set_index(["corpus", "level", "n"])
    rows = []
    for name, X, _, _ in corpora():
        sub = radii[radii.corpus == name]
        if sub.empty:
            continue
        N = X.shape[0]
        rng = np.random.default_rng(61)
        for m in POOLS:
            if m >= N:
                continue
            for draw in range(DRAWS):
                pool = rng.choice(N, m, replace=False)
                P = X[pool]
                for n in NS:
                    picks = kmeans_design(P, n, seed=draw)
                    d_pool = nearest_to_set(P, picks)
                    d_all = nearest_to_set(X, pool[picks])
                    for _, s in sub.iterrows():
                        rows.append(dict(corpus=name, N=N, level=s.level, r=s.r, pool=m, draw=draw, n=n,
                                         M_pool=float((d_pool > s.r).mean()), M_corpus=float((d_all > s.r).mean())))
        df = pd.DataFrame(rows)
        out = (df.groupby(["corpus", "N", "level", "r", "pool", "n"])[["M_pool", "M_corpus"]].mean().round(3)
               .reset_index())
        out["M_kmeans_full"] = [ref.loc[(c, l, n), "M_kmeans"] for c, l, n in zip(out.corpus, out.level, out.n)]
        out["M_random"] = [ref.loc[(c, l, n), "M_random"] for c, l, n in zip(out.corpus, out.level, out.n)]
        out.to_csv(ROOT / "results/two_stage_design.csv", index=False)
        print(out[out.corpus == name].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
