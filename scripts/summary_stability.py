"""Summary stability versus coverage on the five census corpora (card:
experiments/summary-stability.card.md).

Corpus side, per corpus: TwoNN d and participation ratio on a 2,000-doc pilot; G(50, r) and
finite-corpus M(50, r) averaged over N_DRAWS fresh draws of 50 documents at radii fixed across
corpora (quantiles of the pooled pilot 1-NN distances, so the scale is comparable).
Summary side: mean pairwise cosine distance among the 100 summaries, and mean distance to
their centroid, with a bootstrap band over summaries.
Agreement: Spearman across corpora between each corpus-side quantity and summary instability,
with a bootstrap band from resampling summaries and draws.

Reads data/emb/; writes results/summary_stability.csv and prints the table.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corpus_variability.collapse import participation_ratio, twonn_dimension  # noqa: E402

EMB = ROOT / "data/emb"
CORPORA = ["random", "citizens", "trump", "covid", "illegal"]
N_SAMPLE = 50
N_DRAWS = 200
R_QUANTILES = [0.25, 0.5, 0.75]
BOOT = 500


def pairwise_cos_dist(E):
    S = E @ E.T
    iu = np.triu_indices(E.shape[0], k=1)
    return 1.0 - S[iu]


def main():
    rng = np.random.default_rng(3)
    X = {c: np.load(EMB / f"{c}_qwen3e8b.npy").astype(np.float32) for c in CORPORA}
    S = {c: np.load(EMB / f"{c}_summaries_qwen3e8b.npy").astype(np.float32) for c in CORPORA}
    for c in CORPORA:
        X[c] /= np.linalg.norm(X[c], axis=1, keepdims=True)
        S[c] /= np.linalg.norm(S[c], axis=1, keepdims=True)
    # shared radii from pooled pilot 1-NN distances at n = 50 (the scale the summaries saw)
    pooled = []
    for c in CORPORA:
        for _ in range(20):
            idx = rng.choice(X[c].shape[0], N_SAMPLE, replace=False)
            d1 = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X[c][idx]).kneighbors(X[c][idx])[0][:, 1]
            pooled.append(d1)
    pooled = np.concatenate(pooled)
    radii = {q: float(np.quantile(pooled, q)) for q in R_QUANTILES}

    rows, draws = [], {}
    for c in CORPORA:
        Xc = X[c]
        N = Xc.shape[0]
        pilot = Xc[rng.choice(N, min(2000, N), replace=False)]
        d_tw = twonn_dimension(pilot, "cosine")
        pr = participation_ratio(pilot)
        G = {q: [] for q in R_QUANTILES}
        M = {q: [] for q in R_QUANTILES}
        for _ in range(N_DRAWS):
            idx = rng.choice(N, N_SAMPLE, replace=False)
            Xs = Xc[idx]
            nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xs)
            d_s = nn.kneighbors(Xs)[0][:, 1]
            d_c = nn.kneighbors(Xc, n_neighbors=1)[0][:, 0]
            for q, r in radii.items():
                G[q].append(float((d_s > r).mean()))
                M[q].append(float((d_c > r).mean()))
        draws[c] = G
        pd_s = pairwise_cos_dist(S[c])
        cen = S[c].mean(axis=0)
        cen /= np.linalg.norm(cen)
        to_cen = 1.0 - S[c] @ cen
        row = dict(corpus=c, N=N, d_twonn=round(d_tw, 2), part_ratio=round(pr, 2),
                   summary_pairwise=round(float(pd_s.mean()), 4),
                   summary_to_centroid=round(float(to_cen.mean()), 4),
                   doc_pairwise=round(float(pairwise_cos_dist(pilot[:500]).mean()), 4))
        for q in R_QUANTILES:
            row[f"G50_q{q}"] = round(float(np.mean(G[q])), 4)
            row[f"M50_q{q}"] = round(float(np.mean(M[q])), 4)
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results/summary_stability.csv", index=False)
    print("\nradii:", radii)
    print(df.to_string(index=False))

    print("\nSpearman across the five corpora with summary_pairwise (bootstrap 90% band over summaries and draws):")
    for col in ["d_twonn", "part_ratio", "doc_pairwise"] + [f"G50_q{q}" for q in R_QUANTILES] + [f"M50_q{q}" for q in R_QUANTILES]:
        rho = spearmanr(df[col], df.summary_pairwise).statistic
        boots = []
        for _ in range(BOOT):
            y = []
            x = []
            for c in CORPORA:
                bi = rng.choice(S[c].shape[0], S[c].shape[0], replace=True)
                y.append(pairwise_cos_dist(S[c][bi]).mean())
                if col.startswith("G50"):
                    q = float(col.split("q")[1])
                    g = np.array(draws[c][q])
                    x.append(np.mean(rng.choice(g, g.size, replace=True)))
                else:
                    x.append(df.set_index("corpus").loc[c, col])
            boots.append(spearmanr(x, y).statistic)
        lo, hi = np.nanpercentile(boots, [5, 95])
        print(f"  {col:14s} rho={rho:+.2f}  band=[{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
