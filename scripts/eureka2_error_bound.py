"""Eureka 2 test: coverage as a margin of error for any r-stable downstream estimate.

Claim: for a function f on documents with range 1 that is stable at scale r (documents within r
share a value), the error of the 1-NN propagated sample estimate of the corpus mean of f is at
most  viol(r) + M(r), where M(r) is the missing mass at scale r (corpus documents farther than
r from every sampled document) and viol(r) is the share of covered corpus documents whose
nearest sampled document disagrees with them on f. Maurer's G(r) estimates M(r) from the sample
alone.

Evidence collected per corpus, n, r:
  M_true   : missing mass from the full corpus
  G_hat    : Maurer estimator from the sample
  max_err  : max over the f-family of |corpus mean - 1-NN propagated estimate|
  mean_err : mean over the family
  bound    : viol(r) + M_true(r)
  bound_hat: viol(r) + G_hat(r)
  hold     : max_err <= bound
  plain_max_err: max over family of |corpus mean - plain sample mean| (the survey estimator)

f-family: indicators of K = 50 k-means clusters fitted on the full corpus (a stand-in for a
categorical LLM annotation), plus "presence" indicators 1[cluster c has any sampled document].

Writes results/eureka2_error_bound.csv and prints hold rates, tightness and correlations.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corpus_variability.synthetic import make_corpus  # noqa: E402

N_GRID = [100, 200, 500, 1000, 2000, 5000]
K = 50
SUMMEVAL_CACHE = Path("/home/maolee/projects/SummEval/revision/experiments/cache")


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def corpora():
    X, y = make_corpus(20000, 200, 64, 1.0, 0.6, 0)
    yield "zipf_k200", unit(X)
    for ds in ("askdocs", "trump", "tifu"):
        f = SUMMEVAL_CACHE / f"gtr_{ds}_42_10000.npy"
        if f.exists():
            yield f"gtr_{ds}", unit(np.load(f, mmap_mode="r").astype(np.float32))


def main():
    rows = []
    for name, X in corpora():
        N = X.shape[0]
        labels = MiniBatchKMeans(K, random_state=0, n_init=3).fit_predict(X)
        P = np.bincount(labels, minlength=K) / N              # corpus cluster proportions
        rng = np.random.default_rng(11)
        order = rng.permutation(N)
        # radius grid from 1-NN distances of a 2000-doc pilot
        pilot = X[order[:2000]]
        d1 = NearestNeighbors(n_neighbors=2, metric="cosine").fit(pilot).kneighbors(pilot)[0][:, 1]
        r_grid = np.geomspace(np.percentile(d1, 10), np.percentile(d1, 97), 6)
        for n in [m for m in N_GRID if m <= N]:
            idx = order[:n]
            Xs, ls = X[idx], labels[idx]
            nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xs)
            dist_c, nn_c = nn.kneighbors(X, n_neighbors=1)          # corpus -> nearest sample
            dist_c, nn_c = dist_c[:, 0], nn_c[:, 0]
            d_s = nn.kneighbors(Xs)[0][:, 1]                          # sample -> nearest other
            prop = ls[nn_c]                                           # 1-NN propagated labels
            P_prop = np.bincount(prop, minlength=K) / N
            P_plain = np.bincount(ls, minlength=K) / n
            present_s = np.bincount(ls, minlength=K) > 0
            present_c = P > 0
            for r in r_grid:
                covered = dist_c <= r
                M_true = float((~covered).mean())
                G_hat = float((d_s > r).mean())
                viol = float(((prop != labels) & covered).mean())
                err_prop = np.abs(P_prop - P)
                err_presence = np.abs(present_c.astype(float) - present_s.astype(float)) * P
                max_err = float(max(err_prop.max(), err_presence.max()))
                rows.append(dict(corpus=name, n=n, r=round(float(r), 5), M_true=M_true, G_hat=G_hat,
                                 viol=viol, max_err=max_err, mean_err=float(err_prop.mean()),
                                 bound=viol + M_true, bound_hat=viol + G_hat,
                                 plain_max_err=float(np.abs(P_plain - P).max()),
                                 uncovered_mass_presence=float(err_presence.sum())))
            print(name, n, "M_true at r_mid=%.3f G_hat=%.3f" % (rows[-3]["M_true"], rows[-3]["G_hat"]), flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "eureka2_error_bound.csv", index=False)
    print("\nG_hat vs M_true: rho=%.3f MAE=%.4f" % (spearmanr(df.G_hat, df.M_true).statistic, (df.G_hat - df.M_true).abs().mean()))
    print("bound holds (max_err <= viol + M_true): %.3f of cells" % (df.max_err <= df.bound + 1e-12).mean())
    print("bound_hat holds (max_err <= viol + G_hat): %.3f of cells" % (df.max_err <= df.bound_hat + 1e-12).mean())
    print("median tightness bound / max_err (cells with max_err > 0.005): %.2f" % (df[df.max_err > 0.005].eval("bound / max_err").median()))
    print("rho(max_err, G_hat) = %.3f ; rho(plain_max_err, G_hat) = %.3f" % (
        spearmanr(df.max_err, df.G_hat).statistic, spearmanr(df.plain_max_err, df.G_hat).statistic))
    print("\nPer corpus, by n (median over r):")
    print(df.groupby(["corpus", "n"])[["M_true", "G_hat", "viol", "max_err", "bound", "plain_max_err"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
