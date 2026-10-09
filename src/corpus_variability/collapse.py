"""Collapse law for the coverage surface and intrinsic-dimension estimation.

Hypothesis (eureka 1): if documents lie on a manifold of intrinsic dimension d, the isolation
rate G(n, r) of Maurer's estimator follows G ~ exp(-c n r^d), so the (n, r) surface collapses
onto one curve in n r^d, and the required sample size for isolation rate eps at scale r is
n*(eps, r) = log(1/eps) / (c r^d).

``twonn_dimension`` is the TwoNN estimator of Facco et al. (2017, Sci Rep 7:12140).
``fit_collapse`` regresses log(-log G) on log n and log r; the coefficient on log n should be
1 and the coefficient on log r should be d.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def twonn_dimension(X: np.ndarray, metric: str = "cosine", discard_top: float = 0.1) -> float:
    nn = NearestNeighbors(n_neighbors=3, metric=metric).fit(X)
    dist, _ = nn.kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    n = mu.size
    keep = int(np.floor(n * (1.0 - discard_top)))
    mu = mu[:keep]
    F = np.arange(1, keep + 1) / n
    x = np.log(mu)
    y = -np.log(1.0 - F)
    return float((x * y).sum() / (x * x).sum())


def isolation_surface(X: np.ndarray, n_grid: list[int], r_grid: np.ndarray, order: np.ndarray,
                      metric: str = "cosine") -> np.ndarray:
    """G[n_i, r_j] for nested samples order[:n_i]; counts points with no other point within r."""
    G = np.full((len(n_grid), len(r_grid)), np.nan)
    for i, n in enumerate(n_grid):
        Xs = X[order[:n]]
        nn = NearestNeighbors(n_neighbors=2, metric=metric).fit(Xs)
        d1 = nn.kneighbors(Xs)[0][:, 1]
        for j, r in enumerate(r_grid):
            G[i, j] = float((d1 > r).mean())
    return G


def fit_collapse(G: np.ndarray, n_grid: list[int], r_grid: np.ndarray,
                 g_min: float = 0.02, g_max: float = 0.98) -> dict:
    """Least squares of log(-log G) = a + b log n + d log r over cells with g_min < G < g_max."""
    rows = []
    for i, n in enumerate(n_grid):
        for j, r in enumerate(r_grid):
            g = G[i, j]
            if g_min < g < g_max:
                rows.append((np.log(n), np.log(r), np.log(-np.log(g))))
    A = np.array(rows)
    if A.shape[0] < 4:
        return dict(a=np.nan, b=np.nan, d=np.nan, r2=np.nan, cells=A.shape[0])
    Xd = np.column_stack([np.ones(A.shape[0]), A[:, 0], A[:, 1]])
    coef, *_ = np.linalg.lstsq(Xd, A[:, 2], rcond=None)
    pred = Xd @ coef
    ss_res = ((A[:, 2] - pred) ** 2).sum()
    ss_tot = ((A[:, 2] - A[:, 2].mean()) ** 2).sum()
    return dict(a=float(coef[0]), b=float(coef[1]), d=float(coef[2]),
                r2=float(1 - ss_res / ss_tot), cells=A.shape[0])


def required_n(fit: dict, eps: float, r: float) -> float:
    """n with exp(a) n^b r^d = log(1/eps), solved for n."""
    return float((np.log(1.0 / eps) / (np.exp(fit["a"]) * r ** fit["d"])) ** (1.0 / fit["b"]))
