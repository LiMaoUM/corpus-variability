"""Coverage estimators for a sample of a corpus.

Three estimators of how much of a corpus a sample covers:

- ``true_coverage``: the corpus topic mass whose topic appears in the sample. Needs labels on
  the full corpus; it is the validation target.
- ``good_turing_coverage``: ``1 - f1 / n`` on sample topic labels. Needs labels on the sample.
- ``nn_coverage``: the continuous analogue. A sampled document whose nearest sampled neighbour
  is farther than ``r`` counts as a singleton. Needs embeddings only.

``chao_required_n`` extrapolates the coverage curve to the sample size that reaches a target
coverage (Chao and Jost 2012, Ecology 93:2533, eq. 9 solved for m).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def true_coverage(corpus_labels: np.ndarray, sample_labels: np.ndarray) -> float:
    mass = np.bincount(corpus_labels) / corpus_labels.size
    seen = np.unique(sample_labels)
    return float(mass[seen].sum())


def good_turing_coverage(sample_labels: np.ndarray) -> float:
    counts = np.bincount(sample_labels)
    f1 = int((counts == 1).sum())
    return 1.0 - f1 / sample_labels.size


def nn_counts(X: np.ndarray, r: float, metric: str = "cosine") -> tuple[int, int]:
    """Number of sampled points with zero (f1) and exactly one (f2) other sampled point within r."""
    k = min(3, X.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
    dist, _ = nn.kneighbors(X)
    others = dist[:, 1:]  # drop self
    within = (others <= r).sum(axis=1)
    f1 = int((within == 0).sum())
    f2 = int((within == 1).sum())
    return f1, f2


def nn_coverage(X: np.ndarray, r: float, metric: str = "cosine") -> float:
    f1, _ = nn_counts(X, r, metric)
    return 1.0 - f1 / X.shape[0]


def pilot_radius(X: np.ndarray, alpha: float, metric: str = "cosine", rng=None, m: int = 500) -> float:
    """Label-free radius: ``alpha`` times the median pairwise distance of a pilot of ``m`` docs."""
    rng = np.random.default_rng(rng)
    idx = rng.choice(X.shape[0], size=min(m, X.shape[0]), replace=False)
    P = X[idx]
    if metric == "cosine":
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        D = 1.0 - P @ P.T
    else:
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(P.shape[0], k=1)
    return float(alpha * np.median(D[iu]))


def chao_extrapolated_coverage(n: int, f1: int, f2: int, m: int) -> float:
    """Coverage expected at sample size n + m (Chao and Jost 2012, eq. 9)."""
    if f1 == 0:
        return 1.0
    if f2 == 0:
        ratio = (n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2) if f1 > 1 else 0.0
    else:
        ratio = (n - 1) * f1 / ((n - 1) * f1 + 2 * f2)
    return 1.0 - (f1 / n) * ratio ** (m + 1)


def chao_required_n(n: int, f1: int, f2: int, target: float) -> float:
    """Smallest sample size at which extrapolated coverage reaches ``target``; inf if unreachable."""
    if f1 == 0:
        return float(n)
    current = 1.0 - f1 / n
    if current >= target:
        return float(n)
    if f2 == 0:
        ratio = (n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2) if f1 > 1 else 0.0
    else:
        ratio = (n - 1) * f1 / ((n - 1) * f1 + 2 * f2)
    if ratio <= 0 or ratio >= 1:
        return float("inf")
    # 1 - (f1/n) ratio^(m+1) = target  ->  m = log((1-target) n / f1) / log(ratio) - 1
    m = np.log((1.0 - target) * n / f1) / np.log(ratio) - 1.0
    return float(n + max(m, 0.0))


def vendi_score(X: np.ndarray) -> float:
    """Exponential entropy of the normalised cosine kernel (Friedman and Dieng 2023)."""
    Z = X / np.linalg.norm(X, axis=1, keepdims=True)
    K = Z @ Z.T / Z.shape[0]
    w = np.linalg.eigvalsh(K)
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))
