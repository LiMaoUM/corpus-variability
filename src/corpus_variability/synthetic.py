"""Synthetic corpora with known topic structure.

Topic means are random unit vectors; topic weights follow a Zipf law with exponent ``zipf_a``;
each document is its topic mean plus isotropic Gaussian noise, then L2-normalised so cosine
distance is the natural metric. The corpus is finite (``n_docs`` rows), matching the estimand.
"""

from __future__ import annotations

import numpy as np


def make_corpus(n_docs: int, n_topics: int, dim: int, zipf_a: float, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(n_topics, dim))
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    w = np.arange(1, n_topics + 1, dtype=float) ** (-zipf_a)
    w /= w.sum()
    labels = rng.choice(n_topics, size=n_docs, p=w)
    X = means[labels] + noise * rng.normal(size=(n_docs, dim)) / np.sqrt(dim)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32), labels
