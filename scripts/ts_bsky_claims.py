"""Claims 1 and 2 on the Truth Social and Bluesky 50k samples (card:
experiments/ts-bsky-embedding.card.md), with the collapse probe.

Reuses collapse_stats, fpc_stats and probe from replicate_seeds.py. Writes
results/ts_bsky_claims.csv and results/ts_bsky_probe.csv.
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
from replicate_seeds import collapse_stats, fpc_stats, probe, unit  # noqa: E402

EMB = ROOT / "data/emb"


def main():
    rows, probes = [], []
    for name in ("ts", "bsky"):
        X = unit(np.load(EMB / f"{name}_qwen3e8b.npy").astype(np.float32))
        st, order, r_grid = collapse_stats(X, np.random.default_rng(7))
        st.update(fpc_stats(X, order, r_grid))
        rows.append(dict(corpus=name, N=X.shape[0], **st))
        print(rows[-1], flush=True)
        probes.append(probe(name, X, np.random.default_rng(1)))
        print(probes[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/ts_bsky_claims.csv", index=False)
    pd.DataFrame(probes).to_csv(ROOT / "results/ts_bsky_probe.csv", index=False)
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print("\n" + pd.DataFrame(probes).to_string(index=False))


if __name__ == "__main__":
    main()
