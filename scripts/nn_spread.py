"""How tightly each corpus clusters in embedding space (Mao, 2026-10-17).

For each corpus, a random 5,000 documents (the same number everywhere, so that corpus size does not
shorten the distances), the cosine distance from every document to its nearest neighbour among the
5,000, and the median distance over all pairs. Set beside the judged distances r_same (Qwen3.8) and
r_topic from results/designed_samples.csv, which say what a distance means in each corpus.
Bluesky is left out: it has no reliable judged distance. Writes results/nn_spread.csv (summary) and
results/nn_spread.npz (the nearest neighbour distances, for the figure).
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
from collapse_v2 import EMB, unit  # noqa: E402

M = 5000
FILES = {"askdocs": "summeval_askdocs", "trump": "summeval_trump", "tifu": "summeval_tifu",
         "census_random": "random", "census_citizens": "citizens", "census_trump": "trump",
         "census_covid": "covid", "census_illegal": "illegal", "ts": "ts"}


def main():
    radii = (pd.read_csv(ROOT / "results/designed_samples.csv").drop_duplicates(["corpus", "level"])
             .pivot(index="corpus", columns="level", values="r"))
    rng = np.random.default_rng(17)
    rows, arrays = [], {}
    for name, f in FILES.items():
        X = unit(np.load(EMB / f"{f}_qwen3e8b.npy").astype(np.float32))
        X = X[rng.choice(len(X), M, replace=False)]
        D = 1 - X @ X.T
        np.fill_diagonal(D, np.inf)
        nn = D.min(1)
        pairs = D[np.triu_indices(M, 1)]
        r_same = radii.loc[f"{name}_qwen3"].get("same", np.nan)
        r_topic = radii.loc[f"{name}_qwen3"].get("topic", np.nan)
        arrays[name] = nn
        rows.append(dict(corpus=name, nn_q25=np.quantile(nn, .25), nn_median=np.median(nn),
                         nn_q75=np.quantile(nn, .75), pair_median=np.median(pairs), r_same=r_same, r_topic=r_topic,
                         nn_within_same=(nn <= r_same).mean(),
                         nn_within_topic=(nn <= r_topic).mean() if r_topic == r_topic else np.nan))
    out = pd.DataFrame(rows).sort_values("nn_median")
    out.round(4).to_csv(ROOT / "results/nn_spread.csv", index=False)
    np.savez_compressed(ROOT / "results/nn_spread.npz", **arrays)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
