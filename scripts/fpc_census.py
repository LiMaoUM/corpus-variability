"""FPC test on the Qwen3 corpora at large n/N (roadmap item 5): uses fpc_stats from
replicate_seeds on census slices, forum corpora under Qwen3, Truth Social and Bluesky.
Writes results/fpc_census.csv."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replicate_seeds import unit  # noqa: E402
EMB = ROOT / "data/emb"
names = {"census_random": "random", "census_citizens": "citizens", "census_trump": "trump", "census_covid": "covid",
         "census_illegal": "illegal", "askdocs_qwen3": "summeval_askdocs", "tifu_qwen3": "summeval_tifu", "trump_qwen3": "summeval_trump"}
rows = []
for name, f in names.items():
    X = unit(np.load(EMB / f"{f}_qwen3e8b.npy").astype(np.float32)); N = X.shape[0]
    rng = np.random.default_rng(7)
    pilot = X[rng.choice(N, 2000, replace=False)]
    d1 = NearestNeighbors(n_neighbors=2, metric="cosine").fit(pilot).kneighbors(pilot)[0][:, 1]
    radii = np.percentile(d1, [25, 50, 75])
    for frac in (0.01, 0.05, 0.1, 0.25, 0.5, 0.7):
        n = int(frac * N); idx = rng.choice(N, n, replace=False)
        nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X[idx])
        d_s = nn.kneighbors(X[idx])[0][:, 1]; d_c = nn.kneighbors(X, n_neighbors=1)[0][:, 0]
        for r in radii:
            G = float((d_s > r).mean()); M = float((d_c > r).mean())
            rows.append(dict(corpus=name, N=N, n=n, frac=frac, r=round(float(r), 4), M=M, G=G, G_fpc=G * (1 - n / N)))
    print(name, flush=True)
df = pd.DataFrame(rows); df.to_csv(ROOT / "results/fpc_census.csv", index=False)
df["e_raw"] = (df.G - df.M).abs(); df["e_fpc"] = (df.G_fpc - df.M).abs()
print("all cells: raw MAE %.4f max %.3f; fpc MAE %.4f max %.3f; n cells %d" % (df.e_raw.mean(), df.e_raw.max(), df.e_fpc.mean(), df.e_fpc.max(), len(df)))
print(df.groupby("frac")[["M", "G", "G_fpc", "e_raw", "e_fpc"]].mean().round(4).to_string())
print(df.groupby("corpus")[["e_raw", "e_fpc"]].mean().round(4).to_string())
