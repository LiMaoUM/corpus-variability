"""Exploratory follow-up to summary_n_curve (outside the card): corpus-specific radii.

Each corpus gets its own radius, the median 1-NN cosine distance of 50-document draws, and G
is recomputed on the same draws. Reports cross-corpus Spearman of instability with G at each n
and the within-corpus log-log slope. Reads results/summary_n_curve.csv and data/summaries/.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summary_n_curve import CORPUS_EMB, EMB, SUMM, unit  # noqa: E402

df = pd.read_csv(ROOT / "results/summary_n_curve.csv")
rng = np.random.default_rng(9)
rows = []
for c, f in CORPUS_EMB.items():
    X = unit(np.load(EMB / f).astype(np.float32))
    d1 = []
    for _ in range(30):
        idx = rng.choice(X.shape[0], 50, replace=False)
        d1.append(NearestNeighbors(n_neighbors=2, metric="cosine").fit(X[idx]).kneighbors(X[idx])[0][:, 1])
    r = float(np.median(np.concatenate(d1)))
    for n in [25, 50, 100, 200, 400]:
        recs = [json.loads(l) for l in (SUMM / f"{c}_n{n}.jsonl").read_text().splitlines() if l.strip()]
        g = []
        for rec in recs:
            Xs = X[np.array(rec["idx"])]
            ds = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xs).kneighbors(Xs)[0][:, 1]
            g.append(float((ds > r).mean()))
        rows.append(dict(corpus=c, n=n, r_own=round(r, 4), G_own=round(float(np.mean(g)), 4)))
own = pd.DataFrame(rows).merge(df[["corpus", "n", "instability", "d_twonn"]], on=["corpus", "n"])
own.to_csv(ROOT / "results/n_curve_exploratory.csv", index=False)
print(own.to_string(index=False))
print("\nCross-corpus Spearman(instability, G_own) by n:")
for n, g in own.groupby("n"):
    print(f"  n={n:4d} rho={spearmanr(g.G_own, g.instability).statistic:+.2f}")
print("\nSpearman(instability, r_own) by n (is the corpus's own scale predictive?):")
for n, g in own.groupby("n"):
    print(f"  n={n:4d} rho={spearmanr(g.r_own, g.instability).statistic:+.2f}")
