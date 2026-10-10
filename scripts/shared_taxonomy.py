"""Absolute-scale cross-corpus test, label version (DECISIONS.md 2026-10-09).

Shared taxonomy: the 8 x 30 theme centroids (mean Qwen3 embedding of each cluster) are merged
by average-linkage agglomerative clustering at cosine distance MERGE_R; every document of every
corpus is assigned to its nearest shared centroid. For each existing draw: shared coverage =
corpus mass of shared themes with at least one drawn document; themes seen = count. Predictors:
1 - G(n, r) at absolute radii, and the mass-only Good-Turing expectation. Cross-corpus Spearman
at each n with bootstrap bands over draws.

Writes results/shared_taxonomy.csv, results/shared_taxonomy_draws.csv, data/themes/shared_*.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summary_n_curve import CORPUS_EMB, EMB, SUMM, unit  # noqa: E402

TH = ROOT / "data/themes"
MERGE_R = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
RADII = {"r0.20": 0.20, "r0.30": 0.30, "r0.40": 0.40}


def main():
    X = {c: unit(np.load(EMB / f).astype(np.float32)) for c, f in CORPUS_EMB.items()}
    labels = {c: np.load(TH / f"{c}_labels.npy") for c in X}
    names = {c: json.load((TH / f"{c}_themes.json").open())["themes"] for c in X}
    cents, owner = [], []
    for c in X:
        for k in range(30):
            m = labels[c] == k
            v = X[c][m].mean(axis=0)
            cents.append(v / np.linalg.norm(v)); owner.append((c, k))
    C = np.array(cents)
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=MERGE_R, metric="cosine", linkage="average").fit(C)
    S = agg.labels_
    n_shared = S.max() + 1
    shared_cents = np.array([unit(C[S == s].mean(axis=0, keepdims=True))[0] for s in range(n_shared)])
    print(f"merge radius {MERGE_R}: 240 themes -> {n_shared} shared themes", flush=True)
    merged = {}
    for (c, k), s in zip(owner, S):
        merged.setdefault(int(s), []).append(f"{c}:{names[c][k]['name']}")
    json.dump(dict(merge_r=MERGE_R, n_shared=int(n_shared), members=merged), (TH / "shared_themes.json").open("w"), indent=1)
    big = sorted(merged.items(), key=lambda kv: -len(kv[1]))[:5]
    for s, mem in big:
        print(f"  shared {s} ({len(mem)}): " + "; ".join(mem[:6]), flush=True)

    shared_lab, mass = {}, {}
    for c in X:
        sl = np.argmax(X[c] @ shared_cents.T, axis=1)
        shared_lab[c] = sl
        mass[c] = np.bincount(sl, minlength=n_shared) / sl.size
        np.save(TH / f"shared_labels_{c}.npy", sl)
    occupied = {c: int((mass[c] > 0.002).sum()) for c in X}
    print("shared themes with mass > 0.2% per corpus:", occupied, flush=True)

    rows = []
    for c in X:
        for n in (25, 50, 100, 200, 400):
            f = SUMM / f"{c}_n{n}.jsonl"
            if not f.exists():
                continue
            for rec in (json.loads(l) for l in f.read_text().splitlines() if l.strip()):
                idx = np.array(rec["idx"])
                seen = np.zeros(n_shared, bool); seen[np.unique(shared_lab[c][idx])] = True
                Xs = X[c][idx]
                d_s = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xs).kneighbors(Xs)[0][:, 1]
                row = dict(corpus=c, n=n, draw=rec["draw"], shared_cov=float(mass[c][seen].sum()),
                           themes_seen=int(seen.sum()), themes_occupied=occupied[c])
                for k, r in RADII.items():
                    row[f"cov_{k}"] = 1 - float((d_s > r).mean())
                rows.append(row)
    dd = pd.DataFrame(rows)
    dd.to_csv(ROOT / "results/shared_taxonomy_draws.csv", index=False)
    df = dd.groupby(["corpus", "n"]).mean(numeric_only=True).drop(columns="draw").reset_index()
    for i, r in df.iterrows():
        m = mass[r.corpus]
        df.loc[i, "gt_expected"] = float((m * (1 - (1 - m) ** r.n)).sum())
    df.to_csv(ROOT / "results/shared_taxonomy.csv", index=False)
    print(df.round(3).to_string(index=False))

    rng = np.random.default_rng(2)
    print("\nCross-corpus spread of shared_cov by n: min, max, sd")
    for n, g in df.groupby("n"):
        print(f"  n={n:4d}  {g.shared_cov.min():.3f} {g.shared_cov.max():.3f} sd={g.shared_cov.std():.3f}")
    print("\nCross-corpus Spearman of shared_cov with predictors at each n (bootstrap 90% band over draws):")
    for n, g in df.groupby("n"):
        for col in [f"cov_{k}" for k in RADII] + ["gt_expected"]:
            rho = spearmanr(g[col], g.shared_cov).statistic
            boots = []
            for _ in range(300):
                x, y = [], []
                for c in g.corpus:
                    gd = dd[(dd.corpus == c) & (dd.n == n)]
                    bi = rng.choice(len(gd), len(gd), replace=True)
                    y.append(gd.shared_cov.to_numpy()[bi].mean())
                    x.append(gd[col].to_numpy()[bi].mean() if col in gd else float(g[g.corpus == c][col].iloc[0]))
                boots.append(spearmanr(x, y).statistic)
            lo, hi = np.nanpercentile(boots, [5, 95])
            print(f"  n={n:4d} {col:12s} rho={rho:+.2f} band=[{lo:+.2f}, {hi:+.2f}]")
    print("\nPooled per-draw Spearman within (corpus, n), shared_cov vs cov_r0.30: median rho=%+.2f" %
          np.nanmedian([spearmanr(g["cov_r0.30"], g.shared_cov).statistic for _, g in dd.groupby(["corpus", "n"]) if g.shared_cov.std() > 0]))


if __name__ == "__main__":
    main()
