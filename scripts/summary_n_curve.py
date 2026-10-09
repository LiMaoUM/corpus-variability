"""Analysis for the summary-n-curve study (card: experiments/summary-n-curve.card.md).

Step 1 (GPU): embed every summary in data/summaries/ with Qwen3-Embedding-8B into
data/emb/ncurve_summaries_qwen3e8b.npz (keys <corpus>_n<n>, rows ordered by draw).
Step 2 (CPU): per (corpus, n): instability = mean pairwise cosine distance among the 30
summaries; G(n, r) and M(n, r) of the same draws at three shared radii; per-draw G and per-draw
distance to the (corpus, n) summary centroid. Tests: Spearman across corpora at each n with
bootstrap bands; within-corpus slope of log instability on log G; pooled per-draw Spearman.

Writes results/summary_n_curve.csv (per corpus, n), results/summary_n_curve_draws.csv and
prints the summary.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corpus_variability.collapse import twonn_dimension  # noqa: E402

EMB = ROOT / "data/emb"
SUMM = ROOT / "data/summaries"
CORPUS_EMB = {
    "random": "random_qwen3e8b.npy", "citizens": "citizens_qwen3e8b.npy", "trump": "trump_qwen3e8b.npy",
    "covid": "covid_qwen3e8b.npy", "illegal": "illegal_qwen3e8b.npy",
    "sv_askdocs": "summeval_askdocs_qwen3e8b.npy", "sv_trump": "summeval_trump_qwen3e8b.npy",
    "sv_tifu": "summeval_tifu_qwen3e8b.npy",
}
R_QUANTILES = [0.25, 0.5, 0.75]
BOOT = 500


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def load_summaries():
    out = {}
    for f in sorted(SUMM.glob("*_n*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: r["draw"])
        out[f.stem] = rows
    return out


def embed_step(device):
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device=device, model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 512
    arrays = {}
    for key, rows in load_summaries().items():
        E = model.encode([r["summary"] for r in rows], batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        arrays[key] = E.astype(np.float16)
        print(key, E.shape, flush=True)
    np.savez(EMB / "ncurve_summaries_qwen3e8b.npz", **arrays)


def analysis_step():
    rng = np.random.default_rng(5)
    S = np.load(EMB / "ncurve_summaries_qwen3e8b.npz")
    summaries = load_summaries()
    X = {c: unit(np.load(EMB / f).astype(np.float32)) for c, f in CORPUS_EMB.items() if (EMB / f).exists()}
    # shared radii: pooled 1-NN distances of 50-doc draws across corpora
    pooled = []
    for c, Xc in X.items():
        for _ in range(20):
            idx = rng.choice(Xc.shape[0], 50, replace=False)
            pooled.append(NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xc[idx]).kneighbors(Xc[idx])[0][:, 1])
    pooled = np.concatenate(pooled)
    radii = {q: float(np.quantile(pooled, q)) for q in R_QUANTILES}
    d_tw = {c: twonn_dimension(Xc[rng.choice(Xc.shape[0], min(2000, Xc.shape[0]), replace=False)]) for c, Xc in X.items()}

    rows, draw_rows = [], []
    for key, recs in summaries.items():
        c, n = key.rsplit("_n", 1)
        n = int(n)
        if c not in X or key not in S:
            continue
        E = unit(S[key].astype(np.float32))
        cen = E.mean(axis=0)
        cen /= np.linalg.norm(cen)
        iu = np.triu_indices(E.shape[0], k=1)
        instab = float((1.0 - E @ E.T)[iu].mean())
        Gq = {q: [] for q in R_QUANTILES}
        Mq = {q: [] for q in R_QUANTILES}
        for r_i, rec in enumerate(recs):
            idx = np.array(rec["idx"])
            Xs = X[c][idx]
            nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xs)
            d_s = nn.kneighbors(Xs)[0][:, 1]
            d_c = nn.kneighbors(X[c], n_neighbors=1)[0][:, 0]
            g = {q: float((d_s > r).mean()) for q, r in radii.items()}
            m = {q: float((d_c > r).mean()) for q, r in radii.items()}
            for q in R_QUANTILES:
                Gq[q].append(g[q]); Mq[q].append(m[q])
            draw_rows.append(dict(corpus=c, n=n, draw=rec["draw"], to_centroid=float(1.0 - E[r_i] @ cen),
                                  **{f"G_q{q}": g[q] for q in R_QUANTILES}))
        row = dict(corpus=c, n=n, N=X[c].shape[0], d_twonn=round(d_tw[c], 2), instability=round(instab, 4))
        for q in R_QUANTILES:
            row[f"G_q{q}"] = round(float(np.mean(Gq[q])), 4)
            row[f"M_q{q}"] = round(float(np.mean(Mq[q])), 4)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["corpus", "n"])
    dd = pd.DataFrame(draw_rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results/summary_n_curve.csv", index=False)
    dd.to_csv(ROOT / "results/summary_n_curve_draws.csv", index=False)
    print("radii:", radii)
    print(df.to_string(index=False))

    print("\nCross-corpus Spearman of instability with G_q0.5 and d_twonn at each n (bootstrap 90% band over summaries):")
    for n, g in df.groupby("n"):
        if len(g) < 4:
            continue
        for col in ["G_q0.5", "M_q0.5", "d_twonn"]:
            rho = spearmanr(g[col], g.instability).statistic
            boots = []
            for _ in range(BOOT):
                y = []
                for c in g.corpus:
                    E = unit(S[f"{c}_n{n}"].astype(np.float32))
                    bi = rng.choice(E.shape[0], E.shape[0], replace=True)
                    Eb = E[bi]
                    iu = np.triu_indices(Eb.shape[0], k=1)
                    y.append((1.0 - Eb @ Eb.T)[iu].mean())
                boots.append(spearmanr(g[col], y).statistic)
            lo, hi = np.nanpercentile(boots, [5, 95])
            print(f"  n={n:4d} {col:8s} rho={rho:+.2f} band=[{lo:+.2f}, {hi:+.2f}]  (k={len(g)})")

    print("\nWithin-corpus slope of log instability on log G_q0.5 across n (expect negative... G falls with n, so slope positive):")
    for c, g in df.groupby("corpus"):
        if len(g) >= 3:
            b = np.polyfit(np.log(g["G_q0.5"].clip(1e-3)), np.log(g.instability), 1)[0]
            rho = spearmanr(g["G_q0.5"], g.instability).statistic
            print(f"  {c:12s} slope={b:+.2f} rho={rho:+.2f} instability {g.instability.min():.3f}..{g.instability.max():.3f}")

    print("\nPer-draw Spearman(G_q0.5, distance to centroid), pooled within (corpus, n):")
    rhos = [spearmanr(g["G_q0.5"], g.to_centroid).statistic for _, g in dd.groupby(["corpus", "n"]) if len(g) >= 10]
    print(f"  median rho={np.nanmedian(rhos):+.2f}, share positive={np.mean(np.array(rhos) > 0):.2f}, k={len(rhos)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["embed", "analysis", "both"], default="both")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.step in ("embed", "both"):
        embed_step(a.device)
    if a.step in ("analysis", "both"):
        analysis_step()
