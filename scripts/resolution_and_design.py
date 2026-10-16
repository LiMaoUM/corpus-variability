"""Two reframings of the coverage question (Mao, 2026-10-15).

1. Resolution reached by a sample of fixed size. For n in {15, 100, 1000} and 20 random draws, the
   distance r80(n) at which 80% of the corpus has a sampled document within r (true, against the whole
   corpus), the same distance estimated from the sample alone (the r at which G (1 - n/N) = 0.2), and what
   that distance means in content: the proportion of judged pairs (Qwen3.8) at that distance that are the
   same content, and the proportion that are the same content or the same topic.
2. Designed samples against random ones. With the whole corpus embedded, choose the documents to read:
   (a) lazy greedy maximum coverage at r (each pick covers the most documents not yet covered), which is
   within 1 - 1/e of the best possible sample and serves as the bound on what any design can do;
   (b) one document per k-means cluster (the document closest to each centroid), a design a researcher
   would use. Missing mass at n = 15, 100, 1000 at r_same (Qwen3.8) and r_topic, and for greedy the
   sample size at which half of the corpus is covered.
Writes results/resolution_reached.csv and results/designed_samples.csv.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collapse_v2 import corpora  # noqa: E402
from nstar_rsame import RCAL_NAME  # noqa: E402
from revision_round4 import BINS, crossing_p  # noqa: E402

RCAL_NAME = {**RCAL_NAME, "bsky_qwen3": "bsky"}
NS = [15, 100, 1000]
DRAWS = 20
BLOCK = 2048


def judged_curve(c):
    d = pd.DataFrame([json.loads(l) for l in (ROOT / f"data/rcal/{c}.jsonl").read_text().splitlines() if l.strip()])
    mids, ps, pt = [], [], []
    for lo, hi in BINS:
        m = (d.dist >= lo) & (d.dist < hi)
        if m.any():
            mids.append((lo + hi) / 2 if hi < 1 else .7)
            ps.append((d.label[m] == "SAME").mean())
            pt.append(d.label[m].isin(["SAME", "TOPIC"]).mean())
    r_topic = crossing_p(d.dist.to_numpy(), d.label.isin(["SAME", "TOPIC"]).to_numpy())
    return np.array(mids), np.array(ps), np.array(pt), r_topic


def nearest_to_set(X, idx):
    """Cosine distance from every corpus document to its nearest document in X[idx]."""
    out = np.empty(X.shape[0], np.float32)
    S = X[idx]
    for s in range(0, X.shape[0], 8192):
        out[s:s + 8192] = 1 - (X[s:s + 8192] @ S.T).max(axis=1)
    return np.maximum(out, 0)


def resolution_reached(name, X, mids, ps, pt, rng):
    N = X.shape[0]
    rows = []
    for n in NS:
        r_true, r_est = [], []
        for _ in range(DRAWS):
            idx = rng.choice(N, n, replace=False)
            r_true.append(np.quantile(nearest_to_set(X, idx), 0.8))
            Sn = X[idx] @ X[idx].T
            np.fill_diagonal(Sn, -np.inf)
            nn = 1 - Sn.max(axis=1)
            q = 1 - 0.2 / (1 - n / N)  # G (1 - n/N) = 0.2  <=>  G = 0.2 / (1 - n/N)
            r_est.append(np.quantile(nn, min(max(q, 0), 1)))
        rt, re_ = float(np.mean(r_true)), float(np.mean(r_est))
        rows.append(dict(corpus=name, N=N, n=n, r80_true=round(rt, 3), r80_true_p10=round(np.percentile(r_true, 10), 3),
                         r80_true_p90=round(np.percentile(r_true, 90), 3), r80_est=round(re_, 3),
                         p_same_at_r80=round(float(np.interp(rt, mids, ps)), 3),
                         p_topic_at_r80=round(float(np.interp(rt, mids, pt)), 3)))
        print(rows[-1], flush=True)
    return rows


def neighbour_lists(X, r):
    """CSR lists of the documents within cosine distance r of each document (itself included)."""
    N = X.shape[0]
    thr = 1 - r
    indptr, cols = [0], []
    for s in range(0, N, BLOCK):
        rr, cc = np.nonzero((X[s:s + BLOCK] @ X.T) >= thr)
        counts = np.bincount(rr, minlength=min(BLOCK, N - s))
        indptr.extend((indptr[-1] + np.cumsum(counts)).tolist())
        cols.append(cc.astype(np.int32))
    return np.array(indptr, np.int64), np.concatenate(cols)


def greedy_cover(X, r, checkpoints, stop_at):
    """Greedy maximum coverage at radius r with exact gains. Returns missing mass at each checkpoint and n to cover stop_at."""
    N = X.shape[0]
    indptr, cols = neighbour_lists(X, r)
    gain = np.diff(indptr).astype(np.int64)  # uncovered documents each candidate would cover
    covered = np.zeros(N, bool)
    ncov, picks, mm, n_stop = 0, 0, {}, None
    cps = sorted(checkpoints)
    while True:
        j = int(np.argmax(gain))
        g = int(gain[j])
        if g <= 1:
            break  # every further pick covers exactly one document (itself)
        nb = cols[indptr[j]:indptr[j + 1]]
        new = nb[~covered[nb]]
        covered[new] = True
        ncov += len(new)
        picks += 1
        # each newly covered document no longer counts toward the gain of any of its neighbours
        touched = np.concatenate([cols[indptr[u]:indptr[u + 1]] for u in new])
        gain -= np.bincount(touched, minlength=N)
        for c in cps:
            if picks == c:
                mm[c] = 1 - ncov / N
        if n_stop is None and ncov >= stop_at:
            n_stop = picks
        if n_stop is not None and picks >= max(cps):
            break
    for c in cps:
        if c not in mm:  # reached after gains fell to one per pick
            mm[c] = max(0.0, 1 - (ncov + (c - picks)) / N)
    if n_stop is None:
        n_stop = picks + max(0, int(np.ceil(stop_at - ncov)))
    return mm, n_stop


def kmeans_design(X, n, seed=0):
    km = MiniBatchKMeans(n_clusters=n, random_state=seed, batch_size=max(4096, 3 * n), n_init=3).fit(X)
    C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    picks = set()
    for k in range(n):
        members = np.flatnonzero(km.labels_ == k)
        if len(members):
            picks.add(int(members[np.argmax(X[members] @ C[k])]))
    return np.array(sorted(picks))


def main():
    rb = pd.read_csv(ROOT / "results/rsame_boot.csv").set_index(["corpus", "judge"])
    res_rows, des_rows = [], []
    for name, X, metric, _ in corpora():
        if name not in RCAL_NAME:
            continue
        c = RCAL_NAME[name]
        N = X.shape[0]
        mids, ps, pt, r_topic = judged_curve(c)
        rng = np.random.default_rng(53)
        res_rows += resolution_reached(name, X, mids, ps, pt, rng)
        pd.DataFrame(res_rows).to_csv(ROOT / "results/resolution_reached.csv", index=False)
        radii = {"same": rb.loc[(c, "qwen"), "r_same"], "topic": r_topic}
        km_idx = {n: kmeans_design(X, n) for n in NS}
        for level, r in radii.items():
            if not np.isfinite(r):
                continue
            rand = {n: float(np.mean([(nearest_to_set(X, rng.choice(N, n, replace=False)) > r).mean()
                                      for _ in range(DRAWS)])) for n in NS}
            km = {n: float((nearest_to_set(X, km_idx[n]) > r).mean()) for n in NS}
            gm, n_half = greedy_cover(X, r, NS, N / 2)
            for n in NS:
                des_rows.append(dict(corpus=name, N=N, level=level, r=round(float(r), 3), n=n,
                                     M_random=round(rand[n], 3), M_kmeans=round(km[n], 3), M_greedy=round(gm[n], 3),
                                     n_half_greedy=n_half))
                print(des_rows[-1], flush=True)
            pd.DataFrame(des_rows).to_csv(ROOT / "results/designed_samples.csv", index=False)


if __name__ == "__main__":
    main()
