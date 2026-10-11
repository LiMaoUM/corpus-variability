"""Theme coverage against inventory design (DECISIONS.md 2026-10-11, group 2).

For each of the eight corpora and each inventory (k-means K = 10, 30, 100; a skewed inventory
from k-means++ seeding without refinement at K = 30; a skewed inventory from the K = 100 clusters
merged into 30 groups by a Zipf schedule), compute: the effective number of themes exp(H) of the
mass vector; label coverage of the existing 1,200 draws; the expected coverage from the masses;
and the spread of label coverage across corpora at each n. Writes results/theme_k_sensitivity.csv.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summary_n_curve import CORPUS_EMB, EMB, SUMM, unit  # noqa: E402


def inventories(X, rng):
    out = {}
    for K in (10, 30, 100):
        out[f"kmeans{K}"] = MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=4096).fit_predict(X)
    cents, _ = kmeans_plusplus(X, 30, random_state=0)
    out["kpp30_noref"] = np.argmax(X @ cents.T, axis=1)
    lab100 = out["kmeans100"]
    sizes = np.bincount(lab100, minlength=100)
    order = np.argsort(-sizes)
    # Zipf schedule: group sizes proportional to 1/rank over 30 groups
    w = 1 / np.arange(1, 31); w = w / w.sum()
    counts = np.maximum(1, np.round(w * 100).astype(int)); counts[0] += 100 - counts.sum()
    merged = np.empty(100, int); pos = 0
    for g, cnt in enumerate(counts):
        merged[order[pos:pos + cnt]] = g; pos += cnt
    out["zipf30_from100"] = merged[lab100]
    return out


def main():
    rows = []
    for c in CORPUS_EMB:
        X = unit(np.load(EMB / CORPUS_EMB[c]).astype(np.float32))
        rng = np.random.default_rng(0)
        invs = inventories(X, rng)
        for inv, labels in invs.items():
            K = labels.max() + 1
            m = np.bincount(labels, minlength=K) / labels.size
            H = -(m[m > 0] * np.log(m[m > 0])).sum()
            for n in (25, 50, 100, 200, 400):
                f = SUMM / f"{c}_n{n}.jsonl"
                covs = []
                for rec in (json.loads(l) for l in f.read_text().splitlines() if l.strip()):
                    seen = np.unique(labels[np.array(rec["idx"])])
                    covs.append(m[seen].sum())
                rows.append(dict(corpus=c, inventory=inv, K=K, eff_themes=round(float(np.exp(H)), 1),
                                 n=n, label_cov=round(float(np.mean(covs)), 4), label_cov_sd=round(float(np.std(covs)), 4),
                                 expected=round(float((m * (1 - (1 - m) ** n)).sum()), 4)))
        print(c, {k: round(float(np.exp(-(lambda mm: (mm[mm > 0] * np.log(mm[mm > 0])).sum())(np.bincount(v) / v.size))), 1) for k, v in invs.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/theme_k_sensitivity.csv", index=False)
    print("\nMAE of expected vs observed label coverage, all cells: %.4f" % (df.expected - df.label_cov).abs().mean())
    print("\nCross-corpus spread of label coverage (min, max, sd) by inventory and n:")
    for (inv, n), g in df.groupby(["inventory", "n"]):
        print(f"  {inv:16s} n={n:4d}  {g.label_cov.min():.3f} {g.label_cov.max():.3f} sd={g.label_cov.std():.3f}  eff themes {g.eff_themes.min():.1f}..{g.eff_themes.max():.1f}")


if __name__ == "__main__":
    main()
