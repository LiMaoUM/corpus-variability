"""Smoke test: does NN isolation coverage track true coverage on a synthetic corpus?

For each seed and corpus setting, draw nested samples of increasing n without replacement and
record true coverage, Good-Turing coverage (labels), NN coverage at several label-free radii,
and the Vendi score. At n = n_pilot, extrapolate the required n for a target coverage with the
Chao formula and compare it with the n at which true coverage first crosses the target.

Writes results/smoke_synthetic.csv and results/smoke_synthetic_required_n.csv and prints a
summary: per radius, Spearman rho and MAE against true coverage, and the required-n comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from corpus_variability.coverage import (  # noqa: E402
    chao_required_n,
    good_turing_coverage,
    nn_counts,
    pilot_radius,
    true_coverage,
    vendi_score,
)
from corpus_variability.synthetic import make_corpus  # noqa: E402

N_GRID = [50, 100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
SETTINGS = [
    # name, n_topics, zipf_a, noise
    ("k50_flat", 50, 0.5, 0.6),
    ("k200_zipf1", 200, 1.0, 0.6),
    ("k500_zipf1", 500, 1.0, 0.6),
    ("k200_zipf1_noisy", 200, 1.0, 1.0),
]


def main(n_docs: int, dim: int, seeds: int, n_pilot: int, target: float, out: Path):
    rows, req_rows = [], []
    for name, k, a, noise in SETTINGS:
        for seed in range(seeds):
            X, y = make_corpus(n_docs, k, dim, a, noise, seed)
            rng = np.random.default_rng(1000 + seed)
            order = rng.permutation(n_docs)
            radii = {al: pilot_radius(X, al, rng=rng) for al in ALPHAS}
            first_cross = None
            for n in N_GRID:
                idx = order[:n]
                Xs, ys = X[idx], y[idx]
                tc = true_coverage(y, ys)
                if first_cross is None and tc >= target:
                    first_cross = n
                row = dict(setting=name, seed=seed, n=n, true_cov=tc,
                           gt_cov=good_turing_coverage(ys),
                           vendi=vendi_score(Xs) if n <= 2000 else np.nan)
                for al, r in radii.items():
                    f1, f2 = nn_counts(Xs, r)
                    row[f"nn_cov_a{al}"] = 1.0 - f1 / n
                    if n == n_pilot:
                        req_rows.append(dict(setting=name, seed=seed, alpha=al, r=r, f1=f1, f2=f2,
                                             chao_required_n=chao_required_n(n, f1, f2, target)))
                rows.append(row)
            for rr in req_rows:
                if rr["setting"] == name and rr["seed"] == seed:
                    rr["empirical_required_n"] = first_cross if first_cross is not None else np.inf
    df = pd.DataFrame(rows)
    req = pd.DataFrame(req_rows)
    out.mkdir(exist_ok=True)
    df.to_csv(out / "smoke_synthetic.csv", index=False)
    req.to_csv(out / "smoke_synthetic_required_n.csv", index=False)

    print(f"n_docs={n_docs} dim={dim} seeds={seeds} pilot n={n_pilot} target={target}\n")
    print("Agreement with true coverage across all (setting, seed, n):")
    for col in ["gt_cov"] + [f"nn_cov_a{al}" for al in ALPHAS]:
        rho = spearmanr(df[col], df["true_cov"]).statistic
        mae = (df[col] - df["true_cov"]).abs().mean()
        print(f"  {col:14s} rho={rho:.3f}  MAE={mae:.3f}")
    print("\nPer-setting MAE by radius:")
    print(df.groupby("setting")[[f"nn_cov_a{al}" for al in ALPHAS] + ["true_cov"]]
            .apply(lambda g: pd.Series({c: (g[c] - g["true_cov"]).abs().mean() for c in g.columns if c != "true_cov"}))
            .round(3).to_string())
    print(f"\nRequired n for coverage {target} (Chao extrapolation at n={n_pilot} vs empirical):")
    summ = (req.replace(np.inf, np.nan)
               .groupby(["setting", "alpha"])[["chao_required_n", "empirical_required_n"]]
               .median().round(0))
    print(summ.to_string())
    print("\nTrue coverage by n (median over seeds):")
    print(df.pivot_table(index="n", columns="setting", values="true_cov", aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-docs", type=int, default=20000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-pilot", type=int, default=500)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    a = p.parse_args()
    main(a.n_docs, a.dim, a.seeds, a.n_pilot, a.target, a.out)
