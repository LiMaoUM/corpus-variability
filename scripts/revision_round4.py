"""Fourth set of revision checks (ARS re-review of v3, 2026-10-14; DECISIONS.md 2026-10-14).

A. Pilot-only fit: for every corpus and encoder pair of collapse_v2, fit a and b on nested
   subsamples of the first 2,000-document pilot (n in 100..1,000, five draws), with d and the radius
   grid exactly as in collapse_v2 (same seed), and re-score the n* cases of results/nstar_v2.csv
   (n_fit = 1,000) against the same empirical crossings. Writes results/nstar_pilotfit.csv.
B. Bootstrap interval on r_same: resample the judged pairs within each distance bin (B = 2,000),
   recompute the crossing of P(same) = 0.5 under each judge, take the 5th and 95th percentiles,
   and carry them into n*(0.5) by n*(r') = n*(r) (r'/r)^(-d). Writes results/rsame_boot.csv.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from collapse_v2 import corpora, surface, fit_cells  # noqa: E402
from corpus_variability.collapse import twonn_dimension  # noqa: E402
from nstar_rsame import RCAL_NAME  # noqa: E402

PILOT_GRID = [100, 200, 350, 500, 750, 1000]
BINS = [(0, .1), (.1, .15), (.15, .2), (.2, .25), (.25, .3), (.3, .35), (.35, .4), (.4, .5), (.5, .65), (.65, 1)]
B = 2000


def pilot_fit():
    cases = pd.read_csv(ROOT / "results/nstar_v2.csv")
    cases = cases[cases.n_fit == 1000]
    out = []
    for name, X, metric, enc in corpora():
        cs = cases[cases.corpus == name]
        if cs.empty:
            continue
        N = X.shape[0]
        rng = np.random.default_rng(11)
        pilots = [X[rng.choice(N, 2000, replace=False)] for _ in range(5)]
        d = float(np.mean([twonn_dimension(p, metric) for p in pilots]))
        d1 = NearestNeighbors(n_neighbors=2, metric=metric).fit(pilots[0]).kneighbors(pilots[0])[0][:, 1]
        r_grid = np.geomspace(np.percentile(d1, 5), np.percentile(d1, 99), 12)
        Gp, _ = surface(pilots[0], PILOT_GRID, r_grid, metric, np.random.default_rng(29))
        a, b = fit_cells(Gp, PILOT_GRID, r_grid, d, 1000)
        for _, c in cs.iterrows():
            r = r_grid[np.argmin(np.abs(r_grid - c.r))]
            n_pred = (np.log(1 / c.eps) / np.exp(a)) ** (1 / b) / r ** d
            out.append(dict(corpus=name, encoder=enc, eps=c.eps, r=c.r, n_emp=c.n_emp, n_pred_full=c.n_pred,
                            n_pred_pilot=round(n_pred), ratio_full=c.ratio, ratio_pilot=round(n_pred / c.n_emp, 2)))
        g = pd.DataFrame([o for o in out if o["corpus"] == name])
        print(name, len(g), "within1.5 full", int(((g.ratio_full > 1 / 1.5) & (g.ratio_full < 1.5)).sum()),
              "pilot", int(((g.ratio_pilot > 1 / 1.5) & (g.ratio_pilot < 1.5)).sum()), flush=True)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results/nstar_pilotfit.csv", index=False)
    return df


def crossing_p(dist, same):
    mids, ps = [], []
    for lo, hi in BINS:
        m = (dist >= lo) & (dist < hi)
        if m.any():
            mids.append((lo + hi) / 2 if hi < 1 else .7); ps.append(same[m].mean())
    for k in range(len(ps) - 1):
        if ps[k] >= .5 > ps[k + 1]:
            return mids[k] + (ps[k] - .5) / (ps[k] - ps[k + 1]) * (mids[k + 1] - mids[k])
    return np.nan


def rsame_boot():
    base = pd.read_csv(ROOT / "results/nstar_rsame.csv")
    inv = {v: k for k, v in RCAL_NAME.items()}
    rng = np.random.default_rng(31)
    rows = []
    for f in sorted((ROOT / "data/rcal_gemma").glob("*.jsonl")):
        d = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()])
        bins = np.digitize(d.dist, [b[1] for b in BINS[:-1]])
        for judge, col in (("qwen", "label"), ("gemma", "label_gemma")):
            dist, same = d.dist.to_numpy(), (d[col] == "SAME").to_numpy()
            point = crossing_p(dist, same)
            bs = []
            for _ in range(B):
                idx = np.concatenate([rng.choice(np.flatnonzero(bins == k), (bins == k).sum()) for k in np.unique(bins)])
                bs.append(crossing_p(dist[idx], same[idx]))
            bs = np.array(bs)
            lo, hi = np.nanpercentile(bs, [5, 95]) if np.isfinite(bs).any() else (np.nan, np.nan)
            row = dict(corpus=f.stem, judge=judge, r_same=round(point, 3), r_lo=round(lo, 3), r_hi=round(hi, 3),
                       share_no_crossing=round(float(np.isnan(bs).mean()), 3))
            b = base[(base.corpus == inv.get(f.stem, "")) & (base.judge == judge)]
            if len(b) and np.isfinite(point):
                b = b.iloc[0]
                for tag, rr in (("lo", hi), ("hi", lo)):  # larger r, smaller n*
                    row[f"nstar_{tag}"] = float(f"{b['nstar_eps0.5'] * (rr / b.r_same) ** (-b.d):.3g}")
                row["nstar"] = b["nstar_eps0.5"]; row["N"] = b.N
            rows.append(row)
            print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/rsame_boot.csv", index=False)
    return df


if __name__ == "__main__":
    rb = rsame_boot()
    print("\n" + rb.to_string(index=False))
    pf = pilot_fit()
    print("\n" + pf.groupby("corpus").apply(lambda g: pd.Series(dict(
        cases=len(g), in15_full=((g.ratio_full > 2 / 3) & (g.ratio_full < 1.5)).sum(),
        in15_pilot=((g.ratio_pilot > 2 / 3) & (g.ratio_pilot < 1.5)).sum(),
        med_pilot=g.ratio_pilot.median()))).to_string())
