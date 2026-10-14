"""Half-cover prediction with d from one pilot (ARS re-review of v6, X1).

scripts/half_cover.py predicts the half-cover sample with d averaged over five pilots. A user has one
pilot, so this recomputes the prediction at every corpus, judge and radius end of
results/half_cover.csv with d, a and b all from the first pilot (half_cover.pilot_law_one), and
compares it with the measured value. Writes results/half_cover_onepilot.csv.
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
from collapse_v2 import corpora  # noqa: E402
from half_cover import pilot_law_one  # noqa: E402


def main():
    hc = pd.read_csv(ROOT / "results/half_cover.csv")
    out = []
    for name, X, metric, _ in corpora():
        g = hc[hc.corpus == name]
        if g.empty:
            continue
        N = X.shape[0]
        a, b, d = pilot_law_one(X, metric)
        n_fine = np.geomspace(10, N / 2, 4000)
        for _, row in g.iterrows():
            G = np.exp(-np.exp(a + b * np.log(n_fine * row.r ** d)))
            hit = np.flatnonzero(G * (1 - n_fine / N) <= 0.5)
            n1 = float(n_fine[hit[0]]) if len(hit) else N / 2
            out.append(dict(corpus=name, judge=row.judge, end=row.end, r=row.r, d_one=round(d, 2), d_five=row.d,
                            n_half=row.n_half, below_n=row.below_n, n_pred_five=row.n_pred, n_pred_one=round(n1),
                            share=row.share, share_pred_five=row.share_pred, share_pred_one=round(n1 / N, 3)))
        print(name, round(d, 2), "vs", g.d.iloc[0], flush=True)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results/half_cover_onepilot.csv", index=False)
    print(df[df.end == "point"].to_string(index=False))


if __name__ == "__main__":
    main()
