"""What a reader of n random documents has seen (writing revision, 2026-10-14, writing-lessons entry 41).

For each Qwen3 corpus with a calibrated radius r_same (both judges), and n in a few sizes a reader
might actually read, draw 20 random samples and record:
- M, the true missing mass: the share of corpus documents with no sampled document within r_same,
  computed against the whole corpus;
- also at the topic resolution (judge "qwen_topic"), where half of the Qwen3.8 judgments are SAME or TOPIC;
- G, the isolation rate from the sample alone, and G (1 - n/N).
Writes results/reader_view.csv (means over draws, with the 10th and 90th percentiles of M).
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
from nstar_rsame import RCAL_NAME  # noqa: E402
from revision_round4 import crossing_p  # noqa: E402
import json

NS = [15, 50, 100, 500, 1000]
DRAWS = 20


def main():
    rb = pd.read_csv(ROOT / "results/rsame_boot.csv").set_index(["corpus", "judge"])
    # topic resolution: where half of the judged pairs (Qwen3.8) are the same content or the same topic
    r_topic = {}
    for c in set(RCAL_NAME.values()):
        d = pd.DataFrame([json.loads(l) for l in (ROOT / f"data/rcal/{c}.jsonl").read_text().splitlines() if l.strip()])
        r_topic[c] = crossing_p(d.dist.to_numpy(), d.label.isin(["SAME", "TOPIC"]).to_numpy())
    rows = []
    for name, X, metric, _ in corpora():
        if name not in RCAL_NAME:
            continue
        N = X.shape[0]
        rng = np.random.default_rng(41)
        for judge in ("qwen", "gemma", "qwen_topic"):
            r = r_topic[RCAL_NAME[name]] if judge == "qwen_topic" else rb.loc[(RCAL_NAME[name], judge), "r_same"]
            if not np.isfinite(r):
                continue
            for n in NS:
                Ms, Gs = [], []
                for _ in range(DRAWS):
                    idx = rng.choice(N, n, replace=False)
                    S = X @ X[idx].T  # cosine similarity of every corpus document to the sample
                    Ms.append(float((1 - S.max(axis=1) > r).mean()))
                    Sn = S[idx]
                    np.fill_diagonal(Sn, -np.inf)
                    Gs.append(float((1 - Sn.max(axis=1) > r).mean()))
                rows.append(dict(corpus=name, judge=judge, r=r, N=N, n=n, M=round(np.mean(Ms), 3),
                                 M_p10=round(np.percentile(Ms, 10), 3), M_p90=round(np.percentile(Ms, 90), 3),
                                 G=round(np.mean(Gs), 3), G_fpc=round(np.mean(Gs) * (1 - n / N), 3)))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/reader_view.csv", index=False)


if __name__ == "__main__":
    main()
