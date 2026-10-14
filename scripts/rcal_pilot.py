"""Calibrate r_same inside one 2,000-document pilot (ARS re-review of v5, W3; DECISIONS.md 2026-10-14).

The paper's r_same comes from pairs found by nearest neighbour search over the whole corpus. A user
has only the pilot. For three corpora, take the first pilot of scripts/half_cover.py (seed 11), draw
pairs inside it with the same bins and sampler as scripts/r_calibration.py, judge them with
Gemma-4-31B-it (shared server on port 8801, concurrency 8) with the same prompt, and compute r_same with the same
interpolation and pair bootstrap as scripts/revision_round4.py. Then run the whole procedure from the
pilot alone: predict the half-cover share at the pilot's r_same with the law fitted inside the pilot,
and compare it with the share measured on the whole corpus at the full-corpus Gemma r_same.
Writes data/rcal_pilot/<corpus>.jsonl and results/rcal_pilot.csv.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from r_calibration import CORPORA, PROMPT, EMB, BINS, sample_pairs  # noqa: E402
from revision_round4 import crossing_p, B  # noqa: E402
from half_cover import pilot_law, pilot_law_one  # noqa: E402
from summary_n_curve import unit  # noqa: E402

URL, MODEL = "http://127.0.0.1:8801/v1", "google/gemma-4-31B-it"
OUT = ROOT / "data/rcal_pilot"
RUN = {"random": "census_random_qwen3", "ts": "ts_qwen3", "sv_trump": "trump_qwen3",
       "sv_askdocs": "askdocs_qwen3", "sv_tifu": "tifu_qwen3"}  # last two added after the v6 re-review (X1)


async def judge(client, sem, prompt):
    async with sem:
        r = await client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=5, messages=[{"role": "user", "content": prompt}])
    t = (r.choices[0].message.content or "").upper()
    return next((k for k in ("UNRELATED", "SAME", "TOPIC") if k in t), "OTHER")


async def label(c, X, texts, pilot):
    f = OUT / f"{c}.jsonl"
    if f.exists():
        return pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()])
    pairs = [(int(pilot[i]), int(pilot[j]), d) for i, j, d in sample_pairs(X[pilot], np.random.default_rng(5))]
    client = AsyncOpenAI(base_url=URL, api_key="local", timeout=7200, max_retries=8)
    sem = asyncio.Semaphore(8)
    outs = await asyncio.gather(*(judge(client, sem, PROMPT.format(a=texts[i][:800], b=texts[j][:800])) for i, j, _ in pairs))
    OUT.mkdir(exist_ok=True)
    with f.open("w") as fh:
        for (i, j, d), o in zip(pairs, outs):
            fh.write(json.dumps(dict(i=i, j=j, dist=d, label_gemma=o)) + "\n")
    return pd.DataFrame([dict(i=i, j=j, dist=d, label_gemma=o) for (i, j, d), o in zip(pairs, outs)])


def main():
    hc = pd.read_csv(ROOT / "results/half_cover.csv")
    rows = []
    for c, name in RUN.items():
        e, t = CORPORA[c]
        X = unit(np.load(EMB / e).astype(np.float32))
        texts = pd.read_parquet(EMB / t).text.tolist()
        N = X.shape[0]
        pilot = np.random.default_rng(11).choice(N, 2000, replace=False)
        d = asyncio.run(label(c, X, texts, pilot))
        dist, same = d.dist.to_numpy(), (d.label_gemma == "SAME").to_numpy()
        r_pt = crossing_p(dist, same)
        bins = np.digitize(dist, [b[1] for b in BINS[:-1]])
        rng = np.random.default_rng(31)
        bs = np.array([crossing_p(dist[idx], same[idx]) for idx in
                       (np.concatenate([rng.choice(np.flatnonzero(bins == k), (bins == k).sum()) for k in np.unique(bins)])
                        for _ in range(B))])
        lo, hi = np.nanpercentile(bs, [5, 95]) if np.isfinite(bs).any() else (np.nan, np.nan)
        n_fine = np.geomspace(10, N / 2, 4000)

        def predict(law):
            a, b, dim = law
            if not np.isfinite(r_pt):
                return np.nan
            G = np.exp(-np.exp(a + b * np.log(n_fine * r_pt ** dim)))
            hit = np.flatnonzero(G * (1 - n_fine / N) <= 0.5)
            return float(n_fine[hit[0]]) if len(hit) else N / 2
        # no crossing inside the pilot: bound r_same by the lower edge of the closest non-empty bin
        r_bound = np.nan if np.isfinite(r_pt) else BINS[int(bins.min())][0] if (bins == bins.min()).any() else np.nan
        if not np.isfinite(r_pt):
            r_pt = r_bound
        n_pred = predict(pilot_law(X, "cosine"))
        n_pred1 = predict(pilot_law_one(X, "cosine"))
        full = hc[(hc.corpus == name) & (hc.judge == "gemma") & (hc.end == "point")].iloc[0]
        rows.append(dict(corpus=c, pairs=len(d), r_bound=r_bound, pairs_by_bin=np.bincount(bins, minlength=len(BINS)).tolist(),
                         r_same_pilot=round(r_pt, 3), r_lo=round(lo, 3), r_hi=round(hi, 3),
                         no_crossing=round(float(np.isnan(bs).mean()), 3), r_same_full=full.r,
                         share_pred_pilot=round(n_pred / N, 3), n_pred_pilot=round(n_pred) if np.isfinite(n_pred) else np.nan,
                         share_pred_one=round(n_pred1 / N, 3), n_pred_one=round(n_pred1) if np.isfinite(n_pred1) else np.nan,
                         share_meas_full=full.share, n_meas_full=full.n_half))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/rcal_pilot.csv", index=False)


if __name__ == "__main__":
    main()
