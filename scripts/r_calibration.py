"""Calibrate the resolution r against LLM judgments of document pairs (card:
experiments/r-calibration.card.md).

For each corpus, sample PAIRS_PER_BIN document pairs in each cosine-distance bin (near pairs from
nearest neighbour search, far pairs at random), ask Gemma-4 whether the two documents are about the
same specific content, the same general topic, or unrelated, and record the share of each answer
per bin. Output data/rcal/<corpus>.jsonl and results/r_calibration.csv.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summary_n_curve import unit  # noqa: E402
from summarize_draws import BASE_URL, MODEL  # noqa: E402

EMB = ROOT / "data/emb"
OUT = ROOT / "data/rcal"
BINS = [(0.00, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.50), (0.50, 0.65), (0.65, 1.00)]
PAIRS_PER_BIN = 40
CORPORA = {"random": ("random_qwen3e8b.npy", "random_texts.parquet"), "citizens": ("citizens_qwen3e8b.npy", "citizens_texts.parquet"),
           "trump": ("trump_qwen3e8b.npy", "trump_texts.parquet"), "covid": ("covid_qwen3e8b.npy", "covid_texts.parquet"),
           "illegal": ("illegal_qwen3e8b.npy", "illegal_texts.parquet"),
           "sv_askdocs": ("summeval_askdocs_qwen3e8b.npy", "summeval_askdocs_texts.parquet"),
           "sv_trump": ("summeval_trump_qwen3e8b.npy", "summeval_trump_texts.parquet"),
           "sv_tifu": ("summeval_tifu_qwen3e8b.npy", "summeval_tifu_texts.parquet"),
           "ts": ("ts_qwen3e8b.npy", "ts_texts.parquet"), "bsky": ("bsky_qwen3e8b.npy", "bsky_texts.parquet")}
PROMPT = (
    "Two documents from the same collection are shown. Judge their relation.\n"
    "SAME: they are about the same specific content (the same claim, question, event or story), so a reader who has "
    "read one would learn little new from the other.\n"
    "TOPIC: they are about the same general topic but different specific content.\n"
    "UNRELATED: they are about different topics.\n\n"
    "Document A: {a}\n\nDocument B: {b}\n\nAnswer with one word, SAME, TOPIC or UNRELATED."
)


def sample_pairs(X, rng):
    N = X.shape[0]
    pairs = {b: [] for b in BINS}
    # near pairs from the 20 nearest neighbours of 3000 random anchors
    anchors = rng.choice(N, min(3000, N), replace=False)
    dist, ind = NearestNeighbors(n_neighbors=21, metric="cosine").fit(X).kneighbors(X[anchors])
    for a, drow, irow in zip(anchors, dist[:, 1:], ind[:, 1:]):
        for dd, j in zip(drow, irow):
            for b in BINS:
                if b[0] <= dd < b[1] and len(pairs[b]) < 4 * PAIRS_PER_BIN:
                    pairs[b].append((int(a), int(j), float(dd)))
    # far pairs at random
    for _ in range(20000):
        i, j = rng.integers(N, size=2)
        if i == j:
            continue
        dd = float(1 - X[i] @ X[j])
        for b in BINS:
            if b[0] <= dd < b[1] and len(pairs[b]) < 4 * PAIRS_PER_BIN:
                pairs[b].append((int(i), int(j), dd))
    out = []
    for b, lst in pairs.items():
        if lst:
            pick = rng.choice(len(lst), min(PAIRS_PER_BIN, len(lst)), replace=False)
            out += [lst[p] for p in pick]
    return out


async def judge(client, sem, prompt):
    async with sem:
        r = await client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=5,
                                                 messages=[{"role": "user", "content": prompt}])
    t = (r.choices[0].message.content or "").strip().upper()
    for k in ("UNRELATED", "SAME", "TOPIC"):
        if k in t:
            return k
    return "OTHER"


async def run(corpora, limit, concurrency):
    client = AsyncOpenAI(base_url=BASE_URL, api_key="local", timeout=3600, max_retries=5)
    sem = asyncio.Semaphore(concurrency)
    OUT.mkdir(parents=True, exist_ok=True)
    for c in corpora:
        e, t = CORPORA[c]
        X = unit(np.load(EMB / e).astype(np.float32))
        texts = pd.read_parquet(EMB / t).text.tolist()
        rng = np.random.default_rng(5)
        pairs = sample_pairs(X, rng)
        if limit:
            pairs = pairs[:limit]
        outs = await asyncio.gather(*(judge(client, sem, PROMPT.format(a=texts[i][:800], b=texts[j][:800])) for i, j, _ in pairs))
        with (OUT / f"{c}.jsonl").open("w") as fh:
            for (i, j, dd), o in zip(pairs, outs):
                fh.write(json.dumps(dict(i=i, j=j, dist=dd, label=o)) + "\n")
        print(c, len(pairs), "pairs;", pd.Series(outs).value_counts().to_dict(), flush=True)


def analysis():
    rows = []
    for f in sorted(OUT.glob("*.jsonl")):
        df = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()])
        for b in BINS:
            g = df[(df.dist >= b[0]) & (df.dist < b[1])]
            if len(g):
                rows.append(dict(corpus=f.stem, bin_lo=b[0], bin_hi=b[1], n=len(g), p_same=round((g.label == "SAME").mean(), 3),
                                 p_same_or_topic=round(g.label.isin(["SAME", "TOPIC"]).mean(), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/r_calibration.csv", index=False)
    print(out.pivot_table(index=["bin_lo", "bin_hi"], columns="corpus", values="p_same").round(2).to_string())
    print("\nP(same or topic):")
    print(out.pivot_table(index=["bin_lo", "bin_hi"], columns="corpus", values="p_same_or_topic").round(2).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--corpora", nargs="*", default=list(CORPORA))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--analysis", action="store_true")
    a = p.parse_args()
    if a.analysis:
        analysis()
    elif a.smoke:
        asyncio.run(run(["citizens"], 40, a.concurrency))
    else:
        asyncio.run(run(a.corpora, None, a.concurrency))
        analysis()
