"""Summarise repeated n-document draws from each corpus with Gemma-4-31B-it on the local vLLM
server (card: experiments/summary-n-curve.card.md).

For each corpus and n in N_GRID, N_DRAWS draws without replacement (seed 2026 + draw index),
one summary per draw at temperature 0. Output: data/summaries/<corpus>_n<n>.jsonl with fields
draw, idx (row indices into the corpus parquet/embedding), summary, prompt_chars. Existing
rows are skipped, so the script can be resumed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data/emb"
OUT = ROOT / "data/summaries"
BASE_URL = "http://127.0.0.1:8800/v1"
MODEL = "google/gemma-4-31B-it"
N_GRID = [25, 50, 100, 200, 400]
N_DRAWS = 30
DOC_CHARS = 600
CORPORA = {
    "random": "random_texts.parquet", "citizens": "citizens_texts.parquet",
    "trump": "trump_texts.parquet", "covid": "covid_texts.parquet", "illegal": "illegal_texts.parquet",
    "sv_askdocs": "summeval_askdocs_texts.parquet", "sv_trump": "summeval_trump_texts.parquet",
    "sv_tifu": "summeval_tifu_texts.parquet",
}
PROMPT = (
    "Below are {n} documents sampled from a larger collection. Write a summary of the collection "
    "in one paragraph of at most 150 words. Cover the main themes in proportion to how often they "
    "appear, name notable minority themes briefly, and do not mention individual document numbers.\n\n"
    "{docs}\n\nSummary:"
)


def build_prompt(texts: list[str]) -> str:
    docs = "\n".join(f"[{i + 1}] {t[:DOC_CHARS]}" for i, t in enumerate(texts))
    return PROMPT.format(n=len(texts), docs=docs)


async def one(client, sem, prompt):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, temperature=0.0, max_tokens=300,
                    messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content.strip()
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                await asyncio.sleep(2 * (attempt + 1))


async def run(corpora, n_grid, n_draws, concurrency):
    client = AsyncOpenAI(base_url=BASE_URL, api_key="local", timeout=600)
    sem = asyncio.Semaphore(concurrency)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in corpora:
        df = pd.read_parquet(EMB / CORPORA[name])
        texts = df.text.tolist()
        for n in n_grid:
            f = OUT / f"{name}_n{n}.jsonl"
            done = set()
            if f.exists():
                done = {json.loads(l)["draw"] for l in f.read_text().splitlines() if l.strip()}
            jobs = []
            for d in range(n_draws):
                if d in done:
                    continue
                rng = np.random.default_rng(2026 + d)
                idx = rng.choice(len(texts), n, replace=False)
                prompt = build_prompt([texts[i] for i in idx])
                jobs.append((d, idx, prompt))
            if not jobs:
                continue
            outs = await asyncio.gather(*(one(client, sem, p) for _, _, p in jobs))
            with f.open("a") as fh:
                for (d, idx, prompt), s in zip(jobs, outs):
                    fh.write(json.dumps(dict(draw=d, idx=idx.tolist(), summary=s, prompt_chars=len(prompt))) + "\n")
            print(f"{name} n={n}: {len(jobs)} summaries, prompt chars median {int(np.median([len(p) for *_, p in jobs]))}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--corpora", nargs="*", default=list(CORPORA))
    p.add_argument("--concurrency", type=int, default=8)
    a = p.parse_args()
    if a.smoke:
        OUT = ROOT / "data/summaries_smoke"
        asyncio.run(run(["random"], [25, 400], 2, a.concurrency))
    else:
        asyncio.run(run(a.corpora, N_GRID, N_DRAWS, a.concurrency))
