"""Draw the frozen 50k-post samples of Truth Social and Bluesky (card:
experiments/ts-bsky-embedding.card.md) with DuckDB over the Layer 2 parquet partitions.

Filter: created_at in 2024-05-01..2024-12-31, lang = 'en', cleaned text >= 20 chars, dedupe on
text; uniform random sample of N_SAMPLE per platform with a fixed seed. Writes
data/emb/{ts,bsky}_texts.parquet with columns post_id, author_id, created_at, text.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/emb"
SRC = {
    "ts": "/home/data/corpus-parquet/truthsocial/trump/month=2024-*/part-*.parquet",
    "bsky": "/home/data/corpus-parquet/bsky/trump_flattened/month=2024-*/part-*.parquet",
}
N_SAMPLE = 50_000
SEED = 2026
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+")


def clean(t: str) -> str:
    return WS.sub(" ", MENTION.sub("", URL.sub("", str(t)))).strip()


def main(limit: int | None):
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SELECT setseed({(SEED % 1000) / 1000.0})")
    for name, glob in SRC.items():
        n = limit or N_SAMPLE
        # oversample 3x before cleaning and dedupe, then cut to n
        q = f"""
            SELECT post_id, author_id, created_at, text
            FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE created_at >= TIMESTAMP '2024-05-01' AND created_at < TIMESTAMP '2025-01-01'
              AND lang = 'en' AND text IS NOT NULL AND length(text) >= 20
            USING SAMPLE reservoir({3 * n} ROWS) REPEATABLE ({SEED})
        """
        df = con.execute(q).df()
        df["text"] = df["text"].map(clean)
        df = df[df.text.str.len() >= 20].drop_duplicates("text")
        df = df.sample(n=min(n, len(df)), random_state=SEED).reset_index(drop=True)
        df["post_id"] = df["post_id"].astype(str)
        df["author_id"] = df["author_id"].astype(str)
        df.to_parquet(OUT / f"{name}_texts.parquet", index=False)
        print(f"{name}: {len(df)} posts, {df.created_at.min()} .. {df.created_at.max()}, "
              f"median chars {int(df.text.str.len().median())}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    main(p.parse_args().limit)
