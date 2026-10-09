"""Embed the five 2024 census tweet corpora and their 100 repeated-sample summaries.

Model: Qwen/Qwen3-Embedding-8B (Apache 2.0, MTEB English v2 75.22, hub-checked 2026-10-09),
documents encoded without a prompt, L2-normalised, bf16, truncated at 256 tokens.

Inputs (from-windows-2026-07-26/data/): random.csv, full_citizenship_question.csv (Sprinklr
exports, column Message), trump.csv, covid.csv, illegal.csv (columns UniversalMessageId,
Message, Day); summaries in randoms_summaries.csv, citizens_summaries.csv,
trump_sample_summaries.csv, covid_sample_summaries.csv, illegal_sample_summaries.csv.

Outputs (data/emb/, gitignored): <corpus>_qwen3e8b.npy (float16, N x 4096),
<corpus>_texts.parquet (deduped, cleaned text with original ids), <corpus>_summaries_qwen3e8b.npy.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "from-windows-2026-07-26/data"
OUT = ROOT / "data/emb"
MODEL = "Qwen/Qwen3-Embedding-8B"

CORPORA = {
    "random": ("random.csv", "randoms_summaries.csv"),
    "citizens": ("full_citizenship_question.csv", "citizens_summaries.csv"),
    "trump": ("trump.csv", "trump_sample_summaries.csv"),
    "covid": ("covid.csv", "covid_sample_summaries.csv"),
    "illegal": ("illegal.csv", "illegal_sample_summaries.csv"),
}

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+")


def clean(t: str) -> str:
    t = URL.sub("", str(t))
    t = MENTION.sub("", t)
    return WS.sub(" ", t).strip()


def clean_summary(t: str) -> str:
    t = re.sub(r"_header_id\|>", "", str(t))
    t = re.sub(r"\(\s*\d+(\s*,\s*\d+)*\s*\)", "", t)   # tweet-index citations
    return WS.sub(" ", t).strip()


def load_corpus(fname: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / fname, usecols=lambda c: c in {"UniversalMessageId", "Message"}, dtype=str)
    df["text"] = df["Message"].map(clean)
    df = df[df.text.str.len() >= 20].drop_duplicates("text")
    return df[["UniversalMessageId", "text"]].reset_index(drop=True)


def main(device: str, batch: int, limit: int | None):
    from sentence_transformers import SentenceTransformer
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL, device=device, model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 256
    for name, (cf, sf) in CORPORA.items():
        df = load_corpus(cf)
        if limit:
            df = df.head(limit)
        df.to_parquet(OUT / f"{name}_texts.parquet", index=False)
        E = model.encode(df.text.tolist(), batch_size=batch, normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True)
        np.save(OUT / f"{name}_qwen3e8b.npy", E.astype(np.float16))
        S = pd.read_csv(RAW / sf)["summary"].map(clean_summary).tolist()
        ES = model.encode(S, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        np.save(OUT / f"{name}_summaries_qwen3e8b.npy", ES.astype(np.float16))
        print(f"{name}: {len(df)} docs -> {E.shape}, {len(S)} summaries -> {ES.shape}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="smoke test: docs per corpus")
    a = p.parse_args()
    main(a.device, a.batch, a.limit)
