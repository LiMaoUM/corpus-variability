"""Embed the Truth Social and Bluesky samples with Qwen3-Embedding-8B, as for the other corpora.
Reads data/emb/{ts,bsky}_texts.parquet; writes data/emb/{ts,bsky}_qwen3e8b.npy (float16).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data/emb"
MODEL = "Qwen/Qwen3-Embedding-8B"


def main(device: str, batch: int, limit: int | None):
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(MODEL, device=device, model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 256
    for name in ("ts", "bsky"):
        texts = pd.read_parquet(EMB / f"{name}_texts.parquet").text.tolist()
        if limit:
            texts = texts[:limit]
        E = model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False)
        np.save(EMB / f"{name}_qwen3e8b{'_smoke' if limit else ''}.npy", E.astype(np.float16))
        print(f"{name}: {len(texts)} -> {E.shape}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    main(a.device, a.batch, a.limit)
