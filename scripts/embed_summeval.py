"""Build the SummEval triple (AskDocs, Trump, TIFU) as text parquet files and embed them with
Qwen3-Embedding-8B, the same way as the census corpora (scripts/embed_census.py).

Documents come from SummEval's loader, `load_corpus(name, 10000, 42)`, so the sample matches
the GTR caches used in the EACL paper. Outputs: data/emb/summeval_<name>_texts.parquet and
data/emb/summeval_<name>_qwen3e8b.npy (float16, 10000 x 4096).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/emb"
SUMMEVAL = Path("/home/maolee/projects/SummEval/revision/experiments")
MODEL = "Qwen/Qwen3-Embedding-8B"
NAMES = ["askdocs", "trump", "tifu"]


def load_texts(name: str, n: int = 10000, seed: int = 42) -> list[str]:
    sys.path.insert(0, str(SUMMEVAL))
    import importlib
    common = importlib.import_module("common")
    return common.load_corpus(name, n, seed)


def main(device: str, batch: int):
    from sentence_transformers import SentenceTransformer
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL, device=device, model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 256
    for name in NAMES:
        texts = load_texts(name)
        pd.DataFrame({"doc_id": [f"{name}_{i}" for i in range(len(texts))], "text": texts}).to_parquet(
            OUT / f"summeval_{name}_texts.parquet", index=False)
        E = model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False)
        np.save(OUT / f"summeval_{name}_qwen3e8b.npy", E.astype(np.float16))
        print(f"{name}: {len(texts)} docs -> {E.shape}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    main(a.device, a.batch)
