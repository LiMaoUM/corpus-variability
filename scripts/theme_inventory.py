"""Build a fixed-granularity theme inventory per corpus (card: experiments/theme-recall.card.md).

k = 30 MiniBatchKMeans on the full Qwen3 embeddings; each cluster is named by Gemma-4-31B-it
from its 15 documents nearest the centroid. Outputs data/themes/<corpus>_labels.npy (cluster
id per document) and data/themes/<corpus>_themes.json (id, name, definition, mass, r_theme).
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summary_n_curve import CORPUS_EMB, EMB, unit  # noqa: E402
from summarize_draws import BASE_URL, CORPORA as TEXT_FILES, MODEL  # noqa: E402

OUT = ROOT / "data/themes"
K = 30
NAME_PROMPT = (
    "The {m} documents below were grouped together because they are about the same theme. Give the "
    "theme a short name (at most 8 words) and a one-sentence definition that would let a reader "
    "decide whether a new document belongs to it. Answer as JSON with keys \"name\" and \"definition\".\n\n{docs}"
)


async def name_cluster(client, sem, docs):
    prompt = NAME_PROMPT.format(m=len(docs), docs="\n".join(f"- {d[:500]}" for d in docs))
    async with sem:
        r = await client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=150,
                                                 messages=[{"role": "user", "content": prompt}])
    txt = r.choices[0].message.content.strip()
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    try:
        j = json.loads(txt)
        return str(j.get("name", "")).strip(), str(j.get("definition", "")).strip()
    except json.JSONDecodeError:
        return txt[:60], txt


async def build(corpora):
    client = AsyncOpenAI(base_url=BASE_URL, api_key="local", timeout=600)
    sem = asyncio.Semaphore(8)
    OUT.mkdir(parents=True, exist_ok=True)
    for c in corpora:
        X = unit(np.load(EMB / CORPUS_EMB[c]).astype(np.float32))
        texts = pd.read_parquet(EMB / TEXT_FILES[c]).text.tolist()
        km = MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=4096).fit(X)
        labels = km.labels_
        np.save(OUT / f"{c}_labels.npy", labels)
        # theme scale: median within-cluster 1-NN distance over a 3000-doc sample
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], min(3000, X.shape[0]), replace=False)
        d1 = []
        for k in range(K):
            m = idx[labels[idx] == k]
            if m.size >= 3:
                d1.append(NearestNeighbors(n_neighbors=2, metric="cosine").fit(X[m]).kneighbors(X[m])[0][:, 1])
        r_theme = float(np.median(np.concatenate(d1)))
        jobs = []
        for k in range(K):
            members = np.where(labels == k)[0]
            sims = X[members] @ km.cluster_centers_[k] / np.linalg.norm(km.cluster_centers_[k])
            top = members[np.argsort(-sims)[:15]]
            jobs.append(name_cluster(client, sem, [texts[i] for i in top]))
        names = await asyncio.gather(*jobs)
        mass = np.bincount(labels, minlength=K) / labels.size
        themes = [dict(id=k, name=n, definition=d, mass=float(mass[k])) for k, (n, d) in enumerate(names)]
        json.dump(dict(corpus=c, K=K, r_theme=r_theme, themes=themes), (OUT / f"{c}_themes.json").open("w"), indent=1)
        print(f"{c}: r_theme={r_theme:.3f}; " + "; ".join(f"{t['id']}:{t['name']} ({t['mass']:.2f})" for t in themes[:8]) + " ...", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--corpora", nargs="*", default=list(CORPUS_EMB))
    a = p.parse_args()
    asyncio.run(build(["citizens"] if a.smoke else a.corpora))
