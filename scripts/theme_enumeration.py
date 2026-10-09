"""Theme enumeration per draw and the recall analysis (card: experiments/theme-recall.card.md).

Step "enum": for each existing draw in data/summaries/<corpus>_n<n>.jsonl, show Gemma-4 the n
documents and the corpus's 30-theme inventory (fixed shuffled order per corpus) and ask which
themes are present. Output data/themes/<corpus>_n<n>_enum.jsonl (draw, present ids).
Step "analysis": per draw, label coverage (mass of clusters with a drawn document), LLM recall
(mass of themes marked present), precision against the label truth, G at r_theme and at the
shared radii. Cross-corpus Spearman at each n, within-corpus across n, pooled per draw.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summary_n_curve import CORPUS_EMB, EMB, SUMM, unit  # noqa: E402
from summarize_draws import BASE_URL, CORPORA as TEXT_FILES, MODEL, DOC_CHARS  # noqa: E402

TH = ROOT / "data/themes"
SHARED_R = {0.25: 0.2659, 0.5: 0.3367, 0.75: 0.4355}   # from summary_n_curve.log
ENUM_PROMPT = (
    "Below are {n} documents sampled from a larger collection, followed by a list of {K} themes "
    "found in the full collection. Decide which themes are present among these {n} documents. A "
    "theme is present only if you can point to one document that is primarily about it; in a small "
    "sample most themes will be absent, and similar-sounding themes are distinct, so pick the single "
    "best theme for each document. Answer with one line per present theme in the form "
    "\"theme_id: document_number\" (the document that best supports it), nothing else.\n\n"
    "DOCUMENTS\n{docs}\n\nTHEMES\n{themes}\n\nPresent themes:"
)


def load_inventory(c):
    j = json.load((TH / f"{c}_themes.json").open())
    rng = np.random.default_rng(hash(c) % (2**32))
    order = rng.permutation(len(j["themes"]))
    return j, order


async def enum_one(client, sem, prompt):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=400,
                                                         messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                await asyncio.sleep(2 * (attempt + 1))


async def enum_step(corpora, n_grid, max_draws, concurrency):
    client = AsyncOpenAI(base_url=BASE_URL, api_key="local", timeout=600)
    sem = asyncio.Semaphore(concurrency)
    for c in corpora:
        inv, order = load_inventory(c)
        texts = pd.read_parquet(EMB / TEXT_FILES[c]).text.tolist()
        theme_block = "\n".join(f"[{inv['themes'][k]['id']}] {inv['themes'][k]['name']}: {inv['themes'][k]['definition']}" for k in order)
        for n in n_grid:
            f_in = SUMM / f"{c}_n{n}.jsonl"
            if not f_in.exists():
                continue
            recs = [json.loads(l) for l in f_in.read_text().splitlines() if l.strip()][:max_draws]
            f_out = TH / f"{c}_n{n}_enum.jsonl"
            done = {json.loads(l)["draw"] for l in f_out.read_text().splitlines() if l.strip()} if f_out.exists() else set()
            jobs = [(r, ENUM_PROMPT.format(n=n, K=inv["K"], docs="\n".join(f"[{i + 1}] {texts[j][:DOC_CHARS]}" for i, j in enumerate(r["idx"])), themes=theme_block))
                    for r in recs if r["draw"] not in done]
            if not jobs:
                continue
            outs = await asyncio.gather(*(enum_one(client, sem, p) for _, p in jobs))
            with f_out.open("a") as fh:
                counts = []
                for (r, _), o in zip(jobs, outs):
                    pairs = {}
                    for t, d in re.findall(r"\[?(\d+)\]?\s*:\s*\[?(\d+)\]?", o or ""):
                        t, d = int(t), int(d)
                        if t < inv["K"] and 1 <= d <= n and t not in pairs:
                            pairs[t] = d
                    counts.append(len(pairs))
                    fh.write(json.dumps(dict(draw=r["draw"], present=sorted(pairs), support={str(k): v for k, v in pairs.items()}, raw=(o or "")[:300])) + "\n")
            print(f"{c} n={n}: {len(jobs)} enumerations, mean present {np.mean(counts):.1f}", flush=True)


def analysis_step():
    rows = []
    X = {c: unit(np.load(EMB / f).astype(np.float32)) for c, f in CORPUS_EMB.items()}
    for c in CORPUS_EMB:
        if not (TH / f"{c}_themes.json").exists():
            continue
        inv = json.load((TH / f"{c}_themes.json").open())
        mass = np.array([t["mass"] for t in inv["themes"]])
        labels = np.load(TH / f"{c}_labels.npy")
        r_t = inv["r_theme"]
        for f_e in sorted(TH.glob(f"{c}_n*_enum.jsonl")):
            n = int(f_e.stem.split("_n")[1].split("_")[0])
            draws = {json.loads(l)["draw"]: json.loads(l) for l in f_e.read_text().splitlines() if l.strip()}
            recs = {r["draw"]: r for r in (json.loads(l) for l in (SUMM / f"{c}_n{n}.jsonl").read_text().splitlines() if l.strip())}
            for d, e in draws.items():
                present = e["present"]
                idx = np.array(recs[d]["idx"])
                grounded = np.zeros(len(mass), bool)
                for t, dn in e.get("support", {}).items():
                    if labels[idx[int(dn) - 1]] == int(t):
                        grounded[int(t)] = True
                seen = np.zeros(len(mass), bool)
                seen[np.unique(labels[idx])] = True
                rep = np.zeros(len(mass), bool)
                rep[present] = True
                Xs = X[c][idx]
                d_s = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xs).kneighbors(Xs)[0][:, 1]
                rows.append(dict(corpus=c, n=n, draw=d, label_cov=float(mass[seen].sum()), llm_recall=float(mass[rep].sum()),
                                 llm_recall_true=float(mass[rep & seen].sum()),
                                 llm_recall_grounded=float(mass[grounded].sum()),
                                 precision=float((rep & seen).sum() / max(rep.sum(), 1)),
                                 cov_theme=1 - float((d_s > r_t).mean()),
                                 **{f"cov_q{q}": 1 - float((d_s > r).mean()) for q, r in SHARED_R.items()}))
    dd = pd.DataFrame(rows)
    dd.to_csv(ROOT / "results/theme_recall_draws.csv", index=False)
    df = dd.groupby(["corpus", "n"]).mean(numeric_only=True).drop(columns="draw").reset_index()
    df.to_csv(ROOT / "results/theme_recall.csv", index=False)
    print(df.round(3).to_string(index=False))
    rng = np.random.default_rng(1)
    for target in ["label_cov", "llm_recall", "llm_recall_true", "llm_recall_grounded"]:
        print(f"\nCross-corpus Spearman of {target} with coverage predictors at each n (bootstrap 90% band over draws):")
        for n, g in df.groupby("n"):
            if len(g) < 4:
                continue
            for col in ["cov_theme", "cov_q0.5"]:
                rho = spearmanr(g[col], g[target]).statistic
                boots = []
                for _ in range(300):
                    y, x = [], []
                    for c in g.corpus:
                        gd = dd[(dd.corpus == c) & (dd.n == n)]
                        bi = rng.choice(len(gd), len(gd), replace=True)
                        y.append(gd[target].to_numpy()[bi].mean()); x.append(gd[col].to_numpy()[bi].mean())
                    boots.append(spearmanr(x, y).statistic)
                lo, hi = np.nanpercentile(boots, [5, 95])
                print(f"  n={n:4d} {col:9s} rho={rho:+.2f} band=[{lo:+.2f}, {hi:+.2f}] (k={len(g)})")
    print("\nWithin-corpus Spearman across n (llm_recall vs cov_theme):")
    for c, g in df.groupby("corpus"):
        print(f"  {c:12s} rho={spearmanr(g.cov_theme, g.llm_recall).statistic:+.2f}  recall {g.llm_recall.min():.2f}..{g.llm_recall.max():.2f}  label_cov {g.label_cov.min():.2f}..{g.label_cov.max():.2f}")
    print("\nPooled per-draw Spearman within (corpus, n): llm_recall vs cov_theme median rho=%+.2f; label_cov vs cov_theme median rho=%+.2f" % (
        np.nanmedian([spearmanr(g.cov_theme, g.llm_recall).statistic for _, g in dd.groupby(["corpus", "n"]) if len(g) >= 10]),
        np.nanmedian([spearmanr(g.cov_theme, g.label_cov).statistic for _, g in dd.groupby(["corpus", "n"]) if len(g) >= 10])))
    print("\nLLM precision against label truth, mean by n:")
    print(df.groupby("n").precision.mean().round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["enum", "analysis", "both"], default="both")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--corpora", nargs="*", default=list(CORPUS_EMB))
    p.add_argument("--concurrency", type=int, default=8)
    a = p.parse_args()
    if a.smoke:
        asyncio.run(enum_step(["citizens"], [25, 400], 2, a.concurrency))
    else:
        if a.step in ("enum", "both"):
            asyncio.run(enum_step(a.corpora, [25, 50, 100, 200, 400], 30, a.concurrency))
        if a.step in ("analysis", "both"):
            analysis_step()
