"""Re-judge the r calibration pairs with Gemma-4-31B-it (DECISIONS.md 2026-10-13) and report
agreement with Qwen3.8-27B. Resumable per corpus. Writes data/rcal_gemma/<corpus>.jsonl and
prints agreement (exact label, and SAME vs not SAME) with Cohen's kappa."""
import asyncio, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from openai import AsyncOpenAI
from sklearn.metrics import cohen_kappa_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from r_calibration import CORPORA, PROMPT, EMB  # noqa: E402
SRC, OUT = ROOT / "data/rcal", ROOT / "data/rcal_gemma"
URL, MODEL = "http://127.0.0.1:8800/v1", "google/gemma-4-31B-it"

async def judge(client, sem, prompt):
    async with sem:
        r = await client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=5, messages=[{"role": "user", "content": prompt}])
    t = (r.choices[0].message.content or "").upper()
    return next((k for k in ("UNRELATED", "SAME", "TOPIC") if k in t), "OTHER")

async def run():
    client = AsyncOpenAI(base_url=URL, api_key="local", timeout=7200, max_retries=8)
    sem = asyncio.Semaphore(4); OUT.mkdir(exist_ok=True)
    for c, (e, t) in CORPORA.items():
        if (OUT / f"{c}.jsonl").exists() or not (SRC / f"{c}.jsonl").exists():
            continue
        texts = pd.read_parquet(EMB / t).text.tolist()
        pairs = [json.loads(l) for l in (SRC / f"{c}.jsonl").read_text().splitlines() if l.strip()]
        outs = await asyncio.gather(*(judge(client, sem, PROMPT.format(a=texts[p["i"]][:800], b=texts[p["j"]][:800])) for p in pairs))
        with (OUT / f"{c}.jsonl").open("w") as fh:
            for p, o in zip(pairs, outs):
                fh.write(json.dumps(dict(p, label_gemma=o)) + "\n")
        print(c, "done", flush=True)

def report():
    rows = []
    for f in sorted(OUT.glob("*.jsonl")):
        d = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines()])
        rows.append(dict(corpus=f.stem, n=len(d), agree=(d.label == d.label_gemma).mean(),
                         kappa=cohen_kappa_score(d.label, d.label_gemma),
                         kappa_same=cohen_kappa_score(d.label == "SAME", d.label_gemma == "SAME")))
    df = pd.DataFrame(rows).round(3); df.to_csv(ROOT / "results/rcal_crossjudge.csv", index=False); print(df.to_string(index=False))
    allp = pd.concat([pd.DataFrame([json.loads(l) for l in f.read_text().splitlines()]) for f in OUT.glob("*.jsonl")])
    print("pooled agree %.3f kappa %.3f kappa(SAME) %.3f" % ((allp.label == allp.label_gemma).mean(), cohen_kappa_score(allp.label, allp.label_gemma), cohen_kappa_score(allp.label == "SAME", allp.label_gemma == "SAME")))

if __name__ == "__main__":
    asyncio.run(run()); report()
