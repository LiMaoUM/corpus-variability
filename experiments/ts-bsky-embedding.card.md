# Experiment Card: Truth Social and Bluesky 50k samples, embedding

- **Study / version / date**: ts-bsky-embedding v1, 2026-10-11.
- **Hypothesis**: claims 1 and 2 hold on two further platforms with a finite corpus of known
  size: the isolation surface collapses on n r^d with d from TwoNN (held-out R2 at least 0.9
  on a collapsing corpus, or a diagnosed failure as for Trump), and G (1 - n/N) tracks the
  finite-corpus missing mass to MAE below 0.02. The two platforms give the first cross-platform
  comparison of d at one resolution.
- **Sample / cohort definition (FROZEN)**: from `/home/data/corpus-parquet/truthsocial/trump/`
  and `/home/data/corpus-parquet/bsky/trump_flattened/`, posts with `created_at` in 2024-05-01
  to 2024-12-31 (the window the other corpus papers use; this also excludes the stray Bluesky
  partitions), `lang = 'en'`, text at least 20 characters after removing URLs and mentions,
  deduplicated on text; a uniform random sample of 50,000 posts per platform, seed 2026, drawn
  with DuckDB over the parquet partitions. Truth Social is candidate-conditioned (complete
  timelines only for the 35,269 panel authors); Bluesky `trump_flattened` is a broad political
  pool, 16.8% of roots mention Trump. Both caveats are stated in the paper.
- **Conditions and metrics**: Qwen3-Embedding-8B as for the other corpora (bf16, 256 tokens,
  no prompt). Then `scripts/replicate_seeds.py`-style analysis: TwoNN d, collapse R2, held-out
  R2, FPC test, near-duplicate share and local-d spread.
- **Compute plan**: GPU with at least 20 GB free (weights 16 GB bf16 plus activations at batch
  64 x 256). At 2026-10-11 10:00 all eight GPUs run at 100% utilisation; GPUs 0 and 1 have
  about 33 GB free each, GPU 6 (used before) has 9 GB free. Running on a shared GPU is slower;
  the census run (80k docs) took under 10 minutes on an idle GPU 6, so 100k docs on a shared
  card is expected within an hour. Sampling with DuckDB is CPU only.
- **Smoke test**: `uv run scripts/sample_ts_bsky.py --limit 2000` then
  `uv run scripts/embed_ts_bsky.py --limit 500` (schema, dtype, GPU memory).
- **Stopping rule**: this is a replication; whatever the collapse outcome, it is reported once.
  No rerun with a different window or filter without an override.
- **Artifacts**: `data/emb/ts_texts.parquet`, `data/emb/bsky_texts.parquet`,
  `data/emb/{ts,bsky}_qwen3e8b.npy`, `results/ts_bsky_claims.csv`, logs in `results/`.
  Registry entry in `~/maospace/map/DATA-REGISTRY.md` the same session.
- **Monitor contract**: background; report on completion or error only, leading with the two
  held-out R2 values and the artifact path. Expires 2026-10-12 12:00.
- **Status log**:
