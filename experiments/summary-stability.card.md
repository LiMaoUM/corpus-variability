# Experiment Card: summary stability versus coverage (census corpora)

- **Study / version / date**: summary-stability v1, 2026-10-09.
- **Hypothesis**: across the five 2024 census tweet corpora, the instability of LLM summaries of
  50-tweet samples (mean pairwise cosine distance among 100 independent summaries) increases
  with the corpus's missing mass at n = 50 and with its intrinsic dimension d; the ordering of
  the five corpora by summary instability matches the ordering by G(50, r) (Spearman over five
  corpora, with a bootstrap band over draws).
- **Sample / cohort definition (FROZEN)**: `from-windows-2026-07-26/data/`: random.csv,
  full_citizenship_question.csv, trump.csv, covid.csv, illegal.csv; cleaned (URLs and
  mentions removed), length >= 20 characters, deduplicated on text. Summaries: the 100-row
  `*_summaries.csv` files (LLaMa2TS-0.1.0, 2024), header and index-citation artefacts stripped.
- **Conditions and metrics**: embeddings from Qwen/Qwen3-Embedding-8B (bf16, 256 tokens,
  normalised). Corpus side: G(50, r) averaged over 200 fresh draws of 50 at radii from the
  pilot 1-NN distance quantiles, finite-corpus M(50, r), TwoNN d, participation ratio. Summary
  side: mean pairwise cosine distance of the 100 summary embeddings, and mean distance to the
  summary centroid. Agreement: Spearman across the five corpora; bootstrap over summaries and
  draws for bands.
- **Compute plan**: one H100 shared (GPU 6 had about 60 GB free at 2026-10-09 00:30), model
  weights 16 GB bf16, activations small at batch 64 x 256 tokens. About 95k documents; wall time
  from the smoke test.
- **Smoke test**: `uv run scripts/embed_census.py --limit 200` (schema, dtype, GPU memory).
- **Stopping rule**: five corpora give five points; if Spearman is below 0.5 or its bootstrap
  band covers zero, report as inconclusive and do not add corpora from the same pool. The
  SummEval triple with fresh summaries is the next study, with its own card.
- **Artifacts**: `data/emb/<corpus>_qwen3e8b.npy`, `data/emb/<corpus>_summaries_qwen3e8b.npy`,
  `data/emb/<corpus>_texts.parquet`, `results/summary_stability.csv`, log
  `results/embed_census.log`.
- **Monitor contract**: background run; report on completion or error only, leading with the
  headline Spearman and artifact path. Expires 2026-10-09 06:00.
- **Status log**:
