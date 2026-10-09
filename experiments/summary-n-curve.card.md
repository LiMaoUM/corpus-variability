# Experiment Card: summary instability versus sample size

- **Study / version / date**: summary-n-curve v1, 2026-10-09.
- **Hypothesis**: within a corpus, the instability of LLM summaries of n-document samples
  (mean pairwise cosine distance among 30 independent summaries) falls with n at the rate the
  isolation surface G(n, r) predicts; across the eight corpora, the ordering of instability at
  a fixed n matches the ordering of G(n, r) (Spearman over eight corpora, bootstrap band
  excluding zero), and the per-draw G of the sampled documents correlates with that draw's
  summary distance to the corpus summary centroid.
- **Sample / cohort definition (FROZEN)**: SummEval triple loaded by
  `SummEval/revision/experiments/common.py::load_corpus(name, 10000, 42)` (AskDocs, Trump,
  TIFU; 20 to 6,000 characters, deduped); the five census corpora from
  `data/emb/<corpus>_texts.parquet`. Draws: for each corpus and n in {25, 50, 100, 200, 400},
  30 draws without replacement, seed 2026 + draw index. Documents truncated to 600 characters
  in the prompt.
- **Conditions and metrics**: summariser google/gemma-4-31B-it via vLLM at 127.0.0.1:8800,
  temperature 0, max 300 output tokens, one fixed prompt (`scripts/summarize_draws.py`).
  Embeddings Qwen3-Embedding-8B for documents and summaries. Metrics per (corpus, n):
  instability (mean pairwise cosine distance of the 30 summaries); G(n, r) and finite-corpus
  M(n, r) averaged over the same draws at three shared radii; per-draw G and per-draw distance
  to centroid. Tests: Spearman across corpora at each n with bootstrap bands; within-corpus
  slope of log instability on log G; per-draw Spearman pooled.
- **Compute plan**: vLLM server already running (GPUs 4, 5); 1,200 prompts, the n = 400 prompts
  about 60k characters each; client concurrency 8. Embedding on GPU 6 (16 GB). Wall time from
  the smoke test.
- **Smoke test**: `uv run scripts/summarize_draws.py --smoke` (one corpus, n in {25, 400},
  2 draws): prompt length, server throughput, output schema.
- **Stopping rule**: if the cross-corpus Spearman at n = 50 and n = 200 both sit below 0.5 or
  their bands cover zero, and the within-corpus slope on G is not negative in at least six of
  eight corpora, the downstream claim is reported as unsupported; no larger rerun without an
  explicit override.
- **Artifacts**: `data/summaries/<corpus>_n<n>.jsonl` (draw index, doc ids, summary),
  `data/emb/<corpus>_summaries_ncurve_qwen3e8b.npy`, `data/emb/summeval_<corpus>_qwen3e8b.npy`,
  `results/summary_n_curve.csv`, logs in `results/`.
- **Monitor contract**: background run; report on completion or error only, leading with the
  headline Spearman at n = 50 and the artifact path. Expires 2026-10-10 12:00.
- **Status log**:
  - 2026-10-09 v1: ran. 1,200 summaries (Gemma-4-31B-it, temperature 0), embedded with Qwen3-Embedding-8B. Cross-corpus Spearman of instability with G(n, median shared r): n=25 +0.14, n=50 +0.12, n=100 +0.12, n=200 +0.07, n=400 -0.17 (k=8; 90% bands within -0.4 to +0.5); with d_twonn +0.10 to +0.29. Within-corpus: instability falls with n in 8 of 8 corpora (Spearman with G +0.40 to +1.00; 7 of 8 at +0.70 or above), but the log-log slope on G ranges from +0.17 (citizens) to +24 (AskDocs), so no single rate. Per-draw G against distance to centroid: median rho +0.05. Verdict by the stopping rule: the cross-corpus condition fails at n=50 and n=200; the within-corpus condition holds. The downstream claim is UNSUPPORTED in its cross-corpus form and holds only as a direction within corpus. No rerun. Results: results/summary_n_curve.csv, results/summary_n_curve_draws.csv, log results/summary_n_curve.log.
