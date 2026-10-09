# Experiment Card: theme recall versus coverage at the theme scale

- **Study / version / date**: theme-recall v1, 2026-10-09.
- **Hypothesis**: with the reading granularity fixed by a per-corpus inventory of 30 themes,
  the mass-weighted recall of themes an LLM reports from an n-document sample is predicted
  across the eight corpora by coverage at the theme scale, 1 - G(n, r_theme), with Spearman at
  least 0.7 at n = 50 and n = 200 (bootstrap band excluding zero), and within each corpus recall
  rises with n along the coverage curve.
- **Sample / cohort definition (FROZEN)**: the same 1,200 draws as summary-n-curve v1
  (`data/summaries/<corpus>_n<n>.jsonl`, field `idx`), eight corpora, n in {25, 50, 100, 200,
  400}, 30 draws each. Theme inventory: MiniBatchKMeans k = 30 on the full Qwen3 embeddings of
  each corpus (seed 0), each cluster named by Gemma-4-31B-it from its 15 documents nearest the
  centroid; cluster mass = share of corpus documents.
- **Conditions and metrics**: per draw, (a) label coverage = mass of clusters that have at
  least one drawn document (exact, no LLM); (b) LLM recall = mass of inventory themes Gemma-4
  marks present after reading the n documents (temperature 0, themes listed in a fixed random
  order, answer as a list of theme ids); (c) LLM precision against the label truth. Predictors:
  1 - G(n, r_theme) with r_theme = median within-cluster 1-NN distance of the corpus, and
  1 - G at the shared radii from the n-curve study. Tests: cross-corpus Spearman at each n with
  bootstrap bands over draws; within-corpus Spearman across n; per-draw correlation pooled.
- **Compute plan**: 8 x 30 naming prompts plus 1,200 enumeration prompts on the running vLLM
  server (GPUs 4, 5), concurrency 8; k-means on CPU. Under an hour expected from the n-curve
  throughput (1,200 prompts in about 40 minutes).
- **Smoke test**: `uv run scripts/theme_inventory.py --smoke` (one corpus: cluster, name,
  print the 30 names) and `uv run scripts/theme_enumeration.py --smoke` (one corpus, n = 25
  and 400, 2 draws: output parses to theme ids).
- **Stopping rule**: if the cross-corpus Spearman of LLM recall with 1 - G(n, r_theme) is
  below 0.5 at both n = 50 and n = 200 while label coverage is predicted (Spearman at least
  0.7), the LLM reading is the failure and the paper reports label coverage as the enumeration
  result; if label coverage itself is not predicted, the theme-scale coverage claim is
  unsupported and no rerun follows without an override.
- **Artifacts**: `data/themes/<corpus>_themes.json`, `data/themes/<corpus>_labels.npy`,
  `data/themes/<corpus>_n<n>_enum.jsonl`, `results/theme_recall.csv`,
  `results/theme_recall_draws.csv`, logs in `results/`.
- **Monitor contract**: background; report on completion or error only, leading with the
  cross-corpus Spearman at n = 50 and the artifact path. Expires 2026-10-10 18:00.
- **Status log**:
