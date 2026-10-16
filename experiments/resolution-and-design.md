# Experiment card: resolution reached by a fixed sample, and designed samples (2026-10-15)

- Question (Mao, 2026-10-15): is the pessimistic main conclusion (diverse corpora need about half of the
  corpus to cover half at the level of specific content) an artifact of how we posed the question?
- Hypotheses: (1) every sample covers the corpus at some resolution, and the sample alone estimates which;
  (2) a sample chosen with the whole corpus embedded (greedy maximum coverage, k-means medoids) covers more
  than a random one of the same size; the gap is large on repetitive corpora and small on corpora whose
  documents rarely share content.
- Frozen sample: the nine Qwen3 corpora with calibration pairs plus Bluesky (data/emb, data/rcal); r_same
  from results/rsame_boot.csv (Qwen3.8), r_topic from data/rcal labels SAME or TOPIC.
- Metrics: r80(n), true and estimated from the sample, with the judged proportions of same content and same
  topic at that distance; missing mass at n = 15, 100, 1000 for random (20 draws), k-means medoids and greedy;
  greedy n to cover half of the corpus.
- Compute: CPU only, OMP_NUM_THREADS=8, no GPU. Smoke test on Census illegal passed (greedy 6 s on 17k docs).
- Stopping rule: one pass; no reruns at other targets unless Mao asks.
- Script: scripts/resolution_and_design.py; outputs results/resolution_reached.csv, results/designed_samples.csv.
