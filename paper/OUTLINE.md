# Outline and evidence map: How Many Documents Cover a Corpus

ACL/EMNLP via ARR, 8 pages main text plus appendix. Each section lists its claim, the evidence
(file and number), and the exhibit. Word budget in parentheses (about 5,500 words of body).

## Abstract (180 words)

A sample of a corpus should come with the same two numbers a survey sample does: how much of
the population it covers and how many more units a target needs. The missing mass at resolution
r is the share of the corpus farther than r from every sampled document; Maurer's estimator G
reads it from the sample alone. We show (1) G's dependence on sample size n and resolution r
collapses onto one curve in n r^d, with d the intrinsic dimension of the embedded corpus, so d
is the corpus property that sets the sample size, and a fit on 1,000 documents predicts
isolation at 7,000; (2) for a finite corpus the missing mass is G (1 - n/N), the survey finite
population correction, to MAE 0.01; (3) the collapse fails on a corpus whose local dimension is
heterogeneous, so its R2 is a diagnostic; (4) "how many documents to see every theme" is a
different question, answered by Good-Turing on theme masses, and its answer is the same curve
in every corpus for a fixed number of themes. Ten corpora, three platforms, three seeds.

## 1 Introduction (700)

- Problem: samples of corpora are fed to annotators and LLMs with no coverage statement; the
  survey sampler has a margin of error and a sample-size formula, the corpus sampler has
  neither. (R19: motivate by the subject.)
- What the paper provides: a label-free coverage estimate at a chosen semantic resolution, a
  corpus parameter (d) that sets the required n, a finite population correction, a diagnostic
  for corpora the formula does not fit, and the distinction from theme coverage.
- Contributions list (4 items as in the abstract). Numbers: held-out R2 0.95, MAE 0.01.

## 2 Background (650)

- 2.1 Sample coverage in the discrete case: Good-Turing 1 - f1/n (good1953), Chao1 and
  coverage-based rarefaction (chao1984, chao2012coverage), unseen species (efron1976,
  orlitsky2016). Applications to collections: hajic2025, tran2017, chung2018, pal2026 (discrete
  states).
- 2.2 The metric-space extension: Maurer 2022 defines the conditional missing mass at scale r
  and G(X, r) = share of sample points with no other sample point within r; bias at most 1/n,
  variance bound 3/n (Theorem 2.1). Theory only. Chao 2019 attribute diversity as the ecology
  precedent for a distance threshold tau.
- 2.3 Embedding-space diversity and coverage describe the sample in hand: Vendi (friedman2023,
  pasarkar2024), precision/recall and density/coverage (kynkaanniemi2019, naeem2020), MAUVE
  (pillutla2021). None extrapolates to a required n.
- 2.4 Saturation practice: interviews (guest2006, hennink2022), LLM coding (depaoli2024), topic
  models (pham2024topicgpt stopping rule; maeda2025 100 of 522 documents recover 60% of
  topics). Rules of thumb with no estimator.

## 3 Method (1,100)

- 3.1 Setting and estimand. Corpus of N documents embedded in a metric space (cosine on
  normalised embeddings); sample of n without replacement; resolution r chosen by the user as
  "what counts as the same content". Missing mass M(n, r) = share of corpus documents farther
  than r from every sampled one. Estimator G from maurer2022.
- 3.2 Collapse law. Documents on a manifold of intrinsic dimension d: ball mass about r^d,
  isolation probability about exp(-c n r^d); the (n, r) surface collapses on n r^d. Fit
  log(-log G) = a + b log(n r^d) with d from TwoNN (facco2017; levina2005 as alternative).
  Exponent b below 1 is a stretched exponential from density heterogeneity. Required n for
  isolation eps: n* = (log(1/eps) / (e^a r^d))^(1/b). The survey analogue n = z^2 s^2 / e^2.
- 3.3 Finite population correction. G estimates the generating distribution's missing mass; the
  finite corpus's is G (1 - n/N).
- 3.4 Diagnostic. One-variable collapse R2 (and held-out R2) says whether one d describes the
  corpus. Probe for failures: near-duplicate share, local d across k-means clusters.
- 3.5 Theme coverage as a partition question. For a partition with masses m_k, expected
  coverage sum m_k (1 - (1 - m_k)^n); label coverage; LLM enumeration readout.
- Exhibit: Figure 1, schematic of isolation at radius r on a 2-d toy (only if it shows the
  mechanism; else drop per R17). Equation block for G, collapse, n*, FPC.

## 4 Data (450)

- Synthetic: Gaussian manifolds d in {2, 4, 8} in R^64 (true d known); Zipf topic mixtures
  (20,000 docs, 50 to 500 topics).
- SummEval triple: AskDocs, Trump, TIFU, 10,000 documents each, GTR-T5-base embeddings, seeds
  42, 43, 44 (ni2022gtr); Qwen3-Embedding-8B embeddings for the enumeration study.
- Census tweets 2020: five keyword slices, 7,610 to 30,136 documents, Qwen3-Embedding-8B.
- Truth Social and Bluesky: 50,000-post uniform samples each, 2024-05 to 2024-12, English
  (pending).
- Summaries and inventories: Gemma-4-31B-it, temperature 0; 1,200 draws (30 per corpus and n
  in {25, 50, 100, 200, 400}); 30-theme inventories per corpus.
- Table 1: corpora, N, embedding, d (TwoNN).

## 5 Results (1,900)

- 5.1 G tracks true coverage (smoke): rho 0.968, MAE 0.029 on Zipf mixtures at alpha 0.5;
  radius matters, motivating the collapse law. (results/smoke_synthetic.log)
- 5.2 Collapse on n r^d. Table 2: per corpus d (TwoNN), R2 with n r^d, with n only, with r
  only, held-out R2, MAE of predicted G at n = 7,000 from a fit on n <= 1,000. Manifolds
  0.95 to 0.97; AskDocs 0.973 / 0.931 / 0.044; TIFU 0.972 / 0.956 / 0.029; tweets 0.946 /
  0.939 / 0.051; Trump 0.811 / 0.636 / 0.064; Zipf 0.662 / 0.639 / 0.141. Ratio of fitted
  exponents recovers d (2.1, 4.4, 7.3 for 2, 4, 8). Seeds 42 to 44: AskDocs held-out 0.950
  (0.019), TIFU 0.954 (0.004), Trump 0.555 (0.092). Figure 2: collapse plot, G against n r^d,
  one panel per corpus, colour by r. (results/eureka1_collapse.csv, eureka1_predict.csv,
  replicate_seeds.csv)
- 5.3 Why Trump fails. Near-duplicate share 3.4% (0.0% elsewhere); local d across 20 clusters
  2.0 to 16.2 (CV 0.52; AskDocs 0.15, TIFU 0.10); dedupe leaves held-out R2 at 0.74. The
  collapse R2 is a diagnostic. (results/collapse_probe.csv)
- 5.4 Finite population correction. 144 cells: G vs M MAE 0.068, max 0.42; G (1 - n/N) MAE
  0.015, max 0.11, rho 0.996; per corpus 0.009 to 0.019 across seeds. Table 3 by n.
  (results/eureka2_error_bound.csv, replicate_seeds.csv)
- 5.5 Theme coverage is a partition question. Label coverage at n is the same in all eight
  corpora (0.58 to 0.65 at 25, 0.82 to 0.86 at 50, 0.94 to 0.98 at 100) and equals the
  Good-Turing expectation from the masses to MAE 0.005; coverage at absolute r differs tenfold
  (0.02 to 0.15 at n = 25). LLM enumeration tracks label coverage within corpus (Spearman +1.00
  in 7 of 8), precision 0.73 at n = 25 to 0.98 at n = 400. Shared taxonomy at four merge radii:
  G at absolute r does not order corpora (Spearman +0.31, +0.21, +0.38, -0.40 at n = 50), the
  mass vector does (+0.93 to +1.00). Figure 3: label coverage against n, eight corpora on one
  curve, with the Good-Turing expectation. (results/theme_recall.csv, shared_taxonomy_*.log)
- 5.6 Truth Social and Bluesky (pending): d, collapse R2, FPC.

## 6 Discussion (500)

- What a user does: choose r (the resolution of the question), estimate d from a pilot,
  read n* and the coverage band; for a theme list, use the theme masses.
- Two questions, two tools; why free summaries and fixed-k inventories are scale-free.
- Relation to survey sampling: d plays the role of the variance, FPC is the same correction,
  and the coverage statement is a margin of error for enumeration.

## 7 Limitations (250)

- Embedding dependence (two encoders tested; r is encoder-specific).
- Summary stability: within-corpus direction only (8 of 8), no cross-corpus prediction.
- Corpora: English, social media and forum text; Truth Social is candidate-conditioned.
- Stretched exponent b is fitted, no closed form.

## Appendix

- A. Synthetic smoke test table and Chao required-n comparison.
- B. Summary-stability study (n-curve): full table and the negative cross-corpus result.
- C. Prompts (summary, theme naming, enumeration).
- D. Shared taxonomy at four merge radii, full table.
- E. Bound for r-stable functions (viol + M) and its looseness for proportions.

## Figures (table vs figure rule)

- Fig 2 collapse: pattern across an ordered variable, figure.
- Fig 3 theme coverage against n: eight curves on one, figure.
- Table 2 collapse statistics: exact values for reuse, table.
- Table 3 FPC: few cells, table.
