# Evidence for the two claims (2026-10-08)

All numbers from `scripts/smoke_synthetic.py`, `scripts/eureka1_collapse.py`,
`scripts/eureka1_predict.py`, `scripts/eureka2_error_bound.py`; outputs in `results/`.
Real corpora: SummEval GTR-T5-base caches (AskDocs, Trump, TIFU; 10,000 docs each, seed 42)
and 7,860 2024 tweets with 768-d encoder embeddings. One seed per corpus so far.

## Baseline: the estimator tracks true topic coverage

Synthetic Zipf topic mixtures (20,000 docs, 64-d, 50 to 500 topics, 3 seeds). Maurer's
estimator `G(n, r)` with `r` = 0.5 x median pilot pairwise cosine distance gives coverage
`1 - G` with Spearman rho 0.968 and MAE 0.029 against true topic-mass coverage (Good-Turing on
true labels: rho 0.989, MAE 0.016). Smaller radii fail (alpha 0.1: rho 0.31). The radius
matters, which is what claim 1 removes.

## Claim 1: the coverage surface collapses on n r^d, with d the intrinsic dimension

Hypothesis: `log(-log G(n, r)) = a + b log(n r^d)` with `d` estimated by TwoNN from a 2,000-doc
pilot and no reference to labels.

| corpus | d (TwoNN) | R2, n r^d | R2, n only | R2, r only | held-out R2 (fit n <= 1000, test n > 1000) |
|---|---|---|---|---|---|
| manifold d=2 | 2.01 | 0.970 | 0.060 | 0.635 | 0.952 |
| manifold d=4 | 4.21 | 0.954 | 0.044 | 0.666 | 0.924 |
| manifold d=8 | 7.60 | 0.971 | 0.054 | 0.731 | 0.939 |
| AskDocs (GTR) | 13.6 | 0.973 | 0.023 | 0.801 | 0.931 |
| TIFU (GTR) | 17.7 | 0.972 | 0.041 | 0.826 | 0.956 |
| tweets (768-d) | 14.5 | 0.946 | 0.033 | 0.863 | 0.939 |
| Trump (GTR) | 9.2 | 0.811 | 0.000 | 0.731 | 0.636 |
| Zipf k=200 (synthetic) | 11.0 | 0.662 | 0.003 | 0.558 | 0.639 |

Prediction test: fit on samples of at most 1,000 documents, predict the isolation rate at
7,000 documents for every radius. MAE of predicted G: AskDocs 0.044, TIFU 0.029, tweets 0.051,
manifolds 0.031 to 0.056; Trump 0.064; Zipf 0.141.

Two refinements the data forced. First, the exponent on `n r^d` is below 1 (0.60 to 0.75 on
Gaussian manifolds, 0.45 on AskDocs and TIFU), so the law is a stretched exponential: the
missing mass is an average of `exp(-n mu(B(x, r)))` over documents, and density heterogeneity
stretches it. The two-variable fit still recovers `d`: the ratio of the `log r` to the `log n`
coefficient is 2.1, 4.4, 7.3 for true d = 2, 4, 8, and 15.4 and 19.2 for AskDocs and TIFU
against TwoNN 13.6 and 17.7. Second, collapse fails on the Zipf mixture and on Trump, the two
corpora whose density is dominated by a few dense modes (slogans and near-duplicates in Trump).
The collapse R2 is therefore a diagnostic: a corpus that collapses has one scale parameter
`d`; one that does not needs a second (tail) parameter.

Consequence: corpus variability is summarised by `d`, and the required sample size for
isolation rate `eps` at scale `r` is `n* = (log(1/eps) / (c r^d))^(1/b)`, the corpus analogue
of `n = z^2 sigma^2 / e^2`.

## Claim 2: the finite-corpus missing mass is G times the finite population correction

Maurer's `G` estimates the missing mass of the generating distribution. The estimand here is
the finite corpus, of which the sample is part. On four corpora, six sample sizes and six
radii (144 cells), `G` against the true finite-corpus missing mass `M` has MAE 0.068 and a
maximum error of 0.42 (at n = 5,000 of N = 10,000, G is double M). With the survey
finite population correction, `G (1 - n/N)`, MAE is 0.015, maximum 0.11, Spearman 0.996;
per corpus MAE 0.011 to 0.019. The mean over cells by n:

| n | M (true) | G | G (1 - n/N) |
|---|---|---|---|
| 100 | 0.729 | 0.749 | 0.743 |
| 1,000 | 0.479 | 0.534 | 0.486 |
| 5,000 | 0.211 | 0.384 | 0.212 |

Margin-of-error reading. For any document function with range 1 that is stable at scale `r`,
the 1-NN propagated sample estimate of its corpus mean errs by at most `viol(r) + M(r)`, where
`viol` is the share of covered documents whose nearest sampled document disagrees with them.
The bound held in 144 of 144 cells on a family of 50 k-means indicator functions, but it is
loose for proportions (median 22 times the realised maximum error), and the plain sample mean
already estimates proportions to 0.003 to 0.07. Coverage is the right quantity for enumeration
questions (which themes exist, which rare claims appear), where the error equals the missing
mass by construction, and the survey margin of error handles proportions. The paper states
both and keeps the enumeration case as the use case.

## Open items

- One seed per real corpus; repeat with seeds 43 and 44 before any figure.
- The required-n formula is checked through the held-out prediction of G, since no corpus
  reached isolation 0.05 at the tested scale within 10,000 documents. A direct check needs the
  full Truth Social or Bluesky corpora.
- Truth Social and Bluesky, Maeda 2025, and a downstream LLM summary stability check are not
  run.
- Chao and Jost extrapolation from the smoke test over-predicts on the noisy Zipf setting and
  under-predicts on k=500; it is now a comparison line.

## Downstream check on the 2024 census corpora (2026-10-09)

Five census tweet corpora (7.6k to 30.1k docs after cleaning), Qwen3-Embedding-8B, 100
LLaMa2 summaries of 50-tweet draws per corpus. Summary instability (mean pairwise cosine
distance among the 100 summaries) ranks random 0.195 > citizens 0.181 > covid 0.158 >
illegal 0.153 > trump 0.140. Spearman with corpus-side quantities over the five corpora:
TwoNN d +0.60 (90% bootstrap band +0.30 to +0.70), G(50, median r) +0.50 (+0.20 to +0.51),
participation ratio +0.30, mean document pairwise distance +0.30. The direction is right and
G beats the second-moment measures, but five points from one platform and one topic family
cannot lock the claim; the card's stopping rule calls it inconclusive. Trump is the outlier
(moderate G, lowest instability): its summaries converge on one dominant mode, the same
dense-mode structure that made it fail the collapse test. At n = 50 of N >= 7,600 the finite
population correction is negligible and G equals M to 0.005.

## Seed replication and collapse probe (2026-10-09, `scripts/replicate_seeds.py`)

GTR caches, seeds 42, 43, 44, 10,000 documents each. Mean (sd) over seeds:

| corpus | d (TwoNN) | R2 on n r^d | held-out R2 | MAE of predicted G at n = 7,000 | MAE G vs M | MAE G (1 - n/N) vs M |
|---|---|---|---|---|---|---|
| AskDocs | 13.4 (0.25) | 0.972 (0.007) | 0.950 (0.019) | 0.042 (0.007) | 0.076 (0.005) | 0.010 (0.003) |
| TIFU | 17.7 (0.11) | 0.969 (0.004) | 0.954 (0.004) | 0.030 (0.004) | 0.064 (0.006) | 0.013 (0.002) |
| Trump | 9.3 (0.55) | 0.857 (0.040) | 0.555 (0.092) | 0.060 (0.004) | 0.124 (0.006) | 0.009 (0.002) |

Both claims hold on all three seeds: the collapse on AskDocs and TIFU, its failure on Trump,
and the finite population correction (MAE 0.009 to 0.013 after correction on every corpus).

Why Trump fails. In a 5,000-document sample, 3.4% of Trump documents have a near-duplicate
(cosine distance under 0.02; AskDocs and TIFU: 0.0%), and TwoNN d across 20 k-means clusters
runs from 2.0 to 16.2 (coefficient of variation 0.52; AskDocs 0.15, TIFU 0.10). Removing
near-duplicates (greedy at 0.02, 266 documents) leaves the collapse broken (held-out R2
0.74), so the cause is the spread of local dimension, with slogan-like clusters at d near 2
beside discussion clusters at d near 16, and one global d cannot describe both. The synthetic
Zipf mixture fails for the related reason of unequal topic mass (local d CV 0.21, no
duplicates). A corpus with a heterogeneous local dimension needs a mixture law, which is the
second parameter named in claim 1.

## Summary instability versus n, eight corpora (2026-10-09, `scripts/summary_n_curve.py`)

Fresh summaries from Gemma-4-31B-it (temperature 0) of 30 draws at each n in {25, 50, 100,
200, 400} on the five census corpora and the SummEval triple; instability is the mean pairwise
cosine distance among the 30 summaries (Qwen3-Embedding-8B).

Within a corpus, instability falls with n in all eight corpora and tracks G(n, r) in direction
(Spearman +0.40 to +1.00). Across corpora at a fixed n, G does not order the corpora by
instability: Spearman +0.14 at n = 25, +0.12 at n = 50, +0.07 at n = 200, -0.17 at n = 400,
with bootstrap bands that reach zero; d does no better. The per-draw G of a sample does not
predict how far that draw's summary sits from the centroid (median rho +0.05 over 40 cells).
The card's stopping rule reads this as unsupported for the cross-corpus claim.

What the table shows about why. At the shared radii, AskDocs sits at G near 1 for every n
(its documents are long questions spread thinly, so almost no sampled document has a
neighbour within r) while citizens sits at G 0.06 to 0.23; the shared scale is informative for
neither extreme. Instability is also driven by the summariser's behaviour on each genre: at
n = 25 AskDocs summaries vary most (0.27) although its G is saturated, and at n = 400 the
census corpora with the lowest G (citizens, illegal) are not the most stable. Coverage of the
embedding space and stability of a 150-word LLM summary are different quantities, and the
paper should not claim the second from the first. The within-corpus direction is the only
downstream statement the data support.

Exploratory, outside the card: corpus-specific radii (each corpus's own 1-NN quantiles at
n = 50) in `scripts/n_curve_exploratory.py`.
Result of that check: with each corpus at its own scale, cross-corpus Spearman of instability
with G is +0.04, +0.24, +0.55, +0.05, +0.43 at n = 25 to 400, and the corpus's own scale
itself is uninformative (rho about +0.05). The shared-radius choice is not what hides the
relation; the cross-corpus link is weak at any scale.
