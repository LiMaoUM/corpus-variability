# Decisions

## 2026-10-08 Project framing (Mao, interactive)

Context: the idea has sat in the backlog since 2025-07-28 (`_corpus-sweep-2026-06-19-plan.md:65`).
A vault search on 2026-10-08 found no existing variability or saturation metric for a corpus in
any project. Seeds: SummEval's flat coverage-versus-n result and reviewer item RR5, Maeda 2025's
topic-recovery numbers, the agrifood persona-bank saturation stopping rule.

1. **Estimand: coverage of the finite corpus in hand.** The corpus is the population; sampling
   saves annotation and LLM budget; the full corpus validates the estimate. Rationale: this is
   the use case in Mao's projects (full Truth Social and Bluesky corpora exist, samples are fed to
   LLMs), and reviewers accept a verifiable target. The generating-process estimand (Chao-style
   unseen mass) is deferred to an appendix if at all.
2. **Primary metric: nearest-neighbour isolation coverage.** The continuous analogue of
   Good-Turing's `1 - f_1/n`: a sampled document whose nearest sampled neighbour is farther than
   `r` is a singleton. One granularity parameter `r`. Comparison lines: Vendi score convergence
   and discrete-topic iNEXT rarefaction. Rationale: no topic model needed, extrapolation to a
   required n follows Chao's coverage formulas, and it is new.
3. **Validation corpora:** synthetic GMM mixtures with known topic mass (the only setting where
   the metric can be checked against true coverage), the SummEval triple (AskDocs, Trump, TIFU),
   Truth Social and Bluesky full corpora, and Maeda 2025's 522 documents if obtainable.
4. **First step:** scaffold the repo and run a synthetic smoke test of the metric against true
   coverage. No writing before the smoke test reports.

## 2026-10-09 The 2026 restart is the project (Mao)

The 2024 exploration on `LiMaoUM/corpus-variability` (covariance spectra, eigenvector
reconstruction, decoder fine-tuning) is kept as branch `archive-2024`; the 2026 tree is `main`.
Code is not merged. What carries over is recorded in `docs/archive-2024.md`: the spectrum
comparison as a second-moment comparison line, and the 100 x 50-tweet repeated-sample
summaries on five corpora as a downstream-stability check. Rationale (Mao): absorb what is
usable, treat the restart as the project.

## 2026-10-09 Next study: summary instability versus n (Mao, interactive)

The census check at n = 50 was inconclusive (five corpora, rho 0.5 to 0.6). Mao chose the
follow-up design: fresh summaries from Gemma-4-31B-it (vLLM, 127.0.0.1:8800) on the SummEval
triple (AskDocs, Trump, TIFU) plus the five census corpora, n in {25, 50, 100, 200, 400},
30 draws per (corpus, n), about 1,200 summaries. Rationale: eight corpora across two platforms
double the cross-corpus points and one summariser makes the census corpora comparable; the
within-corpus curve of instability against n is the test that G(n, r) should predict. Card:
`experiments/summary-n-curve.card.md`.

## 2026-10-09: Main line, venue, corpora, EACL paragraph (Mao, via /decide)

The paper's main line is the corpus sample-size formula: the n r^d collapse with TwoNN d joined
to the finite population correction on Maurer's G, with the collapse R2 as a corpus diagnostic.
Venue ACL/EMNLP via ARR. Truth Social and Bluesky enter as 50k-document samples each. The EACL
camera-ready of SummEval gets one paragraph citing this project as in prep to answer RR5.
Rationale: the two claims replicate on three seeds and three corpora, the survey analogy is
exact with FPC, and the downstream summary-stability link is a within-corpus direction only and
stays a limitation.

## 2026-10-09: Cross-corpus test by enumeration (Mao, interactive)

The free-summary stability test failed across corpora because the summariser chooses its own
abstraction scale per genre (TIFU reads coarse, AskDocs fine). Mao chose the enumeration study
over the cheaper per-corpus radius fit: build a fixed-granularity theme inventory per corpus
(k = 30 clusters on the full Qwen3 embeddings, named by Gemma-4), then for each existing draw
ask Gemma-4 which themes are present and score mass-weighted recall against the inventory.
Coverage at the theme scale should predict recall across corpora. Card:
`experiments/theme-recall.card.md`.

## 2026-10-09: Absolute-scale cross-corpus test, label version first (Mao, interactive)

The fixed-k enumeration showed that any corpus-relative readout is scale-free. Mao chose the
label version of a shared taxonomy: merge the eight 30-theme inventories into one shared list
by centroid distance, assign every document of every corpus to its nearest shared theme, score
the same 1,200 draws by the corpus mass of shared themes they contain, and test whether G at an
absolute radius predicts it across corpora. No LLM calls; the LLM enumeration on the shared list
follows only if the label version succeeds. Script `scripts/shared_taxonomy.py`.

## 2026-10-09: Shared-taxonomy outcome (Claude, for Mao's confirmation)

Across four merge radii, label-free coverage at absolute r does not predict shared-taxonomy
coverage across corpora; the partition's own mass vector does (tautologically). Proposed
position: the sample-size formula is a resolution-r statement, validated cross-corpus by claim 2
(G with FPC against the true missing mass); theme-level coverage is a partition question,
answered by Good-Turing on theme masses and validated within corpus. Confirmed by the entry below.

## 2026-10-09: Two questions, two tools (Mao, via /decide)

The sample-size formula is a resolution-r statement (how many documents until a new one has a
sampled neighbour within r), label-free, carried across corpora by the intrinsic dimension d
and validated by claims 1 and 2. Theme coverage is a partition question, answered by Good-Turing
on the theme masses, validated within corpus by the enumeration study and universal across
corpora for a fixed k. The paper does not claim that the label-free formula predicts theme
recall across corpora; summary stability stays a within-corpus direction and a limitation.
Confirms the "Shared-taxonomy outcome" entry above. Rationale: across four merge radii of a
shared taxonomy, G at absolute r did not predict partition coverage across corpora while the
partition's own mass vector did, so the two are different functionals of the geometry.
