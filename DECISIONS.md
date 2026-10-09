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
