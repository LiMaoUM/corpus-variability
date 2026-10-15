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

## 2026-10-11: No EACL paragraph; the two papers stay separate (Mao, via /decide)

The SummEval EACL 2027 camera-ready does not cite corpus-variability and gets no RR5 paragraph
from it. Supersedes the EACL-paragraph clause of the 2026-10-09 entry "Main line, venue,
corpora, EACL paragraph"; its other clauses stand. The drafted heterogeneity paragraph is kept
as material for this paper (`docs/drafts/heterogeneity-diagnostic-paragraph.tex`). Rationale:
this is a new paper, and the SummEval camera-ready is not the place for it.

## 2026-10-11: Title and drafting route (Mao, interactive)

Title: "How Many Documents Cover a Corpus". Drafting goes through the ARS academic-paper
pipeline (full mode) on the ACL template, then unslop and fred-me before Mao reads it.

## 2026-10-11: Revision experiments after the simulated panel (Mao, interactive)

The ARS panel (paper/reviews/ars-panel-2026-10-11.md) returned Major Revision. Mao approved four
experiment groups: (1) the n* test on every corpus that passes the diagnostic, with five
independent draws per grid cell, bootstrap intervals, three seeds, and a harder extrapolation from
n <= 300; (2) theme inventories at K in {10, 30, 100} plus a skewed-mass inventory, to test whether
the "same curve" is a function of K; (3) the forum corpora under Qwen3 as a second encoder, plus
two baselines, per-radius Chao and Jost extrapolation and greedy k-center covering; (4) a
calibration of r by LLM judgments of document pairs binned by cosine distance. Claim 4 (theme
coverage) is decided after the K sensitivity result.

## 2026-10-12: Claim 4 demoted; diagnostic threshold 0.9 (applied per Mao's rule)

Mao's rule of 2026-10-11 was to decide claim 4 after the K sensitivity result. The result: for
K = 10, 30, 100 and two skewed inventories, label coverage curves coincide across the eight
corpora (sd < 0.05) and the mass vector predicts them to MAE 0.004, so the "same curve" comes
from inventory construction. Claim 4 leaves the contribution list and becomes one paragraph in the
Discussion (partition coverage is set by the mass vector, a different quantity from coverage at
resolution r), with its table and figure in the appendix. The collapse diagnostic uses an
empirical threshold of out-of-sample R2 0.9, stated as fitted on these ten corpora: n* within a
factor of 1.5 in 59 of 66 cases above it, 10 of 23 below. Mao's interactive answer did not reach
the session (tool call rejected); these are the recommended options under his stated rule, to be
reversed if he says so.

Correction (2026-10-12): the counts in the entry above came from a partial tally. From
`results/nstar_v2.csv` (fit on n <= 1,000): corpora with out of sample R2 >= 0.9 give 77 of 83
cases within a factor of 1.5 and 81 of 83 within 2; corpora below 0.9 give 10 of 23 and 14 of 23.

## 2026-10-13: r calibration judged by Qwen3.8-27B (Mao, interactive)

The shared Gemma-4 server (127.0.0.1:8800) stayed saturated by another project (15 running, 47
waiting; one hour of timeouts). Mao chose the idle Qwen3.8-27B server (127.0.0.1:8830, thinking
disabled) for the r calibration judgments. The paper states that the calibration judge differs
from the Gemma-4 summarizer and enumerator.

## 2026-10-13: Judge validation by a second LLM (Mao, interactive)

The re-review asked for human labels on about 200 calibration pairs. Mao chose a second LLM
judge instead: re-judge all calibration pairs with Gemma-4-31B-it (when the shared server frees)
or another local model, and report agreement with Qwen3.8-27B. No human labels for now; the
limitation is stated.

## 2026-10-14: Third revision after the ARS re-review of v3

Make all three changes the re-review asked for (Mao, 2026-10-14, chosen from three options).
(1) Section 5.7, Table 4, the abstract and the introduction state the feasibility result for the
finite corpus: G at the largest sample and "n* exceeds N" replace the extrapolated values, and with
the finite population correction a corpus without repeated content needs about half of itself to
cover half. (2) Table 4 marks rows whose fit fails the diagnostic or whose prediction falls below
n = 100, and leads with the empirical crossings. (3) Two new runs: a and b fitted inside the
2,000-document pilot to re-score the 77/83 tally, and bootstrap intervals on r_same under each judge
carried into n*. Also correct Section 3.2, which said a and b are fitted on the pilot when the code
drew them from the whole corpus. Rationale: the re-review's critical item N2 and major items N1, N3, N4.

## 2026-10-14: Fourth revision after the ARS re-review of v4

Make all three changes the v4 re-review asked for (Mao, 2026-10-14, chosen from three options).
(1) Section 5.7 uses the formula again: on the corpora that pass the diagnostic and have an r_same
(AskDocs, TIFU, Truth Social, census random), the law fitted inside the pilot predicts G(n, r_same)
out to N/2 and is compared with the measured curve and the half-cover share. Fix the Truth Social
bound, which is an upper bound; give absolute n beside each share and state the 50% cap. (2) Treat
the judge as part of the definition of r: state on how many corpora the two judges' intervals are
disjoint, report the share at both ends of each r_same interval, mark censored crossings (TIFU
under Qwen, Bluesky). (3) One pass over the residual wording and housekeeping: Introduction and
Limitations wording on the pilot fit and the d check, per-seed range in 5.2, Zipf-free counts in the
main text, Truth Social and Bluesky citations, f2 for Chao and Jost, last sentence of 5.1, "a few
hundred" in the abstract, the Table 4 overfull box. Rationale: re-review items V1 to V8, N3, N5, N6,
N8, N9, N11, N12.

## 2026-10-14: Fifth revision after the ARS re-review of v5

Make every change the v5 re-review lists (Mao, 2026-10-14: "change it as they said"). (1) Report the
half-cover prediction by row type (censored below n = 100, at the 50% cap, interpolated inside the
pilot's fitted range, extrapolated), give the error on the gap below 50% for the extrapolated rows,
state the systematic overprediction and the worst error at the interval ends, and align the abstract.
(2) Reconcile 5.7 with the diagnostic: the failing corpora cross inside the fitted range; the
diagnostic governs extrapolation in n and transfer across r. Recalibrate r_same on pairs found inside
a 2,000-document pilot for at least two corpora, so the claim "before embedding the corpus" is
measured. (3) Housekeeping: no-crossing rates (TIFU under Qwen, Bluesky), TIFU r in 5.6, double comma
in 5.2, subsample dependence of the Truth Social share, per-corpus minimum of the 1.5 band and the
81 against 83 bootstrap cases, and the selection criteria and terms for the Truth Social and Bluesky
collections. Rationale: re-review items W1 to W6, V4, V5, N10.

## 2026-10-14: Data release for the Truth Social and Bluesky samples (Mao, interactive)

Release the post identifiers of the two 50,000-post samples, the code, and the pair judgments; do
not release post text. Rationale: common practice for social media research; others can rehydrate by
identifier. Stated in the Data section (re-review item W6).

## 2026-10-14: Sixth revision after the ARS re-review of v6

Make every change the v6 re-review lists (standing instruction from Mao, 2026-10-14: do what the
reviewers say, including the optional experiment). (1) Scope the pilot calibration claim and run the
experiment: d from the first pilot alone for every half-cover prediction, and pilot calibration
extended to AskDocs and TIFU (five corpora, Gemma). (2) Abstract states the grouping with its
accuracy and the judge dependence. (3) Housekeeping: censored rows at the point estimate, the census
covid clause, the Zipf clause, the Data pointer to Table 1, and the collection descriptions of the
Truth Social and Bluesky seeds. Space: the row-type breakdown moved to an appendix section.

## 2026-10-14: Collection terms for Truth Social and Bluesky (Mao, interactive)

The paper states that public posts were collected through the platforms' public interfaces and says
nothing about ethics review; terms and review are left to the ARR checklist at submission.

## 2026-10-14: Seventh revision after the ARS re-review of v7

Make every change the v7 re-review lists (standing instruction from Mao). Cut the abstract to the ACL
limit of 200 words; report the half-cover prediction with d, a and b from one pilot everywhere
(Table 2, 5.7, appendix); store the TIFU bound prediction in results/rcal_pilot.csv; state that
AskDocs and TIFU sit at the cap and test little, that the TIFU bound holds if a crossing exists, and
restore the census trump interval end in 5.7.

## 2026-10-14: Post to arXiv before ARR (Mao, interactive)

Post a signed preprint on arXiv now, ahead of the ACL 2027 ARR window (January 2027); ACL has allowed
non-anonymous preprints at any time since February 2024. Single author: Mao Li, University of
Michigan. Code and data go to a clean public repository (scripts, Truth Social and Bluesky post
identifiers, pair judgments; no post text, no review records or internal files), created private
first and made public after Mao checks it. Mao reads the paper in Listen before the upload.
Rationale: priority on applying Maurer's estimator and the n r^d law to text; the ARR version stays
anonymous.

## 2026-10-15: Ideas and facts recorded from the month retrospective

PROPOSAL: recorded ideas and facts, none decided by Mao; the source is notes/reviews/2026-10-15-month-retrospective/unrecorded-ideas.md in maospace.

- **Corpus as a semantic space that sampling can cover; manifold and metric-space framing.** Embeddings represent language in space, so a corpus can be wrapped by sampling and coverage. A 768-d space is not spanned by semantics: data lie on a low-dimensional manifold, which is the theoretical logic for sample points in a metric space; Maurer is a metric-space result, and the neighbor-substitute idea needs a KNN expression. The reader's question "If I read 15 texts, do I understand the corpus?" is the paper's frame (writing lesson 41). Voiced by Mao in Listen comments, 2026-10-14 (`review/listen/*.md`, Disposition empty). Quotes: "既然语言，语料，语意可以在空间上表示，那它就可以被这个用一种抽样，用一种覆盖的想法去包裹它"; "我们把G这个概念有点太奉为圭臬了"; "比如我抽了15个text，我能不能保证我基本上理解了所有的corpus？" (Item 107)
