# IJCAI 2027 version: structure, points and numbers for Mao to write from

Decided 2026-11-19: main line is the sample size formula; Mao writes the body; Claude polishes, typesets and
checks numbers. Limit: 7 body pages including figures and tables (about 4,600 words of prose with the three floats
below), plus up to 2 pages of references. Everything else goes to the supplementary material (PDF, anonymized).
Each line below is one paragraph: its point, then the numbers it may use, with the place in the current
paper where they are stated and checked. Word budgets are targets for fitting 7 pages.

## Floats kept in the body

- Figure 1 (`fig:collapse`): isolation rate against n r^(d/2), six panels. Full width.
- Table 1 (`tab:collapse`): the law on 17 corpus and encoder pairs (d, out of sample R2, pilot R2, error).
- Table 2 (`tab:resolution`): r_same per corpus and the random sample that covers half the corpus.
- To the supplement: Figure spread, Table design, Table reader, Table rcal, all appendix tables.

## Abstract (about 180 words)

1. If we read 15 documents from a corpus, how much of it have we seen; samples are read because nobody can read everything.
2. Survey sampling plans a sample with a formula and a margin of error; text sampling has neither.
3. We estimate the percentage of the corpus with no document of the same content in the sample, from the sample alone (isolation rate G with the finite population correction; Good and Turing, carried to distances by Maurer).
4. The main result: G follows a law set by the intrinsic dimension d of the corpus, which gives a sample size formula; in 77 of 83 tests the sample actually needed was within two thirds to one and a half times the formula's answer.
5. A pilot of 2,000 documents tells whether the formula holds for a corpus; it holds on ten of 17 corpus and encoder pairs and fails on political corpora that repeat slogans.

## 1 Introduction (about 650 words)

1. Researchers sample text because they cannot analyze all of it (annotation budgets, a model's context window; qualitative saturation after about 9 to 17 interviews [guest2006, hennink2022]); the size is set by budget or habit.
2. A survey sample comes with a sample size formula and a margin of error [cochran1977, kish1965]; a text sample does not, and the question a reader has is how much of the corpus has something like it in the sample.
3. Our answer, in three steps: a measure (missing mass at resolution r, estimated by G (1 - n/N)); a law (a larger sample and a looser resolution trade at a fixed rate, so G is one curve in n r^(d/2)); a formula for n and a pilot test of whether it can be trusted.
4. Contributions, as a short list: (a) the estimator on text with the finite population correction; (b) the collapse law with d in the role the variance plays in a survey; (c) the formula and the pilot test; (d) resolutions with a content meaning from judged pairs.

## 2 Background (about 450 words)

1. Good and Turing estimate the unseen mass of words from singletons [good1953]; Maurer carries this to distances [maurer2022missing]; no published work applies Maurer's estimator to text (checked 2026-10-15).
2. Coverage extrapolation in ecology [chao2012coverage] predicts coverage from singleton and doubleton counts; we compare against it.
3. Intrinsic dimension by TwoNN [facco2017twonn]; distances between random points scale with n^(1/d), which is where the law comes from.
4. Work that needs a sample size for text: topic discovery that stops when 200 documents add no topic [pham2024topicgpt], global summaries of a corpus [edge2024graphrag].

## 3 Method (about 1,000 words)

1. Embedding and distance (Qwen3-Embedding-8B, cosine distance); resolution r; missing mass M(r) = percentage of the corpus with no sampled document within r. (method.tex, background.tex eq:G)
2. Isolation rate G = percentage of sampled documents with no other sampled document within r; G (1 - n/N) estimates M; the factor 1 - n/N because sampled documents cover themselves. (eq:fpc)
3. The collapse law: a larger sample and a looser resolution trade at a fixed rate, so that G falls on one curve in n r^(d/2) for cosine r; d the geometric intrinsic dimension; fitted form log(-log G) = a + b log(n r^(d/2)). (eq:collapse; d/2 because cosine distance is half the squared Euclidean distance)
4. The formula: solve the law for n at a target G; report n* with the band two thirds to one and a half times n*. (eq:nstar)
5. The pilot test: fit on subsamples of a 2,000-document pilot, check on larger subsamples of the same pilot; pass at R2 of 0.95 or more.
6. Giving r a content meaning: a large language model (Qwen3.8; Gemma 4 as second judge) judges pairs at a range of distances; r_same is where half the pairs are the same specific content. One paragraph; details to the supplement.

## 4 Data (about 200 words)

1. Ten real corpora: three Reddit forums of 10,000 (AskDocs, Trump, TIFU), five slices of 2020 Census tweets (7,610 to 30,136), random samples of 50,000 posts from Truth Social and Bluesky; synthetic manifolds and topic mixtures with known answers; two encoders (Qwen3-Embedding-8B, GTR). 17 corpus and encoder pairs. (data.tex)

## 5 Results (about 1,700 words)

1. G tracks true coverage: on synthetic Zipf mixtures, rank correlation 0.968 with true topic coverage, off by 2.9 points on average. (results.tex 5)
2. The law holds on ten of 17 pairs (three manifolds, AskDocs and TIFU under both encoders, Census random, Truth Social, Bluesky): a fit on samples up to 1,000 explains 92% or more of the variance in samples up to 7,000; average error at the largest sample 1.5 to 4.2 points. Figure 1, Table 1. (sec:collapse)
3. The law beats the alternatives: separate curves per resolution have 1.5 to 3.8 times its error on nine of ten passing pairs; Chao and Jost's extrapolation has 2 to 9 times its error. (sec:collapse)
4. The formula: 77 of 83 cases within two thirds to one and a half times n* (76 of 83 with a, b fitted in one pilot of 2,000); on failing pairs 10 of 23. Example: an answer of 1,500 meant 1,000 to 2,250. (sec:nstar)
5. The pilot test sorts every pair as the full test does: passing pairs reach a pilot R2 of 0.96 or more, failing pairs at most 0.93; where it fails (Trump forum) slogan clusters of dimension near 2 sit beside discussion clusters near 16. (sec:diag)
6. The finite population correction: without it G is off by 11 points on average and up to 49; with it 1.2 on average, at most 11 (144 cases). (sec:fpc)
7. What a sample means in content: r_same ranges from 0.05 (TIFU) to 0.28 (Census illegal); the random sample covering half of a corpus at r_same, Table 2. (sec:rcal, tab:resolution)
8. One paragraph on choosing documents: where a corpus repeats itself, reading the document nearest each k-means center misses far less than a random sample (15 Census illegal tweets: 33% against 63% at the content level); a random pilot of 100 predicts the gain (Spearman -0.90); details in the supplement. (sec:design-results)

## 6 Discussion and limitations (about 550 words)

1. The answer to "15 documents": depends on the corpus and the level of content; the formula and the pilot let a researcher plan n before reading.
2. The survey analogy at three points: d in the role of the variance, the finite population correction, the margin of error replaced by the band on n*.
3. Limitations: the law fails on repetitive political corpora (the pilot catches it); the 0.95 threshold was set on these corpora; r_same depends on the judge model, which is part of the definition of r.
