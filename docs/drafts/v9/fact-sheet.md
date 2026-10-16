# Fact sheet, v9 main line (verified 2026-10-15 against results/*.csv and the paper tables)

Venue and reader: ACL / EMNLP via ARR, long paper. Reader: an NLP or computational social science
researcher who samples text for annotation, summarization or qualitative coding; not a statistician.
Length: write what the reader needs to follow once; the page limit is set aside at this stage (Mao).
The reader does not need: model serving details, bootstrap details, our revision history, reviewer
objections nobody raised.

## Mao's own words (translated from Chinese, use as the frame)

- M1. "If I sample 15 texts, can I be sure I basically understand the corpus? Can I be sure I understood this corpus?"
- M2. On why people sample: it is a practical constraint, the capacity to analyze is limited. (Do not say researchers "assume" the sample stands for the rest.)
- M3. On the abstract: give one big lesson; do not name corpora the reader has not met.
- M4. On the introduction: state the contribution clearly; details such as calibrated distances and specific percentages go to Results.
- M5. On the old conclusion: "Maybe the pessimistic conclusion is because of how we posed it. Is there a more suitable way to compute it?" The answer that became the main line: random sampling is not the best use of a reading budget; choosing documents with the whole corpus embedded covers far more.

## Definitions (already fixed in the paper)

- D1. Embedding: a text encoder (Qwen3-Embedding-8B) maps each document to a point so that documents with similar meaning lie close together. Distance between two documents = cosine distance between their embeddings (0 = same meaning, near 1 = unrelated).
- D2. Resolution r: a distance chosen by the researcher; two documents closer than r count as the same content.
- D3. Missing mass at r: the percentage of the corpus that has no document within r in the sample. Coverage = 100% minus missing mass.
- D4. Isolation rate G (Maurer 2022, carrying Good and Turing's estimator from words to distances): the percentage of sampled documents that have no other sampled document within r, that is, documents that are the only one of their kind in the sample. G (1 - n/N) estimates the missing mass from the sample alone; 1 - n/N is the finite population correction of survey sampling, needed because sampled documents cover themselves.
- D5. Judged distances: two large language models, Qwen3.8 and Gemma 4, judged about 4,000 pairs of documents at a range of distances as the same specific content (the same claim, question, event or story), the same topic, or unrelated. r_same = the distance at which half of the judged pairs are the same content; r_topic = the distance at which half are the same content or the same topic.
- D6. Intrinsic dimension d: the number of independent directions along which documents in the embedding space vary, estimated by TwoNN (Facco et al. 2017) from the ratio of each document's distances to its first and second nearest neighbours. About 25 to 32 on the corpora where the law holds (AskDocs 25.5, Truth Social 25.3, Census random 25.7, TIFU 27.3, Bluesky 32.0); 18 to 23 on the corpora where it fails.
- D7. Collapse law: G depends on n and r only through n r^{d/2} for cosine distance r (cosine distance is half the squared Euclidean distance between unit embeddings, hence d/2). Meaning: a larger sample and a looser resolution trade at a fixed rate; halving r requires 2^{d/2} times as many documents, about 8,000 times for d = 26.
- D8. Sample size formula n*: the law solved for n. Pilot test: fit the law on subsamples of a pilot of 2,000 documents and check it on larger subsamples of the same pilot; a corpus passes when the check explains at least 95% of the variance.

## Corpora

- C1. Ten real corpora: three Reddit forums of 10,000 documents each (AskDocs, medical questions; Trump, posts about the 2020 presidential election; TIFU, first-person stories of how people messed something up); five slices of 2020 tweets about the United States Census, four selected by a keyword (citizens, trump, covid, illegal) and one drawn without a keyword (random), 7,610 to 30,136 tweets; random samples of 50,000 English posts each from Truth Social and Bluesky, May to December 2024. Plus synthetic data with a known answer.
- C2. Census illegal slice: 17,202 tweets. TIFU: 10,000 stories.
- C3. Two groups of corpora. Repetitive: the Trump forum and the Census slices selected by a keyword (citizens, trump, illegal, covid), which repeat a limited number of claims. Diverse: AskDocs, TIFU, Truth Social, the random Census slice, whose documents rarely share specific content.

## Results: random samples (Table 3, Qwen3.8 judge, true values over 20 samples)

- R1. 15 random Census illegal tweets: 98% of the corpus has a tweet on the same topic in the sample; 38% has one that makes the same specific point.
- R2. 15 random TIFU stories: 1.5% of the corpus has a story on the same topic in the sample; 1,000 stories: 10% has a story about the same event.
- R3. Diverse corpora at the level of specific content: 15 random documents leave more than 99% of the corpus with nothing like it in the sample; 1,000 leave 84% to 96%.
- R4. Repetitive corpora: 15 random posts leave 1.8% to 11% of the corpus without a post on the same topic, and 62% to 90% without one with the same specific content; 1,000 posts leave 16% to 42% at the content level.
- R5. Estimate from the sample alone: G (1 - n/N) from 15 sampled documents lies within 6 percentage points of the true value on every corpus and at both levels, 1.6 points on average; within 2 points with 100 documents. Example: 15 Census illegal tweets, estimate 64%, truth 62%.

## Results: the resolution a random sample reaches (new, results/resolution_reached.csv)

- S1. Every sample covers the corpus at some resolution. Define r80(n): the distance at which 80% of the corpus has a sampled document within that distance. Estimated from the sample alone (the r at which G (1 - n/N) = 20%), it is within 0.02 of the true value on every corpus at n = 15, 100 and 1,000.
- S2. What r80 means in content, as the percentage of judged pairs at that distance that share at least a topic: with 15 random documents, 65% to 100% on the Census keyword slices and the Trump forum (83%), 19% on AskDocs, 18% on TIFU, 36% on Truth Social; with 1,000 documents, 95% to 100% on the keyword slices, 70% on AskDocs, 72% on Truth Social, 39% on TIFU.
- S3. At the level of specific content, the percentage of pairs judged the same content at r80 never exceeds 55% on any corpus, even with 1,000 documents.

## Results: designed samples (new, results/designed_samples.csv)

- G1. Designs: (a) k-means: divide the whole embedded corpus into n groups of similar documents with k-means (Lloyd 1982) and read the document nearest the centre of each group; (b) greedy maximum coverage: pick, one at a time, the document that covers the most documents not yet covered; it is guaranteed to cover at least 63% (1 - 1/e) of what the best possible set of n documents covers (Nemhauser et al. 1978), and we use it as a near-best bound. Both need the whole corpus embedded, which needs no human reading and no calls to a large language model.
- G2. Percent of the corpus missed at the same-content level, 15 documents, random / k-means / greedy: Census illegal 63 / 33 / 29; Trump forum 88 / 64 / 60; Census citizens 90 / 65 / 61; Census trump 87 / 59 / 56; Census covid 98 / 87 / 84. With 100 documents: Census illegal 35 / 21 / 17; Trump forum 68 / 51 / 44.
- G3. Percent missed at the topic level, 15 documents, random / k-means / greedy: AskDocs 58 / 24 / 18; Truth Social 64 / 30 / 26; Census random 16 / 8 / 5; TIFU 98.5 / 92 / 87 (random values as in Table 3). With 1,000 TIFU stories: 57 / 46 / 34.
- G4. Where nothing repeats, design cannot help: AskDocs at the same-content level, 15 documents 99.8 / 99.8 / 99.5, 1,000 documents 89.9 / 89.6 / 89.3. A document with no other document of the same content in the corpus can only be covered by reading it.
- G5. Sample that covers half of the corpus at the same-content level, random against greedy: Census trump 604 against 33 of 30,136; Trump forum 442 against 50 of 10,000; Census citizens 370 against 39 of 7,849; Census covid 3,356 against 1,262 of 17,837; Census illegal under 100 against 3. On AskDocs, TIFU and Truth Social the two are close: 4,953 against 4,933 of 10,000; 4,997 against 4,998 of 10,000; 21,942 against 20,015 of 50,000.
- G6. A pilot tells in advance how much a design saves. Across 17 combinations of corpus and level (same content, topic), the isolation rate G of a random sample of 100 documents predicts the share of missing mass that k-means removes at n = 100: Spearman rank correlation -0.90. Where G of the random sample is 0.99 or more, k-means removes 0% to 2% of the missing mass; where G is 0.7 or less, it removes 26% to 67%.
- G7. k-means, a design any researcher can run, obtains most of what the near-best greedy design obtains (G2, G3).

## Results: the law, the formula, the pilot test (existing, unchanged)

- L1. On 10 of 17 pairs of corpus and encoder, G depends on n and r only through n r^{d/2}; a fit on samples up to 1,000 documents predicts G for samples seven times larger.
- L2. Formula test: 83 cases (one per combination of corpus, target isolation rate and distance at which the true answer lay beyond 1,000 documents). In 77 of 83, the sample actually needed fell between two thirds and one and a half times the formula's answer; an answer of 1,500 meant 1,000 to 2,250.
- L3. The law fails on seven pairs: five political corpora (the Trump forum under two encoders and Census keyword slices), and a synthetic mixture of topics; a pilot test on 2,000 documents separates passing from failing pairs before the full sample is drawn.
- L4. Comparison: the coverage extrapolation of Chao and Jost (2012), standard in ecology, predicts G for a sample twice as large with an average error two to nine times that of the law.

## Literature (in the bibliography; keys in brackets)

- Saturation in qualitative research: about 9 to 17 interviews [guest2006, hennink2022].
- Topic discovery with a large language model: TopicGPT stops when 200 further documents add no new topic [pham2024topicgpt].
- Global questions over a whole corpus (Graph RAG, query-focused summarization) [edge2024graphrag].
- Precision and recall for the diversity of large language model output [lebronnec2024].
- Survey sample size and margin of error [cochran1977, kish1965]; Good and Turing [good1953]; Maurer [maurer2022missing]; TwoNN [facco2017twonn]; k-means [lloyd1982]; greedy coverage guarantee [nemhauser1978]; core sets [sener2018coreset]; Chao and Jost [chao2012coverage].
- No published work applies Maurer's estimator to text (checked 2026-10-15).
