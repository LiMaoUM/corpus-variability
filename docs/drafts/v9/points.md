# Point lists, v9 (one plain sentence per paragraph)

## W1. Abstract (one paragraph, about 250 words, no corpus names, one big lesson)

1. If we read 15 documents from a corpus, how much of the corpus have we seen; researchers read samples because they cannot analyze everything (M1, M2).
2. A survey sample comes with a formula for its size and a margin of error, and a sample of text comes with neither.
3. We measure what a sample has missed as the percentage of the corpus with no document of the same content in the sample, tie "same content" to a distance between embeddings judged by large language models, and estimate the percentage from the sample alone with a classical estimator (D2 to D5).
4. The main lesson: a random sample is a poor use of a reading budget on corpora that repeat themselves; with the whole corpus embedded, reading one document from each of n groups of similar documents covers far more, and at the topic level it roughly halves what 15 random documents miss (G2, G3).
5. On corpora whose documents rarely share content, no choice of documents helps, and the isolation rate of a random pilot of 100 documents tells in advance which kind of corpus a researcher has (G4, G6).
6. To plan the size of a sample, we show that the estimator follows a law set by the intrinsic dimension of the corpus, which gives a sample size formula; in 77 of 83 tests the sample actually needed fell within two thirds to one and a half times the formula's answer, and a pilot of 2,000 documents shows whether the formula holds (D7, L2, D8).

## W1. Introduction (no result numbers, no corpus percentages, no calibrated distances; contributions stated clearly)

1. Researchers who study large collections of text work with samples because the collections hold more text than people or models can process (annotation teams, a large language model's context window, qualitative saturation after about 9 to 17 interviews); the size is set by budget or convention; if we read 15 posts from 10,000, have we understood the corpus, and would 100 or 1,000 be enough (M1, M2).
2. Survey research answers the same question for a population estimate with a sample size formula and a margin of error, and a sample of a corpus comes with neither; this paper builds both for the question a reader has in mind, how much of the corpus has a document like it among the documents we read.
3. To answer it we first say when a sampled document stands for a corpus document: two documents count as the same content when their embeddings lie within a distance r chosen by the researcher, and judgments of document pairs by large language models tell which distance means the same specific claim and which the same topic (D1, D2, D5).
4. With r fixed, the percentage of the corpus with no document within r in the sample is the missing mass, and Good and Turing's insight, carried to distances by Maurer, estimates it from the sample alone through the documents that are the only one of their kind in the sample (D3, D4); Maurer's paper is theory, and the estimator has not been used on text.
5. Every practice above draws its sample at random, yet embedding a whole corpus needs no human reading, so a researcher can choose which documents to read, much as a survey stratifies its population; whether that helps depends on how much the corpus repeats itself, since a document unlike every other can only be covered by reading it (G1, G4).
6. How the missing mass falls as the sample grows follows from a classical result about distances between random points: a larger sample and a looser resolution trade at a fixed rate set by the intrinsic dimension d, so that halving r requires 2^{d/2} times as many documents; d does for a corpus what the variance does for a survey population (D6, D7).
7. Contributions, as a numbered list without numbers: (a) a measure of what a sample of text has missed, estimated from the sample alone, with the finite population correction; (b) evidence that choosing documents with the whole corpus embedded covers far more than random sampling on corpora that repeat themselves, and that a small random pilot tells in advance how much choosing will save; (c) a sample size formula from the law and a pilot test that tells whether the formula can be trusted; (d) a way to give r a meaning in content by having large language models judge pairs. One sentence before the list says the corpora tested (C1).

## W2. Method subsections (two short subsections, placed after the subsection on calibrating the resolution)

Title A: "Choosing the documents to read"
1. A researcher who has embedded the whole corpus can choose the sample; we compare random samples with two designs (G1).
2. The k-means design: divide the corpus into n groups of similar documents and read the document nearest the centre of each group; any researcher can run it (G1, lloyd1982).
3. The greedy design: pick one document at a time that covers the most documents not yet covered; its coverage is at least 63% of the best possible, so it shows close to the most that any choice of n documents can cover (G1, nemhauser1978). With the whole corpus embedded, the missing mass of a chosen sample is computed exactly, without an estimator.

Title B: "The resolution a sample reaches"
1. Every sample covers the corpus at some resolution: r80(n), the distance within which 80% of the corpus has a sampled document; it is estimated from the sample alone as the r at which G (1 - n/N) equals 20%, and translated into content through the judged pairs at that distance (S1, D5).

## W3. Results subsections (replace the old "Coverage after 15, 100 and 1,000 documents" and add one)

Title A: "What a random sample covers" (Table 3 shows the missing mass after 15, 100, 1,000 random documents at both levels; Table 4 will show r80 and its meaning)
1. The corpora fall into two groups: on the diverse corpora a random sample covers almost nothing at the level of specific content, and on the repetitive ones 15 posts already show nearly every topic (R3, R4, C3).
2. Two concrete cases: 15 Census illegal tweets show 98% of the corpus a tweet on the same topic but only 38% one with the same point; 15 TIFU stories show 1.5% a story on the same topic, and 1,000 stories show 10% a story about the same event (R1, R2).
3. A reader learns which case applies from the sample itself: G (1 - n/N) from 15 documents is within 6 points of the truth everywhere and 1.6 points on average, with the illegal example 64% against 62% (R5).
4. Asked the other way, every random sample covers the corpus at some resolution, and the sample alone estimates that resolution to within 0.02; what it means in content varies: 15 documents cover 80% of a keyword Census slice at a distance where most judged pairs share a topic, but 80% of AskDocs or TIFU only at a distance where about one pair in five does (S1, S2).
5. At the level of specific content no random sample of up to 1,000 documents covers 80% of any corpus at a distance where most pairs are the same content (S3).

Title B: "Choosing the documents to read" (Table 5 will give random, k-means and greedy at 15 and 100 documents, and the sample that covers half of the corpus)
1. Choosing documents with the whole corpus embedded covers far more than random sampling wherever the corpus repeats itself (G2): give the illegal and Trump forum cases.
2. At the topic level the gain appears even on diverse corpora: 15 chosen AskDocs questions or Truth Social posts miss less than half of what 15 random ones miss (G3).
3. The sample that covers half of a repetitive corpus shrinks from hundreds of documents to a few dozen (G5).
4. Where documents rarely share content, no choice helps, because a document unlike every other is covered only by reading it (G4, G5 for AskDocs, TIFU, Truth Social).
5. k-means obtains most of what the near-best greedy choice obtains, so the gain is available with a standard tool (G7).
6. A random pilot of 100 documents tells in advance how much choosing will save: its isolation rate predicts the reduction (G6).

## W4. Discussion (what the study teaches; no repetition of results)

1. The question we started from (15 documents) has an answer that depends on the corpus and on the level of content the reader needs; 15 documents can show the topics of a repetitive corpus and show the specific content of no corpus we studied (M1).
2. The practical lesson: where the corpus can be embedded, do not read a random sample; read one document from each group of similar documents, because this covers far more where the corpus repeats and costs nothing where it does not (G2 to G7).
3. A short procedure: give r a meaning with judged pairs; draw a random pilot of 100 and compute G to learn which kind of corpus it is; choose the sample with k-means; when a sample size target is needed, fit the law on a pilot of 2,000 and check the pilot test; report the missing mass with the sample (computed exactly for a chosen sample, estimated by G (1 - n/N) for a random one).
4. The comparison with survey sampling at four points: the finite population correction; d in the role of the variance; choosing by k-means in the role of stratification; and the margin of error for proportions (the plain percentage in a random sample estimates corpus percentages within 0.3 to 7 points for fifty clusters) against the missing mass for content.
5. Coverage against a list of themes is a different quantity: the expected theme mass in a sample depends only on the sizes of the themes in the list (matched 1,200 samples from eight corpora within 0.4 points on average), whereas G needs no list.
6. Who can use the measure, and what comes next: a stopping rule for coding and topic discovery with large language models (pham2024topicgpt), a check that the documents behind a global summary of a corpus represent it (edge2024graphrag), a measure of how much of a reference set of human texts a set of generated texts covers (lebronnec2024); extensions to samples of accounts and to a human standard for r.
