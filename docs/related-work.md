# Related work (checked 2026-10-08)

Search by a Sonnet agent over web search and arXiv, with the Maurer paper read in full from the
PDF. Second pass on 2026-10-09 verified items at the arXiv HTML or publisher record; items still
marked [U] need the full text before citation.

## The construction already exists in theory

**Maurer (2022), "Concentration of the missing mass in metric spaces", arXiv 2206.02012.**
Defines the conditional missing mass at scale r as the probability that a new draw lies farther
than r from every sample point, and the extended Good-Turing estimator

    G(X, r) = (1/n) * #{k : d(X_k, X_i) > r for all i != k},

"the relative number of sample points, which are further than r from all other sample points"
(Section 2.1). Theorem 2.1 gives bias at most 1/n and variance bound 3/n for G minus its
target. This is exactly our nearest-neighbour isolation estimator. The paper is theory only:
no data, no embeddings, no text, no extrapolation to a larger sample, no required-n. Its
sketched applications are anomaly detection, nearest-neighbour coding, Wasserstein distance
and learning bounds.

Consequence for the paper: the estimator is cited to Maurer, and our contributions are
(a) the extrapolation layer (f1, f2 at scale r into Chao and Jost 2012 coverage-based
rarefaction, giving the required n for a target coverage), (b) the finite-corpus estimand with
validation against the full corpus, (c) the radius rule, and (d) the application to corpus
sampling for annotation and LLM budgets.

## Coverage estimators applied to collections

- Hajič and Moss (2025), "Knowing when to stop: insights from ecology for building catalogues,
  collections, and corpora", DLfM 2025, arXiv 2507.14614. Chao1 on discrete catalogue items
  (Gregorian chant). Closest applied Chao-on-a-collection paper. No embeddings, no required-n.
- Tran, Porcher, Tran and Ravaud (2017), "Predicting data saturation in qualitative surveys
  with mathematical models from ecological research", J Clin Epidemiol 82:71-78.e2
  (ScienceDirect pii S0895435616305431). Ecology species models on open-ended survey answers.
  Estimators and sample sizes still unread [U]; the publisher page needs a browser.
- Chung, Haas, Upfal and Kraska (2018), "Unknown Examples and Machine Learning Model
  Generalization", arXiv 1808.08294 (v2 2019, no venue). Chao92 with Good-Turing sample
  coverage; a species is a unique training example, counted within buckets of one feature.
  Verified at arxiv.org/html/1808.08294.
- Noorani et al. (2025), "Conformal Prediction Beyond the Seen: A Missing Mass Perspective",
  arXiv 2506.05497. Good-Turing on discrete LLM outputs.
- Pal, Bhattacharya and Singh (2026), "Blind-Spot Mass: A Good-Turing Framework for
  Quantifying Deployment Coverage Risk in Machine Learning Systems", arXiv 2604.05057
  (submitted to JMLR). Discrete label tuples (binned sensor states, ICD prefixes), exact
  matches, plug-in f1/n at the observed n, no radius, no embeddings, no extrapolation.
  Verified at the arXiv HTML.

## Embedding-space coverage and diversity (describe the sample, no extrapolation)

- Kynkäänniemi et al. (2019), improved precision and recall, NeurIPS; Naeem et al. (2020),
  density and coverage, ICML; Han et al. (2023), rarity score, ICLR. All k-NN ball constructions
  against a reference set.
- Friedman and Dieng (2023), Vendi score, TMLR; Pasarkar and Dieng (2024), "Cousins of the
  Vendi Score", AISTATS 2024, arXiv 2310.12952: Hill-number orders q, with the Vendi score
  as q = 1. Verified.
- Pillutla et al. (2021), MAUVE, NeurIPS.
- Ba, Abolhasani, Mancenido and Pan (2026), "Measuring Dataset Diversity from a Geometric
  Perspective", arXiv 2602.09340. PLDiv, persistence landscapes (H0) on sentence embeddings,
  Tevet and Berant (2021) sets, five encoders including Qwen3-Embedding-4B and 8B. Verified.

## Saturation in qualitative research and LLM coding

- Guest, Bunce and Johnson (2006), Field Methods; Hennink and Kaiser (2022), Soc Sci Med
  292:114523 (saturation at 9 to 17 interviews across 23 studies).
- Rowlands, Waddell and McKenna (2016), "Are We There Yet? A Technique to Determine
  Theoretical Saturation", Journal of Computer Information Systems 56(1):40-47, DOI
  10.1080/08874417.2015.11645799 (Crossref). Method unread [U]; cited as an interview
  saturation technique, so a richness estimator is unlikely.
- De Paoli and Mathis (2024), thematic saturation with LLM coding, arXiv 2401.03239: slope of
  cumulative unique codes, a heuristic.

## Sample size for topic models and LLM summarization

- Pham et al. (2024), TopicGPT, NAACL: topic count plateaus after about 600 documents;
  stopping rule of no new topic over 200 documents.
- Maeda, Wang, Zhang, Banks and Kenney (2025), "Balancing human and machine coding:
  evaluating the credibility and potential of topic modeling for open-ended survey responses",
  Computers in Human Behavior (Zotero key IRMW6N5N, read at full text in the vault note
  `~/maospace/wiki/sources/syn-ppl/maeda-2025-balancing-human-machine.md`). An 8-topic LDA
  on 522 open-ended responses; with responses of at least 15 meaningful words, 100 documents
  recover over 60% of the full-corpus topics and 250 recover about 70% at cosine threshold
  0.5. Topic recovery is the discrete, label-dependent version of our coverage curve.

## Ecology precedent for a distance threshold

Chao, Chiu, Villéger, Sun, Thorn, Lin, Chiang and Sherwin (2019), "An attribute-diversity
approach to functional diversity, functional beta diversity, and related (dis)similarity
measures", Ecological Monographs 89(2):e01343. Two species at trait distance of at least tau
count as fully distinct, closer pairs as partly redundant; attribute diversity is a Hill number
of order q giving the effective number of equally distinct virtual functional groups; reported
as a tau profile. This is the distance-threshold richness construction in ecology, and the
natural citation for treating r as a profile parameter. Whether it defines coverage or
rarefaction at a given tau is unchecked (abstract only).

## Still open

No 2024 to 2026 paper applying Good-Turing or Chao coverage to nearest-neighbour distances in
embedding space was found; Blind-Spot Mass (discrete states) and Chung et al. (2018) are the
nearest. On "how many posts" for a social media corpus, practice is ad hoc: arXiv 2606.04450
(a Reddit study) justified 250 posts from interview-saturation literature and 384 from
Cochran's formula. Weller et al. (2018, PLoS ONE) report a median of 75 interviews to
saturation. Tran 2017 and Rowlands 2016 still need the full text.
