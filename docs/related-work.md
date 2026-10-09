# Related work (checked 2026-10-08)

Search by a Sonnet agent over web search and arXiv, with the Maurer paper read in full from the
PDF. Items marked [U] came from snippets only and need verification before citation.

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
  with mathematical models from ecological research", J Clin Epidemiol 82:71-78. Ecology
  species models on open-ended survey answers. Full text not yet read.
- Chung, Haas, Upfal and Kraska (2018), "Unknown Examples and Machine Learning Model
  Generalization", arXiv 1808.08294. Species estimation for unseen training examples [U on
  estimator].
- Noorani et al. (2025), "Conformal Prediction Beyond the Seen: A Missing Mass Perspective",
  arXiv 2506.05497. Good-Turing on discrete LLM outputs.
- Pal, Bhattacharya and Singh (2026), "Blind-Spot Mass: A Good-Turing Framework for
  Quantifying Deployment Coverage Risk", arXiv 2604.05057. Discrete states with a support
  threshold; methods not read.

## Embedding-space coverage and diversity (describe the sample, no extrapolation)

- Kynkäänniemi et al. (2019), improved precision and recall, NeurIPS; Naeem et al. (2020),
  density and coverage, ICML; Han et al. (2023), rarity score, ICLR. All k-NN ball constructions
  against a reference set.
- Friedman and Dieng (2023), Vendi score, TMLR; Pasarkar and Dieng (2023), cousins of the
  Vendi score, arXiv 2310.12952 (Hill-number orders q).
- Pillutla et al. (2021), MAUVE, NeurIPS.
- Ba et al. (2026), "Measuring Dataset Diversity from a Geometric Perspective",
  arXiv 2602.09340 [U].

## Saturation in qualitative research and LLM coding

- Guest, Bunce and Johnson (2006), Field Methods; Hennink and Kaiser (2022), Soc Sci Med
  292:114523 (saturation at 9 to 17 interviews across 23 studies).
- Rowlands, Waddell and McKenna (2016), "Are We There Yet?" [U on venue and method].
- De Paoli and Mathis (2024), thematic saturation with LLM coding, arXiv 2401.03239: slope of
  cumulative unique codes, a heuristic.

## Sample size for topic models and LLM summarization

- Pham et al. (2024), TopicGPT, NAACL: topic count plateaus after about 600 documents;
  stopping rule of no new topic over 200 documents.
- Maeda 2025 "Balancing human and machine" (vault note
  `~/maospace/wiki/sources/syn-ppl/maeda-2025-balancing-human-machine.md`): 100 of 522
  documents recover over 60% of topics, 250 recover about 70%. The agent could not locate the
  paper online; confirm the citation from the vault note.

## Gaps the agent flagged

Chao et al. (2019) functional and attribute diversity (the ecology precedent for coverage with
distances between species) was not retrieved. iNEXT applied to NLP and 2026 arXiv work were
thin. A Semantic Scholar pass is still owed.
