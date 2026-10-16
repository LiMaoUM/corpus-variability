# Literature scan: who would use or cite the missing mass measure (2026-10-15)

Delegated scan (Sonnet subagent), each paper opened on its arXiv page by the agent. Semantic Scholar's
citation API was rate limited, so the list of papers citing Maurer (2022) was not checked; check it by hand
before submission.

## Bottom line

- No paper found applies Maurer's estimator to text, fits a coverage or sample size law with intrinsic
  dimension to text embeddings, or estimates the missing mass of a reference text distribution with a
  nearest neighbour or Good-Turing estimator. No scoop found.

## Synthetic data and generation diversity

1. Le Bronnec et al. Exploring Precision and Recall to assess the quality and diversity of LLMs. ACL 2024. https://arxiv.org/abs/2402.10693 (k-NN precision and recall for text; G is a label-free recall at radius r with a required n.)
2. Farnia, Jalali, Ospanov. Exposing Diversity Bias in Deep Generative Models. arXiv 2602.14682, 2026. https://arxiv.org/abs/2602.14682 (Vendi and RKE scores rise with sample size; motivates a metric with a stated n dependence.)
3. Avestimehr, Duffy, Medard. NOVA: Fundamental Limits of Knowledge Discovery Through AI. arXiv 2605.15219, 2026. https://arxiv.org/abs/2605.15219 (Good-Turing unseen mass over a finite discrete domain with a Zipf scaling law; the discrete counterpart. Read in full before submission.)
4. Wong et al. SimpleStrat. arXiv 2410.09038. https://arxiv.org/abs/2410.09038 (recall of reference answers on CoverageQA.)
5. Liu et al. No Single Best Model for Diversity. COLM 2026, arXiv 2604.02319. https://arxiv.org/abs/2604.02319
6. Jung et al. Prismatic Synthesis (G-Vendi). arXiv 2505.20161. https://arxiv.org/abs/2505.20161
7. Amballa et al. VOYAGER. ACL 2026, arXiv 2512.12072. https://arxiv.org/abs/2512.12072
8. Li, Li, Qiu. ChatGPT is not A Man but Das Man (silicon samples homogenized relative to ANES). arXiv 2507.02919. https://arxiv.org/abs/2507.02919

## Corpus-level summarization and sensemaking

1. Edge et al. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv 2404.16130. https://arxiv.org/abs/2404.16130
2. Kim et al. ReTAG. EMNLP 2025 Findings, arXiv 2509.25814. https://arxiv.org/abs/2509.25814
3. Lu et al. CorpusQA. arXiv 2601.14952. https://arxiv.org/abs/2601.14952
4. Pham et al. TopicGPT. NAACL 2024, arXiv 2311.01449. https://arxiv.org/abs/2311.01449
- Gap: no paper tests whether the sample used for global sensemaking represents the corpus.

## Sample size and stopping for LLM-assisted coding

1. De Paoli, Mathis. Inductive Thematic Saturation with LLMs. arXiv 2401.03239. https://arxiv.org/abs/2401.03239
2. Tutar, Erden, Senturk. Q-Sat AI. arXiv 2511.01935. https://arxiv.org/abs/2511.01935
3. Camuffo et al. Variance-Aware LLM Annotation for Strategy Research. arXiv 2601.02370. https://arxiv.org/abs/2601.02370
- Gap: no stopping rule based on embedding coverage.

## Scoop check

- Ruppik et al. Less is More: Local Intrinsic Dimensions of Contextual Language Models. NeurIPS 2025, arXiv 2506.01034 (local ID as a training signal; no coverage law).
- Maurer. Concentration of the missing mass in metric spaces. arXiv 2206.02012 (applications: anomaly detection, coding, learning bounds; no text).
