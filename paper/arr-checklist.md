# ARR Responsible NLP checklist answers (draft for the OpenReview form)

A. Limitations
- A1 Limitations section: Yes, the unnumbered "Limitations" section after the Discussion.
- A2 Risks: Yes, Data section (public posts, no post or account quoted or identified, text not released) and Limitations (political pools, not platform samples).

B. Use or creation of scientific artifacts
- B1 Cite creators: Yes. Qwen3-Embedding-8B (Section 4), GTR-T5-base (Section 4), Gemma-4-31B-it and Qwen3.8-27B (Section 5.6, Appendix prompts), SummEval forum corpora (Section 4).
- B2 Licenses: Yes. Qwen3-Embedding-8B, GTR-T5-base, Qwen3.8-27B and Gemma-4-31B-it are Apache 2.0 (checked on the Hugging Face hub, 2026-10-14).
- B3 Intended use: Yes. Models are used for embedding and pair judgment, within their licenses. The released post identifiers serve replication of the corpus-level results.
- B4 PII and offensive content: Yes. Data section: only embeddings and corpus-level statistics are analysed, no post or account is quoted or identified, URLs and mentions are removed, and post text is not released.
- B5 Documentation of artifacts: Yes. Data section (sources, languages, date range, selection criteria, deduplication).
- B6 Statistics: Yes. Data section and Table 1 (N per corpus).

C. Computational experiments
- C1 Parameters and compute: Yes. Model sizes in Section 4 and 5.6; GPUs and serving in the Prompts appendix.
- C2 Experimental setup and hyperparameters: Yes. Sections 3 and 5, Appendix B (grids, draws, fit ranges, thresholds).
- C3 Descriptive statistics: Yes. Intervals from bootstrap (Sections 5.3, 5.6, Table 2), number of draws per cell stated.
- C4 Existing packages: Yes. vLLM, NumPy and scikit-learn, cited in the Prompts appendix.

D. Human annotators
- D1 to D5: No human annotators. Pair judgments come from two language models, described in Section 5.6 and Appendix prompts.

E. AI assistants
- E1: The author(s) used an AI model to improve writing and coding.
