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
