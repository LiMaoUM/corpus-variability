# Experiment Card: calibrating the resolution r

- **Study / version / date**: r-calibration v1, 2026-10-11.
- **Hypothesis**: the probability that two documents are judged to be about the same specific
  content falls with their cosine distance in Qwen3 space along a curve that is similar across
  corpora, so a user can choose r as the distance at which that probability crosses a chosen
  level (for example 0.5), and the curve gives r a meaning that does not depend on the corpus.
- **Sample / cohort definition (FROZEN)**: ten corpora (five census slices, three forum corpora
  under Qwen3, Truth Social, Bluesky). Per corpus, 40 document pairs in each of ten cosine
  distance bins from 0 to 1 (near pairs from the 20 nearest neighbours of 3,000 random anchors,
  far pairs at random), seed 5, documents truncated to 800 characters. About 4,000 pairs.
- **Conditions and metrics**: Gemma-4-31B-it at temperature 0 answers SAME, TOPIC or UNRELATED
  (prompt in `scripts/r_calibration.py`). Per corpus and bin: share SAME and share SAME or
  TOPIC. The r at which share SAME crosses 0.5 per corpus, and its spread across corpora.
- **Compute plan**: about 4,000 short prompts on the running vLLM server (GPUs 4, 5),
  concurrency 8; from the enumeration run's throughput, under 15 minutes.
- **Smoke test**: `uv run scripts/r_calibration.py --smoke` (citizens, 40 pairs): parse rate of
  the one-word answer, distribution of labels.
- **Stopping rule**: this is a calibration; it is reported whatever the curve looks like. If the
  parse rate is below 95% the prompt is fixed once and the smoke test repeated.
- **Artifacts**: `data/rcal/<corpus>.jsonl`, `results/r_calibration.csv`, log
  `results/r_calibration.log`.
- **Monitor contract**: background; report on completion or error only, leading with the r at
  which P(SAME) crosses 0.5 per corpus. Expires 2026-10-12 12:00.
- **Status log**:
  - 2026-10-13 v1: ran with Qwen3.8-27B (Gemma server congested; Mao's decision). 3,913 pairs over ten corpora. P(SAME) crosses 0.5 at r = 0.05 (TIFU), 0.15 (AskDocs), 0.16 (Truth Social), 0.22 to 0.30 (census), 0.24 (Trump forum); Bluesky never reaches 0.5 (0.42 below 0.1). Hypothesis of a corpus-independent curve rejected; per corpus calibration on a few hundred pairs is the recommended step. Results: results/r_calibration.csv.
  - 2026-10-13 v2 (correction): the labels for census random, trump, covid and illegal had been
    written by Gemma-4-31B-it before the switch to Qwen3.8-27B, and the resumable runner kept
    them. The four were moved to `data/rcal/gemma_initial/` and re-judged by Qwen3.8-27B on the
    same pairs (seed 5). New Qwen crossings: random 0.12 (was 0.28), census trump 0.24 (0.32),
    covid 0.20 (0.29), illegal 0.275 (0.30). Cross-judge with Gemma-4-31B-it on all 3,913 pairs
    (`scripts/rcal_crossjudge.py`, `results/rcal_crossjudge.csv`): agreement 0.71, kappa 0.55,
    kappa 0.68 for SAME vs rest; Gemma labels SAME on 39% of pairs against 28%, so its crossing is
    0.02 to 0.16 farther out in every corpus. n* at both crossings in `results/nstar_rsame.csv`
    (`scripts/nstar_rsame.py`): AskDocs, TIFU, Truth Social infeasible under both judges; Trump
    forum, census citizens, trump, illegal feasible under both (8 to 600 documents); census random
    flips (7e7 under Qwen, 1,200 under Gemma); covid 16,000 vs 350. Empirical crossings within
    a factor of 1.6 of the prediction at eps 0.5 where the grid reaches them.
