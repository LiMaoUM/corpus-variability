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
