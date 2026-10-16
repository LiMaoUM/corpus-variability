# corpus-variability

A metric for how much of a corpus's topical spread a sample covers, and how many documents a
sample needs to reach a target coverage. The survey analogy: a sample of a population gives a
point estimate with a margin of error; a sample of a corpus should give a coverage estimate with a
confidence band and a required-n.

## Standing decisions (see DECISIONS.md)

- Estimand: coverage of the finite corpus in hand. Sampling exists to save annotation and LLM
  budget. The full corpus is the ground truth for validation.
- Primary metric: nearest-neighbour isolation coverage, the continuous analogue of Good-Turing.
  `C_r(n) = 1 - #{i : d(x_i, NN_{-i}(x_i)) > r} / n` on document embeddings. Comparison lines:
  Vendi score convergence, discrete topics with iNEXT coverage-based rarefaction.
- Main line (2026-10-15): designed samples (k-means medoids, greedy maximum coverage) against random ones,
  with the isolation rate, the collapse law, the pilot test and the resolution reached as tools.
  Refined 2026-10-16: k-means selection is presented as existing practice with citations; the contribution is
  measuring its coverage, when it pays, and the pilot that predicts the gain. Method states the conditions for
  embedding the whole corpus and the fallbacks (embed a large random sample; else estimate with G (1 - n/N)).
  Earlier main line (2026-10-09): the corpus sample-size formula, n r^d collapse with TwoNN d plus the
  finite population correction on Maurer's G; collapse R2 is the corpus diagnostic. Venue:
  ACL/EMNLP via ARR. Two questions, two tools (2026-10-09): the formula answers coverage at
  resolution r, label-free, cross-corpus through d; theme coverage is a partition question
  answered by Good-Turing on theme masses, within corpus and universal for fixed k. Summary
  stability is a within-corpus direction only, a limitation.
- Validation corpora: synthetic GMM mixtures (known truth), the SummEval triple (AskDocs, Trump,
  TIFU; GTR caches in `~/projects/SummEval`, Qwen3 embeddings in `data/emb/`), the five 2024
  census corpora, and Truth Social and Bluesky as 50k-document samples each
  (`~/projects/social-media-corpus`).

## Data paths

- SummEval GTR-T5-base caches: `/home/maolee/projects/SummEval/revision/experiments/cache/gtr_{askdocs,trump,tifu}_{42,43,44}_{1000,10000}.npy`, shape (N, 768) float32, mean-pooled, max length 128. Text: `SummEval/data/AskDocs.csv` (column `Question`), `SummEval/data/variability/trump.csv` (`Message`), `SummEval/data/tifu_all_tokenized_and_filtered.json` (JSON lines, `selftext_without_tldr`). Loader and cleaning: `SummEval/revision/experiments/common.py::load_corpus`. The EACL coverage metric (`covg`, mean-max cosine to k probes) is in `revision/experiments/scaling_curve.py`.
- Old 2024 tweets: `from-windows-2026-07-26/data/random.csv` (Sprinklr census export) with `random_embeddings_encoder.csv` (BGE-base, 768-d, 7,860 rows, no header) and `mistral_random_embedding.csv` (4,096-d). Five corpora each have 100 repeated 50-tweet samples with LLaMa2 summaries (`*_summaries.csv`). Inventory and what carries over: `docs/archive-2024.md`; the 2024 code is branch `archive-2024` on the remote.

## Environment

- Python via `uv run`; deps in `pyproject.toml` (numpy, scipy, scikit-learn, pandas).
- Embedding model: `Qwen/Qwen3-Embedding-8B` (see `~/.claude/models.md`), run through this
  project's venv (torch 2.6 cu124, sentence-transformers >= 5, transformers >= 4.56; the
  SummEval venv is on transformers 4.49 and cannot load it). Shared HF cache:
  `HF_HOME=/home/model_cache/huggingface/` (set in `~/.bashrc`). Use `CUDA_VISIBLE_DEVICES`
  to pick a GPU; GPU 6 is the usual one with free memory, check `nvidia-smi` first.
- Embeddings for the census corpora land in `data/emb/` (gitignored, see
  `scripts/embed_census.py`).
- The box runs at load 60 to 80 on 64 cores; always set `OMP_NUM_THREADS=8` (the scripts do), or
  BLAS thread contention makes a one-minute job take fifteen.
- `from-windows-2026-07-26/` is 3.1 GB of 2023-2024 summarization data (50-sampled CSVs,
  covariance files, Mistral embeddings). Gitignored. Nothing in it is used yet.
- vLLM: `~/.local/bin/vllm` is broken (torch ABI mismatch, 2026-10-13). Serve with
  `/home/maolee/projects/arena/.venv/bin/vllm` (0.21.0), `--host 127.0.0.1`, TMPDIR=/home/maolee/tmp.
  After the 2026-10-13 reboot no shared servers ran; this project served Gemma-4-31B-it itself on
  GPUs 0,1, port 8840, for the r calibration cross-judge, and stopped it afterwards.
  On 2026-10-14 the pilot calibration (`scripts/rcal_pilot.py`, about 1,200 pairs) used the shared
  Gemma server on port 8801 at concurrency 8, because port 8800 had a queue and every GPU was busy.
- Experiment cards live in `experiments/`; results in `results/` (large arrays gitignored).

## Related work in the vault

- Backlog origin: `~/maospace/wiki/_corpus-sweep-2026-06-19-plan.md:65` (2025-07-28).
- SummEval reviewer item RR5 (coverage as heterogeneity grows):
  `~/projects/SummEval/revision/reviews/eacl_panel_rereview_round2_2026-06-12.md:52`.
- Maeda 2025 topic recovery (100 docs -> 60%, 250 -> 70% of 522):
  `~/maospace/wiki/sources/syn-ppl/maeda-2025-balancing-human-machine.md:23`.
- Overlap check: `~/maospace/map/CORPUS-PAPERS.md` holds only Truth Social and Bluesky
  conclusion papers; add a row once those corpora enter the validation set.

## Release and preprint

- Public release repo (created private 2026-10-14, made public only after Mao checks it):
  `~/projects/how-many-documents`, remote LiMaoUM/how-many-documents. Code, results CSVs, Truth Social
  and Bluesky post identifiers, pair judgments; no post text, no review records or internal files.
- arXiv version: `paper/arxiv.tex` (defines `\arxivversion`, signed, repo link); ARR version:
  `paper/main.tex` (anonymous). `paper/make_arxiv.sh` builds the source tarball.
- Listen: registered as key `corpus-variability` in `~/maospace/listen/papers.toml`.

## Agent room

Exchanges with Codex are archived in `docs/agent-room/` (see its README).

## Dimension units (found 2026-10-15)

- Every d in the paper and results (`d_twonn`, about 13 on AskDocs) is TwoNN on cosine distance. Cosine
  distance between unit vectors is half the squared Euclidean distance, so this d is half the geometric
  intrinsic dimension (TwoNN with Euclidean distance gives about 25 on AskDocs, 24 on Zipf, 4.2 on the
  d = 4 manifold, which uses Euclidean). The collapse law and every prediction are unaffected, because r is
  also a cosine distance, so n r_cos^(d_cos) = n r_euc^(d_euc) up to a constant. Prose that calls d "the number
  of independent directions" is off by a factor of two. Decided 2026-10-15: the paper reports the geometric d and writes the law
  as n r^(d/2) for cosine r.
- The package (`~/projects/how-many-documents/package`, `corpus_coverage`) reports the Euclidean d and uses
  exponent d / 2 with cosine r. With one pilot it reproduces the paper's pilot verdict on 9 of 10 Qwen3
  corpora (Census covid passes with pilot R2 0.966 against 0.85 in the paper).
