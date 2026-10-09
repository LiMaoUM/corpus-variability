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
- Main line (2026-10-09): the corpus sample-size formula, n r^d collapse with TwoNN d plus the
  finite population correction on Maurer's G; collapse R2 is the corpus diagnostic. Venue:
  ACL/EMNLP via ARR. Summary stability is a within-corpus direction only, a limitation.
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
- Experiment cards live in `experiments/`; results in `results/` (large arrays gitignored).

## Related work in the vault

- Backlog origin: `~/maospace/wiki/_corpus-sweep-2026-06-19-plan.md:65` (2025-07-28).
- SummEval reviewer item RR5 (coverage as heterogeneity grows):
  `~/projects/SummEval/revision/reviews/eacl_panel_rereview_round2_2026-06-12.md:52`.
- Maeda 2025 topic recovery (100 docs -> 60%, 250 -> 70% of 522):
  `~/maospace/wiki/sources/syn-ppl/maeda-2025-balancing-human-machine.md:23`.
- Overlap check: `~/maospace/map/CORPUS-PAPERS.md` holds only Truth Social and Bluesky
  conclusion papers; add a row once those corpora enter the validation set.

## Agent room

Exchanges with Codex are archived in `docs/agent-room/` (see its README).
