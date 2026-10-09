# Experiment Card: synthetic coverage

- **Study / version / date**: synthetic-coverage v1, 2026-10-08.
- **Hypothesis**: on a finite corpus with known topic mass, nearest-neighbour isolation coverage
  `C_r(n)` tracks true topic coverage across n (Spearman rho above 0.9, MAE below 0.05 for at
  least one label-free radius), and the Chao extrapolation from a 500-document pilot predicts
  the empirical n at which true coverage crosses 0.95 within a factor of two.
- **Sample / cohort definition (FROZEN)**: `src/corpus_variability/synthetic.py::make_corpus`,
  N = 20,000 documents, dim 64, four settings (50 flat topics; 200 Zipf(1) topics; 500 Zipf(1)
  topics; 200 Zipf(1) topics with noise 1.0), 3 seeds, nested samples at
  n in {50, 100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000}.
- **Conditions and metrics**: estimators true_coverage (target), good_turing_coverage (labelled
  ceiling), nn_coverage at r = alpha x median pilot pairwise cosine distance, alpha in
  {0.1, 0.2, 0.3, 0.4, 0.5}; Vendi score for n <= 2000. Agreement by Spearman rho and MAE over
  all (setting, seed, n) rows; required-n by median over seeds.
- **Compute plan**: CPU only, numpy and scikit-learn. No GPU.
- **Smoke test**: this card is the smoke test. `uv run scripts/smoke_synthetic.py`.
- **Stopping rule**: if no radius reaches rho 0.9 the metric as defined fails; report and
  redesign the radius rule (candidate: per-point adaptive r from k-th NN) before any real corpus
  run. Do not tune alpha on real corpora.
- **Artifacts**: `results/smoke_synthetic.csv`, `results/smoke_synthetic_required_n.csv`.
- **Monitor contract**: foreground run, under a minute expected; no monitor.
- **Status log**:
