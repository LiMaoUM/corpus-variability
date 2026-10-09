# Handoff (2026-10-09, session cut by usage limit)

## Running in the background when the session ended
- `summarize_draws.py` for the five census corpora (750 summaries) and the SummEval triple
  (450), followed automatically by `summary_n_curve.py --step both` (embeds summaries on GPU 6,
  then analysis). Logs: `results/summarize_census.log`, `results/summarize_summeval.log`,
  `results/summary_n_curve.log`; outputs `results/summary_n_curve.csv`, `_draws.csv`.
  When it ends: apply the stopping rule in `experiments/summary-n-curve.card.md` and append a
  status-log line with the verdict. Card expires 2026-10-10 12:00.
- `replicate_seeds.py` (CPU): claims 1 and 2 on GTR seeds 42/43/44 plus the Trump/Zipf
  collapse probe. Log `results/replicate_seeds.log`; outputs `results/replicate_seeds.csv`,
  `results/collapse_probe.csv`. When it ends: add the seed means to `docs/eureka-evidence.md`.

## Not done
- Full-text verification of the [U] items in `docs/related-work.md` (Tran 2017, Rowlands 2016,
  Chung 2018, Blind-Spot Mass 2026, Chao 2019, Maeda 2025, Ba 2026, Pasarkar 2023): the web
  tools hit the session limit (resets 11:00 ET). Rerun the same nine checks.

## For Mao
- Thread, venue, title, and which claim the paper is built on.
- Truth Social / Bluesky embedding (needs a GPU card and a CORPUS-PAPERS.md row first).
- Whether the EACL camera-ready should cite the collapse diagnostic in reply to RR5.
