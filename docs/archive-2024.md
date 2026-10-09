# What the 2024 archive contains and what carries over

Branch `archive-2024` on `LiMaoUM/corpus-variability` holds the April to October 2024
exploration (three notebooks, one generation script, wandb logs). Its data lives locally in
`from-windows-2026-07-26/data/` (3.1 GB, gitignored). Reviewed 2026-10-09.

## Methods, and their status

- **Covariance spectrum of sample versus corpus** (`experiment.ipynb`). For the random and
  citizenship tweet corpora: 100 draws of 50 tweets, covariance of each draw against the full
  corpus, eigenvalue spectra normalised by the top eigenvalue; projection of the LLM-summary
  covariance onto corpus eigenvectors as a per-direction ratio. Carries over as a comparison
  line: a second-moment measure of spread (participation ratio of the spectrum is a linear
  dimension estimate to set against TwoNN; the spectrum ratio is a Vendi-type quantity). It is
  the proportion side of claim 2: moments are matched at n = 50, coverage is not.
- **Eigenvector-to-sentence reconstruction and FastLexRank typicality**
  (`sanity_check.ipynb`). The seed of vec2summ and FastLexRank, both published through
  SummEval. Nothing to carry over.
- **Embedding-to-text decoder fine-tuning** (`finetune_decoder.ipynb`, Mistral-7B LoRA and a
  from-scratch transformer decoder). Superseded by vec2text. Nothing to carry over.

## Data that carries over

Sprinklr tweet export on the 2020 US Census (`random.csv`, 14,323 rows, Sept 2024 extract) and
keyword slices of the same pool: `full_citizenship_question.csv` 15,030, `trump.csv` 30,180,
`covid.csv` 17,849, `illegal.csv` 17,237, `biden.csv` 2,497 (line counts; CSV fields may wrap).
Embeddings: `random_embeddings_encoder.csv` (BGE-base-en-v1.5, 768-d, 7,860 rows after
dedupe), `mistral_*_embedding.csv` (4,096-d).

**Repeated-sample summaries.** For random, citizens, trump, covid and illegal: 100 independent
draws of 50 tweets, each summarised by LLaMa2TS-0.1.0 (a LLaMa-2 fine-tune on
`/nfs/turbo/isr-fconrad1/model/`), one summary per row in `*_summaries.csv` (100 rows each;
the trump file carries a `_header_id|>` artefact and tweet-index citations in parentheses).
This is a ready-made downstream-stability measurement at n = 50: the spread of summaries
across draws is the quantity claims 1 and 2 should predict from G(50, r) and d. Planned use:
embed the 100 summaries per corpus, take mean pairwise cosine distance as summary
instability, and compare across the five corpora against their missing mass at n = 50 and
their TwoNN d. Limits: a single n, five corpora from one platform and topic family, a 2024
summariser.
