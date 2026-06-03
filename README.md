# TrAIding — Quantitative Research Portfolio

Personal research repo on financial markets. Projects combine classical quant techniques
(from *Advances in Financial Machine Learning*, Lopez de Prado) with modern ML.

---

## Projects

| # | Project | Status | Key Finding |
|---|---------|--------|-------------|
| 01 | [Polymarket Calibration + Sentiment Signal](projects/01_polymarket_calibration/) | In progress | — |

---

## Shared Library (`lib/`)

Reusable modules across projects:

- **`data_utils`** — parquet I/O, HTTP caching, rate-limit-aware retries
- **`labeling`** — triple-barrier labeling, meta-labeling (AFML Ch. 3)
- **`features`** — fractional differentiation, entropy bars, microstructure features (AFML Ch. 5, 18)
- **`backtest`** — walk-forward validation, Brier score, combinatorial purged CV

---

## Reading

- *Advances in Financial Machine Learning* — Lopez de Prado (2018) → [notes](notes/advances_in_fml.md)

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
pip install -e .             # installs lib/ as editable package
```

All data is fetched via free public APIs and cached locally as parquet.
Raw data files are gitignored; re-fetch with `python src/ingest.py` inside each project.
