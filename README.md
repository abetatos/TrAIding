# TrAIding — Quantitative Research Portfolio

Personal research repo on financial markets. Projects combine classical quant techniques
(from *Advances in Financial Machine Learning*, Lopez de Prado) with modern ML.

---

## Projects

| # | Project | Status | Key Finding |
|---|---------|--------|-------------|
| 01 | [Polymarket Calibration](projects/01_polymarket_calibration/) | Complete | Polymarket is **well calibrated** (Brier Skill Score 0.49); sharp at the extremes, coin-flip in the 30–70% middle |
| 02 | [Crypto Microstructure: Order Flow Imbalance](projects/02_crypto_microstructure_ofi/) | Complete | OFI strongly explains the *current* bar (R²=0.58) but has **no predictive edge** out-of-sample — deflated Sharpe 0.13 |
| 03 | [Cross-Sectional Equity Factors](projects/03_equity_factor_xsection/) | Complete | Price-only factors on the S&P 500 are too weak to trade: every long-short book is **negative net of costs**, deflated Sharpe ≈ 0; low-vol even has the *wrong sign* in-sample |
| 04 | [Labeling Pitfalls](projects/04_labeling_pitfalls/) | Complete | An audit of the AFML pipeline on simulated data with known ground truth. Random k-fold reports **AUC 0.732** where the label is a coin flip; meta-labeling's **F1 argument fails by construction** (but sizing beats filtering, +0.45 Sharpe); and the best of ~186 defensible pipeline configurations on **pure noise** shows in-sample Sharpe **1.086**, out-of-sample **−0.10** |

---

## Shared Library (`lib/`)

Reusable modules across projects:

- **`data_utils`** — parquet I/O, HTTP caching, rate-limit-aware retries
- **`bars`** — information-driven bar construction: tick/volume/dollar, imbalance & run bars, diagnostics (AFML Ch. 2)
- **`labeling`** — triple-barrier labeling, meta-labeling, uniqueness sample weights (AFML Ch. 3–4)
- **`features`** — fractional differentiation, microstructure features, Order Flow Imbalance (AFML Ch. 5, 18)
- **`backtest`** — Brier score, walk-forward & purged K-fold CV, probabilistic/deflated Sharpe (AFML Ch. 7, 14)

Project 04 is deliberately self-contained (its own `src/`, no `lib/` imports) so that every claim
it audits is checked against an implementation written from scratch for that purpose.

---

## Reading

- *Advances in Financial Machine Learning* — Lopez de Prado (2018) → [notes](notes/advances_in_fml.md)

---

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync          # creates .venv + installs all dependencies from uv.lock
uv run jupyter notebook
```

To add a new dependency:
```bash
uv add <package>
```

All data is fetched via free public APIs and cached locally as parquet.
Raw data files are gitignored; re-fetch with `uv run python src/ingest.py` inside each project.

