# Project 02 — Crypto Microstructure: Order Flow Imbalance

A complete, leakage-free research loop on BTC perpetual-futures microstructure, built from free public
data. The deliverable is not a profitable strategy — it is a **correctly wired AFML pipeline that
honestly reports there is no predictable edge** at this horizon, instead of a leaky backtest that
pretends otherwise.

## Hypothesis

> **Order Flow Imbalance (OFI)** at the top of the book of BTC perpetual futures predicts the
> short-horizon return above chance — and **the signal survives purged cross-validation and a
> deflated Sharpe test**.

The hypothesis carries its own falsification condition on purpose. Top-of-book OFI alpha is largely
arbitraged away at latencies a retail researcher cannot reach, so the expected, honest conclusion is
"strong *contemporaneous* relation, no *predictive* edge out-of-sample" — and the value is the rigor
of demonstrating that.

## TL;DR result

| Question | Answer |
|---|---|
| Is the OFI feature correct? | **Yes** — contemporaneous R² = **0.58**, corr **+0.76** vs same-bar return (reproduces Cont et al. price impact) |
| Does OFI predict the next bar? | **No** — corr(OFI_t, ret_{t+1}) = **−0.03** ≈ 0 |
| Does any feature survive purged CV? | **No** — OOS accuracy 0.516 vs 0.512 baseline; MDA(OFI) = −0.002 |
| Does the P&L survive deflation? | **No** — Sharpe +0.02, **Deflated Sharpe 0.13** (needs ~0.95) |

The equity curve *looks* like it works; the deflated Sharpe rejects it. That gap is the whole point.

## Data reality (read before assuming a feed exists)

All data is **free** from [Binance Vision](https://data.binance.vision) (S3 public dumps, no key).
Probed 2026-06-18:

| Dataset | Path (futures USDⓂ, daily) | Availability for BTCUSDT | Scale |
|---|---|---|---|
| `bookTicker` (top-of-book) | `futures/um/daily/bookTicker/BTCUSDT/` | **2023-05-16 → 2024-03-30** (discontinued after) | ~30M rows/day, ~87 MB zip/day |
| `aggTrades` (trades w/ sign) | `futures/um/daily/aggTrades/BTCUSDT/` | 2019-12-31 → present (continuous) | ~1–2M rows/day |

**Working window = the overlap `2023-05-16 → 2024-03-30`.** Two engineering constraints shaped the
code and are worth calling out:
- **Memory:** one month of `aggTrades` ≈ 66M rows and 3 days of `bookTicker` ≈ 97M rows — too large
  to hold at once on a laptop. Bars are built on a few-day slice; OFI is aggregated **day-by-day**
  (`dataset.book_day_paths`), the production streaming pattern.
- **L2 depth** is *not* free in bulk → deferred to a future self-hosted websocket collector (home
  server). This project uses **top-of-book only**, which is exactly what the classic OFI definition
  (Cont, Kukanov & Stoikov 2014) requires.

### Schemas
- **bookTicker**: `update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, transaction_time, event_time` (ms epoch).
- **aggTrades**: `agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker`. `is_buyer_maker=true` ⇒ aggressor is the **seller** ⇒ trade sign **−1**.

## Notebooks

| Notebook | Question | Key finding |
|---|---|---|
| [`01_bar_comparison`](notebooks/01_bar_comparison.ipynb) | Which bar sampling scheme has the best statistical properties? | **Standard info bars win** (tick/volume/dollar: JB ~9–12, \|autocorr\|<0.07). **Time bars worst** (JB 429, fat tails). **Imbalance bars unstable** (JB in millions) — a tuning rabbit hole, not alpha. → **chose dollar bars** |
| [`02_ofi_signal`](notebooks/02_ofi_signal.ipynb) | Does OFI explain / predict returns? | **Contemporaneous R²=0.58, corr +0.76** (feature validated) but **predictive corr ≈ −0.03** (no edge one bar ahead) |
| [`03_labeling_cv`](notebooks/03_labeling_cv.ipynb) | Does OFI+spread+depth survive leakage-free validation? | **No.** OOS acc 0.516 vs 0.512; MDA(OFI)=−0.002; **Deflated Sharpe 0.13** |

Plots: [`01_bar_comparison.png`](plots/01_bar_comparison.png) ·
[`02_ofi_contemporaneous.png`](plots/02_ofi_contemporaneous.png) ·
[`02_ofi_predictive.png`](plots/02_ofi_predictive.png) ·
[`03_strategy_equity.png`](plots/03_strategy_equity.png)

## Method (AFML mapping)

| AFML chapter | Where | What it becomes here |
|---|---|---|
| Ch. 2 — Financial Data | `lib/bars.py` | Tick/volume/dollar/imbalance/run bars + `bar_diagnostics` → **dollar bars chosen** |
| Ch. 3 — Labeling | `lib/labeling.py` | **Triple-barrier** labels (horizon = 10 bars) |
| Ch. 4 — Sample Weights | `lib/labeling.py` | Label weights by **uniqueness / overlap** (avg uniqueness 0.30) |
| Ch. 7 — Cross-Validation | `lib/backtest.py` | **Purged K-Fold + embargo** |
| Ch. 8 — Feature Importance | notebook 03 | **MDA** (mean-decrease-accuracy) under purged CV |
| Ch. 14 — Backtest stats | `lib/backtest.py` | **Probabilistic & Deflated Sharpe** |
| Ch. 18 — Microstructure | `lib/features.py` | **OFI** (Cont et al.), depth imbalance, spread |

## Reproduce

```bash
# 1. Fetch data (per-day parquet partitions under data/, gitignored, resumable)
python src/ingest.py --dataset agg_trades --start 2024-03-01 --end 2024-03-30
python src/ingest.py --dataset book_ticker --start 2024-03-01 --end 2024-03-03

# 2. Run the notebooks in order (use the venv 3.14 kernel registered as 'traiding')
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=traiding notebooks/01_bar_comparison.ipynb
# ... 02_ofi_signal, 03_labeling_cv
```

## Structure

```
02_crypto_microstructure_ofi/
├── data/                 # gitignored; ingest.py populates, partitioned by day
│   └── features/         # bar-level feature tables saved by notebook 02
├── notebooks/
│   ├── 01_bar_comparison.ipynb   # bar sampling experiments → dollar bars
│   ├── 02_ofi_signal.ipynb       # OFI feature, contemporaneous vs predictive
│   └── 03_labeling_cv.ipynb      # triple-barrier + purged CV + deflated Sharpe
├── plots/                # figures rendered by the notebooks
└── src/
    ├── ingest.py         # Binance Vision → per-day parquet (bookTicker + aggTrades)
    └── dataset.py        # partition loaders (load_trades, load_book, book_day_paths)
```

Shared logic lives in the repo-level [`lib/`](../../lib/) (`bars`, `features`, `labeling`,
`backtest`) so it is reusable across projects.

## Key findings

- **Bars:** dollar bars give near-IID-normal returns out of the box; imbalance/run bars add
  instability without statistical benefit on a single liquid symbol.
- **OFI is contemporaneous, not predictive:** it explains ~58% of same-bar return variance yet has
  ~0 correlation with the next bar — the information is already in the price by bar close.
- **No edge survives leakage-free validation:** purged-CV accuracy is at baseline, OFI's MDA is zero,
  and the strategy's deflated Sharpe (0.13) is consistent with random search.
- **The methodology is the result:** a pipeline that *would* detect a real edge and correctly reports
  there isn't one. Extensions worth trying: longer horizon, a less-liquid symbol, meta-labeling,
  CUSUM event sampling, and L2-depth features once the home-server collector exists.
