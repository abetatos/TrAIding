# Project 03 — Cross-Sectional Equity Factors

A monthly **long-short cross-sectional** study on the S&P 500, built from free data. Where project 02
set out to honestly fail on a single liquid instrument, this project's explicit goal is the opposite:
**hunt for a factor edge that is real and defensible** — and only believe it if it survives
sector-neutralisation, realistic turnover costs, purged cross-validation and a deflated Sharpe.

## Hypothesis

> A cross-section of S&P 500 stocks can be **ranked each month by factor signals** (momentum,
> short-term reversal, low-volatility, quality) such that a **decile long-short book earns a positive
> risk-adjusted return net of costs** — and the edge **survives purged CV + a deflated Sharpe test**.

Survivorship bias is the headline caveat (see below): a positive backtest here is an **upper bound**,
so the bar for believing it is deliberately high.

## Data reality (read before assuming)

All data is **free** and keyless:

| Source | What | Notes |
|---|---|---|
| Wikipedia | S&P 500 constituents + GICS sectors | `src/universe.py`. **Current** members only ⇒ survivorship bias. Needs a browser `User-Agent` (Wikipedia 403s urllib's default). |
| Yahoo Finance (`yfinance`) | Daily split/dividend-adjusted OHLCV | `src/ingest.py`, `auto_adjust=True`. ~500 names × 15y. |

**Survivorship bias — stated up front.** The universe is *today's* index, so delisted/acquired/bankrupt
names are absent. This inflates factor returns (the losers that left the index are exactly the ones a
factor would have shorted/avoided). We carry the caveat through every conclusion; a production study
would use point-in-time constituents (CRSP / Compustat).

## Notebooks

| Notebook | Question | Key finding |
|---|---|---|
| [`01_universe_and_returns`](notebooks/01_universe_and_returns.ipynb) | Is the panel sound, and is there a cross-section to exploit? | **Yes.** 3,774 days × 503 names; coverage 422→500; median daily dispersion **143 bps** (809 bps COVID peak); 180 monthly rebalances, ~332 tradable names after a top-70%-ADV screen. Momentum 12-1 teaser rank-IC **+0.004 (IR 0.02)** — weak/noisy raw signal, as expected. |
| [`02_factor_zoo`](notebooks/02_factor_zoo.ipynb) | Do sector-neutral factors rank returns, and does a composite help? | **Barely / no.** All ICs single-bps. Reversal best IC (**+0.009**) but ~0 decile spread, decays <5m. Momentum only positive decile book (**+2.5%/yr, t≈0.6**, insignificant). **Low-vol has the WRONG sign** (−13.5%/yr, t≈−2.6) in this bull-market mega-cap universe. Naïve equal-weight composite **loses** (−5.5%/yr) by stacking the wrong-sign factor. |
| [`03_backtest_dsr`](notebooks/03_backtest_dsr.ipynb) | Does a sign-aware L/S book survive turnover costs + a deflated Sharpe? | **No.** Net of 10 bps/side every book loses: momentum −0.7%/yr, equal-composite −4%/yr, **look-ahead-free IC-weighted −6%/yr** (worst). Net Sharpes ≤ 0; PSR ≈ 0; **Deflated Sharpe ≈ 0** (needs ~0.95) for all three. The methodology rejects the edge. |

Plots: [`01_coverage.png`](plots/01_coverage.png) · [`01_dispersion.png`](plots/01_dispersion.png) ·
[`01_momentum_ic.png`](plots/01_momentum_ic.png) · [`02_factor_ic.png`](plots/02_factor_ic.png) ·
[`02_ic_decay.png`](plots/02_ic_decay.png) · [`02_decile_ls.png`](plots/02_decile_ls.png) ·
[`03_ic_weights.png`](plots/03_ic_weights.png) · [`03_net_equity.png`](plots/03_net_equity.png)

## TL;DR result

| Question | Answer |
|---|---|
| Is there a cross-section to rank? | **Yes** — 143 bps median daily dispersion across ~330 tradable names |
| Do price-only factors rank returns? | **Barely** — ICs are single bps; low-vol has the *wrong sign* in-sample |
| Does smart (IC-weighted, leak-free) combination help? | **No** — it loses *most*; rank-IC ≠ tradable tail P&L |
| Does any book survive 10 bps/side costs? | **No** — all net returns negative |
| Does the P&L survive deflation? | **No** — Deflated Sharpe ≈ 0 vs ~0.95 needed |

A tempting set of equity curves that the deflated Sharpe correctly refuses to believe — the same
discipline as project 02, on the cross-sectional equity side. **Honest conclusion: no edge in
price-only factors on the survivorship-biased S&P 500.**

## Method (AFML / standard-practice mapping)

| Concept | Where | What it becomes here |
|---|---|---|
| Clean panel / coverage audit | notebook 01 | Wide adj-close panel, coverage floor, liquidity screen |
| Cross-sectional dispersion | notebook 01 | The opportunity set a ranking strategy monetises |
| Factor construction & IC | notebook 02 | Momentum / reversal / low-vol / quality, sector-neutralised, rank-IC + decay |
| AFML Ch. 7 — Purged CV | `lib/backtest.py` | Leakage-free evaluation of the signal |
| AFML Ch. 14 — Backtest stats | `lib/backtest.py` | Probabilistic & **Deflated Sharpe** after the factor search |

## Reproduce

```bash
# 1. Download the panel (cached to data/, gitignored, resumable; ~500 names × 15y)
python src/ingest.py --start 2010-01-01 --end 2025-01-01

# 2. Run the notebooks in order (venv 3.14 kernel registered as 'traiding')
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=traiding notebooks/01_universe_and_returns.ipynb
# ... 02_factor_zoo, 03_backtest_dsr  (each notebook has a _build_NN.py that regenerates its source)
```

`notebooks/_build_01.py` regenerates the notebook source reproducibly (then nbconvert executes it).

## Structure

```
03_equity_factor_xsection/
├── data/                 # gitignored
│   ├── prices/           # adj_close.parquet, dollar_volume.parquet (wide: dates × tickers)
│   ├── panel/            # monthly panel saved by notebook 01 (price/return/adv/mask)
│   └── universe.parquet  # symbol, security, sector, sub_industry
├── notebooks/
│   ├── _build_0{1,2,3}.py            # reproducible notebook builders
│   ├── 01_universe_and_returns.ipynb # universe, coverage, dispersion, momentum teaser
│   ├── 02_factor_zoo.ipynb           # factors, neutralise, IC, decay, decile L/S, composite
│   └── 03_backtest_dsr.ipynb         # IC-weighted combo, turnover costs, PSR + deflated Sharpe
├── plots/                # figures rendered by the notebooks
└── src/
    ├── universe.py       # S&P 500 constituents (Wikipedia)
    ├── ingest.py         # yfinance → wide parquet panel
    ├── dataset.py        # panel loaders
    └── factors.py        # factor construction + IC / decay / decile-L/S evaluation
```

Shared logic lives in the repo-level [`lib/`](../../lib/) (`backtest` for purged CV + deflated Sharpe).

## Key findings

- **The panel is sound and audited** — coverage, liquidity and a monthly rebalance grid are in place,
  so factor work starts from vetted data rather than a silent leak. A real cross-section exists to rank
  (median 143 bps daily dispersion).
- **Price-only factors are weak on this universe.** Sector-neutral rank-ICs are single basis points;
  reversal has the best IC (+0.009) but **no tail edge**; momentum is the only positive decile book
  (+2.5%/yr) but statistically insignificant; **low-vol has the wrong sign** (−13.5%/yr) — the anomaly
  fails inside the S&P 500 over a tech-led bull market.
- **Smart combination does not rescue it.** A look-ahead-free, IC-weighted combo *loses most* — IC
  weighting optimises rank correlation, but a quintile book lives in the tails where the high-IC factor
  has no edge. Rank-IC is not tradable P&L.
- **Nothing survives costs or deflation.** Net of 10 bps/side every book is negative; Probabilistic
  Sharpe ≈ 0 and **Deflated Sharpe ≈ 0** (vs ~0.95 needed). The honest verdict is *no edge*.
- **The methodology is the deliverable** — a leak-free, costed, deflation-tested pipeline that would
  detect a real edge and correctly reports there isn't one in price-only S&P 500 factors. A genuine
  edge would more likely need **fundamentals/quality**, a **point-in-time** universe (kill survivorship
  bias), or a **broader/less-liquid** cross-section.
