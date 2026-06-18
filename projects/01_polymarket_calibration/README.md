# Project 01 — Polymarket Calibration + Sentiment Signal

## Hypothesis

Prediction markets are considered well-calibrated — a market trading at 70% should resolve
YES ~70% of the time. But are they? And when NLP sentiment diverges from market odds, does
the market subsequently correct toward sentiment (signal alpha) or away (contrarian signal)?

## Data

| File | Source | Description |
|------|--------|-------------|
| `data/markets.parquet` | Gamma API | Resolved markets: question, category, dates, outcome |
| `data/price_history.parquet` | Gamma API | Daily YES-price per market |
| `data/sentiment_signal_vader.parquet` | GDelt | Sentiment score + divergence per market |

Fetch fresh:
```bash
python src/ingest.py               # ~1000+ resolved markets
python src/signal_sentiment.py     # VADER baseline (fast)
python src/signal_sentiment.py --model finbert  # upgrade
```

## Notebooks

| Notebook | Content |
|----------|---------|
| `notebooks/01_calibration.ipynb` | Calibration plots, Brier score by category, biggest mispricings |
| `notebooks/02_sentiment_signal.ipynb` | Sentiment vs odds divergence, forward return analysis |

## Category taxonomy (NLU)

Polymarket only labels ~41% of markets, so the taxonomy is **discovered from the data** instead of
hand-picked (notebook `01`, section 2):

1. Strip temporal/numeric noise from questions (dates, months, `$`/`%`) so topics — not deadlines —
   drive the grouping.
2. Embed with `all-MiniLM-L6-v2` and cluster with K-Means (`k` chosen by silhouette over an
   interpretable range, not the global argmax which over-splits).
3. Inspect each cluster's c-TF-IDF keywords + representative questions.
4. Name each cluster with **BART zero-shot** on representative examples (≈9 calls), then propagate
   the label to every market — no per-market inference.

This yields 8 data-driven categories (Politics, Soccer, NFL, Cryptocurrency, Inflation/Fed, Movie
box office, Chess, Social media). See the notebook's *Design notes* for the pitfalls each step fixes.

## Key Findings

Based on 140 resolved binary markets with daily price history (2023, YES base rate 30.7%).

- **Calibration:** well calibrated and genuinely skilled — Brier `0.109` vs base-rate baseline
  `0.213` (**Brier Skill Score 0.488**, resolving ~half the uncertainty).
- **Extremes are sharp, if slightly under-confident:** markets priced ≥85% resolved YES 94.4%;
  markets priced ≤15% resolved YES 3.3%. No favourite–longshot overpricing.
- **The middle (30–70%) is the hard zone:** Brier ≈ 0.24, barely better than a coin flip — genuine
  uncertainty (which team wins, close elections).
- **Prices sharpen over time:** Brier falls start `0.165` → mid `0.117` → final `0.101`.
- **Mild optimism bias:** mean signed error `+0.037` ("Will X happen?" priced a touch high).
- **Best calibrated category:** Social media (BS 0.012) and Cryptocurrency (BS 0.028) —
  near-mechanical questions (price thresholds, tweet counts).
- **Worst calibrated category:** Inflation / Fed (BS 0.173) and sports/chess — truly uncertain
  competitive outcomes.
- **Biggest mispricing:** *"Will Donald J. Trump be indicted by March 31, 2023?"* — priced 6%,
  resolved YES (error −0.94).
- **Sentiment signal direction:** *(pending — notebook `02`)*

> **Caveats:** only 140 of 375 markets have price history; several calibration bins and per-category
> counts have n ≤ 5–9, so findings are **directional, not statistically tight**. The 2023 sample is
> dominated by a few event types and is not blindly extrapolable.

## Structure

```
01_polymarket_calibration/
├── data/               # gitignored; run ingest.py to populate
├── notebooks/
│   ├── 01_calibration.ipynb
│   └── 02_sentiment_signal.ipynb
└── src/
    ├── ingest.py           # Gamma API → parquet
    └── signal_sentiment.py # GDelt + VADER/finBERT → signal
```
