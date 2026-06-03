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

## Key Findings

*(To be filled after analysis)*

- Calibration:
- Best calibrated category:
- Worst calibrated category:
- Biggest mispricing:
- Sentiment signal direction:

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
