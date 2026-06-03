"""
NLP sentiment signal vs Polymarket odds.

Pipeline:
  1. For each resolved market, fetch news headlines via GDelt (free, no key).
  2. Score headlines with VADER (fast baseline) or finBERT (better).
  3. Compute daily sentiment_score per market.
  4. Build divergence signal: sentiment vs market odds.
  5. Forward-return analysis: does the market correct toward or away from sentiment?

Usage:
    python src/signal_sentiment.py --model vader
    python src/signal_sentiment.py --model finbert
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.data_utils import HttpCache, ThrottledSession, load_parquet, save_parquet

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"


# ── News fetching via GDelt ──────────────────────────────────────────────────

def build_gdelt_query(question: str) -> str:
    """Extract 2-3 key terms from a market question for a GDelt query."""
    stopwords = {"will", "the", "a", "an", "by", "in", "of", "to", "be", "win",
                 "does", "is", "are", "for", "on", "at", "with", "this", "that"}
    words = [w.strip("?.,'\"") for w in question.split()]
    keywords = [w for w in words if w.lower() not in stopwords and len(w) > 3]
    return " ".join(keywords[:4])


def fetch_gdelt_headlines(
    session: ThrottledSession,
    query: str,
    start_date: str,
    end_date: str,
    max_records: int = 50,
) -> list[str]:
    """
    Fetch news headlines from GDelt Doc 2.0 API.
    Dates: 'YYYYMMDDHHMMSS' format.
    """
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "startdatetime": start_date,
        "enddatetime": end_date,
        "format": "json",
        "sort": "DateDesc",
    }
    try:
        data = session.get(GDELT_BASE, params=params)
        articles = data.get("articles", []) if isinstance(data, dict) else []
        return [a.get("title", "") for a in articles if a.get("title")]
    except Exception as e:
        log.debug("GDelt fetch failed for '%s': %s", query, e)
        return []


# ── Sentiment scoring ────────────────────────────────────────────────────────

def load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def load_finbert():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=-1,  # CPU
        truncation=True,
        max_length=512,
    )


def score_headlines_vader(analyzer, headlines: list[str]) -> float:
    """Returns mean compound score [-1, 1]."""
    if not headlines:
        return np.nan
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return float(np.mean(scores))


def score_headlines_finbert(pipe, headlines: list[str]) -> float:
    """
    Returns mean positive probability (finBERT outputs positive/negative/neutral).
    Positive → bullish sentiment → market likely to resolve YES.
    """
    if not headlines:
        return np.nan
    results = pipe(headlines, batch_size=16)
    # Map: positive=+1, negative=-1, neutral=0, then average
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    scores = [label_map.get(r["label"], 0.0) * r["score"] for r in results]
    return float(np.mean(scores))


# ── Build signal dataset ─────────────────────────────────────────────────────

def build_sentiment_dataset(
    markets_df: pd.DataFrame,
    model: str = "vader",
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    For each market, compute a sentiment score from news in the week before resolution.
    Returns DataFrame with columns:
        market_id, question, outcome, final_odds, sentiment_score, divergence
    """
    cache = HttpCache(cache_dir=DATA_DIR / ".http_cache")
    session = ThrottledSession(min_interval_s=1.0, cache=cache)  # GDelt: be polite

    if model == "vader":
        scorer_obj = load_vader()
        score_fn = lambda headlines: score_headlines_vader(scorer_obj, headlines)
    elif model == "finbert":
        scorer_obj = load_finbert()
        score_fn = lambda headlines: score_headlines_finbert(scorer_obj, headlines)
    else:
        raise ValueError(f"Unknown model: {model}. Choose 'vader' or 'finbert'.")

    # Load final-day odds from price history
    prices = load_parquet(DATA_DIR / "price_history.parquet")
    final_odds = (
        prices.sort_values("timestamp")
        .groupby("market_id")
        .last()["price"]
        .rename("final_odds")
    )
    markets_df = markets_df.join(final_odds, on="market_id")

    rows = []
    for _, mkt in tqdm(markets_df.iterrows(), total=len(markets_df), desc=f"Sentiment ({model})"):
        if pd.isna(mkt.get("end_date")):
            continue

        end_dt = pd.Timestamp(mkt["end_date"])
        start_dt = end_dt - pd.Timedelta(days=lookback_days)

        query = build_gdelt_query(mkt["question"])
        headlines = fetch_gdelt_headlines(
            session,
            query=query,
            start_date=start_dt.strftime("%Y%m%d%H%M%S"),
            end_date=end_dt.strftime("%Y%m%d%H%M%S"),
        )

        sentiment = score_fn(headlines)
        final_odds_val = mkt.get("final_odds", np.nan)

        # Divergence: positive = sentiment more bullish than market odds
        # Normalise sentiment from [-1,1] to [0,1] for comparison with odds
        sentiment_norm = (sentiment + 1) / 2 if not np.isnan(sentiment) else np.nan
        divergence = sentiment_norm - final_odds_val if not np.isnan(sentiment_norm) else np.nan

        rows.append(
            {
                "market_id": mkt["market_id"],
                "question": mkt["question"],
                "category": mkt.get("category"),
                "outcome": mkt["outcome"],
                "final_odds": final_odds_val,
                "sentiment_score": sentiment,
                "sentiment_norm": sentiment_norm,
                "divergence": divergence,
                "n_headlines": len(headlines),
            }
        )

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(model: str = "vader", sample: int | None = None) -> None:
    markets = load_parquet(DATA_DIR / "markets.parquet")

    if sample:
        markets = markets.sample(n=min(sample, len(markets)), random_state=42)

    log.info("Building sentiment signal for %d markets using %s...", len(markets), model)
    signal_df = build_sentiment_dataset(markets, model=model)

    out_path = DATA_DIR / f"sentiment_signal_{model}.parquet"
    save_parquet(signal_df, out_path)
    log.info("Saved → %s  (%d rows)", out_path.name, len(signal_df))

    # Quick summary
    valid = signal_df.dropna(subset=["divergence"])
    print(f"\nMarkets with sentiment signal: {len(valid)}/{len(signal_df)}")
    print(f"Avg headlines per market: {signal_df['n_headlines'].mean():.1f}")
    print(f"\nDivergence distribution:")
    print(valid["divergence"].describe().round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vader", "finbert"], default="vader")
    parser.add_argument("--sample", type=int, default=None, help="Test on N markets only")
    args = parser.parse_args()
    main(model=args.model, sample=args.sample)
