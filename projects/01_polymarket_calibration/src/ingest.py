"""
Polymarket data ingestion.

Endpoints used:
  Gamma API  https://gamma-api.polymarket.com
    GET /markets          → resolved market metadata
    GET /prices-history   → daily price timeseries per market

Usage:
    python src/ingest.py                  # fetch all, save to data/
    python src/ingest.py --limit 200      # quick test run
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Allow running from project root or from src/
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.data_utils import HttpCache, ThrottledSession, save_parquet

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ── Market metadata ──────────────────────────────────────────────────────────

def fetch_resolved_markets(
    session: ThrottledSession,
    limit: int | None = None,
    page_size: int = 100,
) -> list[dict]:
    """
    Page through /markets filtering for resolved/closed markets.
    Returns list of raw dicts from the API.
    """
    markets = []
    offset = 0

    while True:
        params = {
            "closed": "true",
            "limit": page_size,
            "offset": offset,
        }
        data = session.get(f"{GAMMA_BASE}/markets", params=params)

        # Gamma API returns a list directly
        if not isinstance(data, list):
            log.warning("Unexpected response shape: %s", type(data))
            break

        if not data:
            break

        markets.extend(data)
        log.info("Fetched %d markets so far (offset=%d)", len(markets), offset)

        if limit and len(markets) >= limit:
            markets = markets[:limit]
            break

        if len(data) < page_size:
            break

        offset += page_size

    return markets


def parse_markets(raw: list[dict]) -> pd.DataFrame:
    """Extract relevant fields from raw market dicts."""
    rows = []
    for m in raw:
        # Determine resolved outcome
        # Gamma API: resolutionPrice is 1.0 for YES, 0.0 for NO resolution
        res_price = m.get("resolutionPrice")
        if res_price is None:
            # Try tokens: the winning token has price 1.0
            tokens = m.get("tokens", [])
            for t in tokens:
                if t.get("price") == 1.0:
                    res_price = 1.0 if t.get("outcome", "").upper() == "YES" else 0.0
                    break

        rows.append(
            {
                "market_id": m.get("id"),
                "condition_id": m.get("conditionId"),
                "question": m.get("question"),
                "category": (m.get("tags") or [{}])[0].get("label") if m.get("tags") else None,
                "start_date": m.get("startDate"),
                "end_date": m.get("endDate"),
                "resolution_price": res_price,
                "outcome": "YES" if res_price == 1.0 else ("NO" if res_price == 0.0 else None),
                "volume_usd": m.get("volume"),
                "liquidity_usd": m.get("liquidity"),
                "slug": m.get("slug"),
            }
        )

    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
    df["volume_usd"] = pd.to_numeric(df["volume_usd"], errors="coerce")
    df["liquidity_usd"] = pd.to_numeric(df["liquidity_usd"], errors="coerce")
    return df


# ── Price history ────────────────────────────────────────────────────────────

def fetch_price_history(
    session: ThrottledSession,
    markets_df: pd.DataFrame,
    fidelity: int = 1440,  # 1440 min = 1 day
) -> pd.DataFrame:
    """
    For each market, fetch the daily price time series from Gamma API.
    fidelity: candle granularity in minutes (1440 = daily).
    """
    all_rows = []
    failed = []

    for _, row in tqdm(markets_df.iterrows(), total=len(markets_df), desc="Price history"):
        market_id = row["market_id"]
        if pd.isna(market_id):
            continue

        try:
            params = {"market": market_id, "fidelity": fidelity}
            data = session.get(f"{GAMMA_BASE}/prices-history", params=params)

            # Response: {"history": [{"t": unix_ts, "p": price}, ...]}
            history = data.get("history", []) if isinstance(data, dict) else []

            for point in history:
                all_rows.append(
                    {
                        "market_id": market_id,
                        "timestamp": pd.Timestamp(point["t"], unit="s", tz="UTC"),
                        "price": float(point["p"]),
                    }
                )

        except Exception as e:
            log.warning("Failed price history for %s: %s", market_id, e)
            failed.append(market_id)
            continue

    if failed:
        log.warning("%d markets failed price fetch", len(failed))

    return pd.DataFrame(all_rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(limit: int | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cache = HttpCache(cache_dir=DATA_DIR / ".http_cache")
    session = ThrottledSession(min_interval_s=0.25, cache=cache)

    # 1. Fetch market metadata
    log.info("Fetching resolved markets from Gamma API...")
    raw_markets = fetch_resolved_markets(session, limit=limit)
    log.info("Total raw markets: %d", len(raw_markets))

    markets_df = parse_markets(raw_markets)

    # Keep only markets with a definitive YES/NO resolution
    markets_df = markets_df[markets_df["outcome"].isin(["YES", "NO"])].reset_index(drop=True)
    log.info("Markets with clean YES/NO resolution: %d", len(markets_df))

    save_parquet(markets_df, DATA_DIR / "markets.parquet")
    log.info("Saved → data/markets.parquet  (%d rows)", len(markets_df))

    # 2. Fetch daily price history
    log.info("Fetching daily price history...")
    prices_df = fetch_price_history(session, markets_df)
    log.info("Total price rows: %d", len(prices_df))

    save_parquet(prices_df, DATA_DIR / "price_history.parquet")
    log.info("Saved → data/price_history.parquet  (%d rows)", len(prices_df))

    # Quick sanity check
    print("\n── Markets summary ──────────────────────────")
    print(markets_df["category"].value_counts().head(15).to_string())
    print(f"\nDate range: {markets_df['end_date'].min().date()} → {markets_df['end_date'].max().date()}")
    print(f"YES resolutions: {(markets_df['outcome'] == 'YES').sum()}")
    print(f"NO  resolutions: {(markets_df['outcome'] == 'NO').sum()}")
    print(f"\nPrice history: {len(prices_df):,} rows across {prices_df['market_id'].nunique():,} markets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Polymarket historical data")
    parser.add_argument("--limit", type=int, default=None, help="Max markets to fetch (default: all)")
    args = parser.parse_args()
    main(limit=args.limit)
