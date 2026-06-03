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
CLOB_BASE = "https://clob.polymarket.com"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# CLOB API only supports up to 14-day windows per request
CLOB_MAX_WINDOW_DAYS = 14


# ── Market metadata ──────────────────────────────────────────────────────────

def fetch_resolved_markets(
    session: ThrottledSession,
    limit: int | None = None,
    page_size: int = 100,
    end_date_min: str = "2023-01-01",
) -> list[dict]:
    """
    Page through /markets filtering for resolved/closed markets.
    end_date_min: only fetch markets that closed on or after this date.
                  CLOB price data is only available for 2023+ markets.
    Returns list of raw dicts from the API.
    """
    markets = []
    offset = 0

    while True:
        params = {
            "closed": "true",
            "end_date_min": end_date_min,
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


RESOLUTION_THRESHOLD = 0.99  # price >= this → market resolved to that outcome


def _parse_outcome(outcomes_str: str, prices_str: str) -> tuple[float | None, str | None]:
    """
    Parse outcomePrices JSON string and return (resolution_price, label).
    outcomePrices are floats that converge to ~1.0 for the winning outcome.
    Only binary YES/NO markets are kept for calibration analysis.
    Returns (None, None) for unresolved or non-binary markets.
    """
    import json

    try:
        outcomes = json.loads(outcomes_str)
        prices = [float(p) for p in json.loads(prices_str)]
    except Exception:
        return None, None

    # Only binary markets
    if len(outcomes) != 2:
        return None, None

    # Normalise outcome labels
    labels = [o.strip().upper() for o in outcomes]

    # Find winning outcome (price >= threshold)
    winner_idx = next((i for i, p in enumerate(prices) if p >= RESOLUTION_THRESHOLD), None)
    if winner_idx is None:
        return None, None

    winner_label = labels[winner_idx]

    # Map to YES/NO — handle "YES"/"NO", "1"/"0", or first/second outcome
    if winner_label in ("YES", "TRUE", "1"):
        return 1.0, "YES"
    elif winner_label in ("NO", "FALSE", "0"):
        return 0.0, "NO"
    else:
        # Non-standard binary (e.g. "CROATIA"/"ENGLAND") — skip for calibration
        return None, None


def parse_markets(raw: list[dict]) -> pd.DataFrame:
    """Extract relevant fields from raw market dicts."""
    rows = []
    for m in raw:
        outcomes_str = m.get("outcomes", "[]")
        prices_str = m.get("outcomePrices", "[0,0]")
        res_price, outcome = _parse_outcome(outcomes_str, prices_str)

        # startDate lives inside events[], not directly on market
        events = m.get("events") or []
        start_date = events[0].get("startDate") if events else m.get("createdAt")

        rows.append(
            {
                "market_id": m.get("id"),
                "condition_id": m.get("conditionId"),
                "question": m.get("question"),
                "category": m.get("category"),
                "start_date": start_date,
                "end_date": m.get("endDate"),
                "resolution_price": res_price,
                "outcome": outcome,
                "volume_usd": m.get("volume"),
                "liquidity_usd": m.get("liquidity"),
                "slug": m.get("slug"),
                "clob_token_ids": m.get("clobTokenIds", "[]"),
            }
        )

    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
    df["volume_usd"] = pd.to_numeric(df["volume_usd"], errors="coerce")
    df["liquidity_usd"] = pd.to_numeric(df["liquidity_usd"], errors="coerce")
    return df


# ── Price history ────────────────────────────────────────────────────────────

def _clob_price_chunks(
    session: ThrottledSession,
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 1440,
) -> list[dict]:
    """
    Fetch price history from CLOB API in 14-day chunks.
    CLOB enforces a max window of 14 days per request.
    Returns flat list of {t, p} dicts.
    """
    window = CLOB_MAX_WINDOW_DAYS * 86400
    points = []
    chunk_start = start_ts

    while chunk_start < end_ts:
        chunk_end = min(chunk_start + window, end_ts)
        try:
            data = session.get(
                f"{CLOB_BASE}/prices-history",
                params={
                    "market": token_id,
                    "startTs": chunk_start,
                    "endTs": chunk_end,
                    "fidelity": fidelity,
                },
            )
            points.extend(data.get("history", []) if isinstance(data, dict) else [])
        except Exception as e:
            log.debug("Chunk fetch failed token=%s start=%s: %s", token_id[:16], chunk_start, e)
        chunk_start = chunk_end

    return points


def fetch_price_history(
    session: ThrottledSession,
    markets_df: pd.DataFrame,
    fidelity: int = 1440,
) -> pd.DataFrame:
    """
    Fetch daily YES-price history for each market via the CLOB API.
    Uses the YES token (clobTokenIds[0]) and paginates in 14-day chunks.
    """
    all_rows = []
    failed = []

    for _, row in tqdm(markets_df.iterrows(), total=len(markets_df), desc="Price history"):
        market_id = row["market_id"]
        clob_ids_raw = row.get("clob_token_ids", "[]") or "[]"

        try:
            import json as _json
            clob_ids = _json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
        except Exception:
            clob_ids = []

        if not clob_ids:
            log.debug("No CLOB token for market %s, skipping", market_id)
            continue

        yes_token = clob_ids[0]  # index 0 = YES token

        # Use actual market dates for the query window
        start_dt = row.get("start_date")
        end_dt = row.get("end_date")
        if pd.isna(start_dt) or pd.isna(end_dt):
            continue

        start_ts = int(pd.Timestamp(start_dt).timestamp())
        end_ts = int(pd.Timestamp(end_dt).timestamp())

        try:
            points = _clob_price_chunks(session, yes_token, start_ts, end_ts, fidelity)
            for pt in points:
                all_rows.append({
                    "market_id": market_id,
                    "timestamp": pd.Timestamp(pt["t"], unit="s", tz="UTC"),
                    "price": float(pt["p"]),
                })
        except Exception as e:
            log.warning("Failed price history for %s: %s", market_id, e)
            failed.append(market_id)

    if failed:
        log.warning("%d markets failed price fetch", len(failed))

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame(columns=["market_id", "timestamp", "price"])


# ── Main ─────────────────────────────────────────────────────────────────────

def main(limit: int | None = None, end_date_min: str = "2023-01-01") -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cache = HttpCache(cache_dir=DATA_DIR / ".http_cache")
    session = ThrottledSession(min_interval_s=0.25, cache=cache)

    # 1. Fetch market metadata
    log.info("Fetching resolved markets from Gamma API (end_date >= %s)...", end_date_min)
    raw_markets = fetch_resolved_markets(session, limit=limit, end_date_min=end_date_min)
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
    print("\n--- Markets summary ---")
    print(markets_df["category"].value_counts().head(15).to_string())
    end_min = markets_df['end_date'].min()
    end_max = markets_df['end_date'].max()
    print(f"\nDate range: {end_min.date() if pd.notna(end_min) else 'N/A'} to {end_max.date() if pd.notna(end_max) else 'N/A'}")
    print(f"YES resolutions: {(markets_df['outcome'] == 'YES').sum()}")
    print(f"NO  resolutions: {(markets_df['outcome'] == 'NO').sum()}")
    print(f"\nPrice history: {len(prices_df):,} rows across {prices_df['market_id'].nunique():,} markets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Polymarket historical data")
    parser.add_argument("--limit", type=int, default=None, help="Max markets to fetch (default: all)")
    parser.add_argument("--end_date_min", default="2023-01-01",
                        help="Only fetch markets closed on or after this date (default: 2023-01-01)")
    args = parser.parse_args()
    main(limit=args.limit, end_date_min=args.end_date_min)
