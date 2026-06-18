"""
Load partitioned Binance Vision parquet written by ingest.py (Project 02).

Partitions live at  data/<dataset>/<symbol>/<YYYY-MM-DD>.parquet  and are concatenated
in date order. `start` / `end` are inclusive 'YYYY-MM-DD' strings (lexicographic on filename).
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load(dataset: str, symbol: str, start: str | None, end: str | None,
          data_dir: Path) -> pd.DataFrame:
    d = Path(data_dir) / dataset / symbol
    files = sorted(d.glob("*.parquet"))
    if start:
        files = [f for f in files if f.stem >= start]
    if end:
        files = [f for f in files if f.stem <= end]
    if not files:
        raise FileNotFoundError(
            f"No {dataset} partitions for {symbol} in {d} (start={start}, end={end}). "
            f"Run: python src/ingest.py --dataset {dataset} --start ... --end ..."
        )
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return df.sort_values("ts").reset_index(drop=True)


def load_trades(symbol: str = "BTCUSDT", start: str | None = None, end: str | None = None,
                data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Concatenated aggTrades: columns [ts, agg_id, price, qty, sign]."""
    return _load("agg_trades", symbol, start, end, data_dir)


def load_book(symbol: str = "BTCUSDT", start: str | None = None, end: str | None = None,
              data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Concatenated bookTicker: columns [ts, update_id, bid_px, bid_qty, ask_px, ask_qty, mid, spread].
    NOTE: bookTicker is ~30M rows/day — concatenating several days can exhaust RAM. For multi-day
    work prefer `book_day_paths` and process one partition at a time (see notebook 02)."""
    return _load("book_ticker", symbol, start, end, data_dir)


def book_day_paths(symbol: str = "BTCUSDT", start: str | None = None, end: str | None = None,
                   data_dir: Path = DATA_DIR) -> list[Path]:
    """Per-day bookTicker partition paths in date order (memory-safe day-by-day processing)."""
    d = Path(data_dir) / "book_ticker" / symbol
    files = sorted(d.glob("*.parquet"))
    if start:
        files = [f for f in files if f.stem >= start]
    if end:
        files = [f for f in files if f.stem <= end]
    return files
