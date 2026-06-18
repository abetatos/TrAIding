"""
Ingest daily adjusted prices for the S&P 500 universe from Yahoo Finance (free, no key).

Writes two wide parquet tables under data/ (dates x tickers), gitignored:
    data/prices/adj_close.parquet     # split/dividend-adjusted close
    data/prices/dollar_volume.parquet # adj_close * volume (ADV proxy for liquidity screens)
plus the universe metadata:
    data/universe.parquet             # symbol, security, sector, sub_industry

Resumable: if the parquet files already exist they are reused unless --force is passed.
Downloads in batches to be gentle on the endpoint.

Usage:
    python src/ingest.py --start 2010-01-01 --end 2025-01-01
    python src/ingest.py --force
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from universe import sp500_constituents  # noqa: E402

DATA = PROJ / "data"
PRICES = DATA / "prices"


def _download_batch(symbols: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download one batch; return (adj_close, volume) wide frames (dates x symbols)."""
    raw = yf.download(
        symbols, start=start, end=end, auto_adjust=True, progress=False, threads=True,
    )
    # yfinance returns a column MultiIndex (field, ticker) for multi-symbol requests.
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    vol = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]
    if not isinstance(raw.columns, pd.MultiIndex):  # single symbol edge case
        close.columns = vol.columns = symbols
    return close, vol


def ingest(start: str, end: str, batch_size: int = 50, force: bool = False) -> None:
    PRICES.mkdir(parents=True, exist_ok=True)
    close_path, dv_path = PRICES / "adj_close.parquet", PRICES / "dollar_volume.parquet"
    if close_path.exists() and dv_path.exists() and not force:
        print(f"prices already present ({close_path.relative_to(PROJ)}); use --force to redownload")
        return

    uni = sp500_constituents()
    uni.to_parquet(DATA / "universe.parquet", index=False)
    symbols = uni["symbol"].tolist()
    print(f"universe: {len(symbols)} symbols | window {start} -> {end}")

    closes, vols = [], []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        c, v = _download_batch(batch, start, end)
        closes.append(c)
        vols.append(v)
        print(f"  batch {i // batch_size + 1}: {len(batch)} symbols, {c.shape[0]} rows")
        time.sleep(1.0)  # be polite

    adj_close = pd.concat(closes, axis=1).sort_index()
    volume = pd.concat(vols, axis=1).sort_index()
    adj_close = adj_close.loc[:, ~adj_close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]
    dollar_volume = adj_close * volume

    adj_close.to_parquet(close_path)
    dollar_volume.to_parquet(dv_path)
    print(f"saved {close_path.relative_to(PROJ)}  shape={adj_close.shape}")
    print(f"saved {dv_path.relative_to(PROJ)}  shape={dollar_volume.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    ingest(args.start, args.end, args.batch_size, args.force)
