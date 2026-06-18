"""
Loaders for the ingested S&P 500 panel. Keeps notebooks free of path/IO boilerplate.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
PRICES = DATA / "prices"


def load_universe() -> pd.DataFrame:
    return pd.read_parquet(DATA / "universe.parquet")


def load_adj_close() -> pd.DataFrame:
    """Wide frame (DatetimeIndex x ticker) of split/dividend-adjusted closes."""
    df = pd.read_parquet(PRICES / "adj_close.parquet")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_dollar_volume() -> pd.DataFrame:
    """Wide frame (DatetimeIndex x ticker) of adj_close * volume (ADV proxy)."""
    df = pd.read_parquet(PRICES / "dollar_volume.parquet")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()
