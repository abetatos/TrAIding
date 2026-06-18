"""
Investable universe: current S&P 500 constituents from Wikipedia.

Caveat (read this, it matters for honesty): Wikipedia lists *today's* members, so any
panel built from it carries **survivorship bias** — firms that were dropped, delisted or went
bankrupt are missing, which inflates backtested factor returns. We accept this for a learning
project and flag it explicitly in every result; a production study would use a point-in-time
constituents file (e.g. CRSP / Compustat) instead.
"""
from __future__ import annotations

import pandas as pd

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_constituents() -> pd.DataFrame:
    """
    Scrape the current S&P 500 membership table.

    Returns a DataFrame with columns ['symbol', 'security', 'sector', 'sub_industry'].
    Tickers are normalised for Yahoo Finance (BRK.B -> BRK-B).
    """
    # Wikipedia 403s the default urllib UA; send a browser-like one.
    table = pd.read_html(
        WIKI_SP500, storage_options={"User-Agent": "Mozilla/5.0 (research; quant-portfolio)"}
    )[0]
    df = table.rename(
        columns={
            "Symbol": "symbol",
            "Security": "security",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )[["symbol", "security", "sector", "sub_industry"]]
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False).str.strip()
    return df.sort_values("symbol").reset_index(drop=True)
