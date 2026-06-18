"""
Feature engineering for financial ML.
Based on AFML Ch. 5 (fractional diff) and Ch. 18 (microstructure).
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ── Fractional Differentiation (AFML Ch. 5) ─────────────────────────────────

def _frac_diff_weights(d: float, size: int) -> np.ndarray:
    """Compute binomial series weights for fractional differencing."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])


def frac_diff(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """
    Fixed-window fractional differentiation.
    d=1 ≡ standard differencing; d=0 ≡ original series.
    thresh controls how many lagged weights are included.
    """
    weights = _frac_diff_weights(d, len(series))
    weights = weights[np.abs(weights) > thresh]
    width = len(weights)

    out = {}
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1 : i + 1].values
        out[series.index[i]] = float(np.dot(weights, window))

    return pd.Series(out, name=f"fracdiff_{d}")


def find_min_d(series: pd.Series, pvalue_thresh: float = 0.05) -> float:
    """
    Find minimum d such that fracdiff(series, d) is stationary (ADF test).
    Returns d rounded to 2 decimal places.
    """
    from statsmodels.tsa.stattools import adfuller

    for d in np.arange(0.0, 1.01, 0.05):
        fd = frac_diff(series.dropna(), round(float(d), 2))
        if len(fd) < 20:
            continue
        pval = adfuller(fd.dropna(), maxlag=1, autolag=None)[1]
        if pval < pvalue_thresh:
            return round(float(d), 2)
    return 1.0


# ── Microstructure features (AFML Ch. 18) ───────────────────────────────────

def kyle_lambda(price: pd.Series, volume: pd.Series) -> float:
    """
    Kyle's lambda: price impact coefficient via OLS of dp on sign(dv)*sqrt(|dv|).
    Higher lambda → less liquid market.
    """
    dp = price.diff().dropna()
    dv = volume.diff().dropna()
    idx = dp.index.intersection(dv.index)
    dp, dv = dp.loc[idx], dv.loc[idx]

    signed_vol = np.sign(dv) * np.sqrt(np.abs(dv))
    valid = signed_vol != 0
    if valid.sum() < 10:
        return np.nan

    return float(np.polyfit(signed_vol[valid], dp[valid], 1)[0])


def roll_spread(close: pd.Series) -> float:
    """
    Roll's effective spread estimator: 2 * sqrt(-cov(dp_t, dp_{t-1})).
    Returns NaN if covariance is positive (not applicable).
    """
    dp = close.diff().dropna()
    cov = dp.cov(dp.shift(1).dropna().reindex(dp.index))
    return 2 * np.sqrt(-cov) if cov < 0 else np.nan


def entropy_bar(prob_series: pd.Series) -> float:
    """Shannon entropy of a probability distribution (e.g. order-flow imbalance)."""
    p = prob_series.dropna().clip(1e-10, 1 - 1e-10)
    p = p / p.sum()
    return float(scipy_entropy(p))


# ── Order Flow Imbalance (Cont, Kukanov & Stoikov 2014) ──────────────────────

def order_flow_imbalance(book: pd.DataFrame) -> pd.Series:
    """
    Per-update order flow imbalance e_n from top-of-book (Cont, Kukanov & Stoikov 2014).

        e_n =  q^b_n 1{P^b_n >= P^b_{n-1}}  -  q^b_{n-1} 1{P^b_n <= P^b_{n-1}}
              - q^a_n 1{P^a_n <= P^a_{n-1}}  +  q^a_{n-1} 1{P^a_n >= P^a_{n-1}}

    Positive e_n = net buy-side pressure (bid replenished / ask consumed). Sum e_n over an
    interval to get the bar-level OFI, a strong contemporaneous driver of price moves.

    book : DataFrame ordered by time with columns [bid_px, bid_qty, ask_px, ask_qty].
    Returns e_n aligned to book.index (first element = 0).
    """
    pb = book["bid_px"].to_numpy(float); qb = book["bid_qty"].to_numpy(float)
    pa = book["ask_px"].to_numpy(float); qa = book["ask_qty"].to_numpy(float)
    pb0, qb0 = np.roll(pb, 1), np.roll(qb, 1)
    pa0, qa0 = np.roll(pa, 1), np.roll(qa, 1)

    bid_term = np.where(pb >= pb0, qb, 0.0) - np.where(pb <= pb0, qb0, 0.0)
    ask_term = np.where(pa <= pa0, qa, 0.0) - np.where(pa >= pa0, qa0, 0.0)
    e = bid_term - ask_term
    e[0] = 0.0  # no previous update for the first row
    return pd.Series(e, index=book.index, name="ofi")


def depth_imbalance(bid_qty: pd.Series, ask_qty: pd.Series) -> pd.Series:
    """Top-of-book depth imbalance in [-1, 1]: (q^b - q^a) / (q^b + q^a)."""
    denom = (bid_qty + ask_qty).replace(0, np.nan)
    return ((bid_qty - ask_qty) / denom).fillna(0.0)
