"""
Cross-sectional factor construction & evaluation helpers (price-only, monthly rebalance).

All functions are pure (no globals) so notebooks stay thin. Conventions:
- `me`   : month-end price, wide (date x ticker)
- `mret` : month-over-month return, wide
- a "factor" is a wide (date x ticker) frame; higher value = more attractive (long) name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Raw price-only factors ───────────────────────────────────────────────────

def build_price_factors(
    daily_close: pd.DataFrame, me: pd.DataFrame, mret: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    Three classic price-only signals, each oriented so HIGH = expected outperformer:
      - mom_12_1     : 11-month return ending one month ago (skip last month / reversal)
      - reversal_1m  : negative of last month's return (short-term mean reversion)
      - low_vol      : negative trailing 12m daily volatility (low-vol anomaly)
    """
    dret = daily_close.pct_change().clip(-0.5, 0.5)
    vol_m = dret.rolling(252, min_periods=120).std().resample("ME").last()
    return {
        "mom_12_1": me.shift(1) / me.shift(12) - 1,
        "reversal_1m": -mret,
        "low_vol": -vol_m,
    }


# ── Neutralisation ───────────────────────────────────────────────────────────

def neutralize(factor: pd.DataFrame, mask: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """
    Per month: restrict to tradable names, cross-sectionally z-score, then subtract the
    sector mean so the factor carries no net GICS-sector tilt (sector-neutral).
    """
    out = {}
    for dt in factor.index:
        row = factor.loc[dt]
        if dt in mask.index:
            row = row.where(mask.loc[dt])
        z = (row - row.mean()) / row.std()
        z = z.groupby(sector).transform(lambda s: s - s.mean())
        out[dt] = z
    return pd.DataFrame(out).T


# ── Evaluation ───────────────────────────────────────────────────────────────

def rank_ic(factor_z: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 50) -> pd.Series:
    """Monthly Spearman rank-IC between a (neutralised) factor and next-month return."""
    out = {}
    for dt in factor_z.index:
        if dt not in fwd.index:
            continue
        a, b = factor_z.loc[dt], fwd.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() < min_names:
            continue
        out[dt] = a[m].rank().corr(b[m].rank())
    return pd.Series(out)


def ic_decay(factor_z: pd.DataFrame, me: pd.DataFrame, hmax: int = 12,
             min_names: int = 50) -> pd.Series:
    """Mean rank-IC of the factor against h-month-forward returns, for h = 1..hmax."""
    out = {}
    for h in range(1, hmax + 1):
        fwd_h = me.shift(-h) / me - 1
        ics = []
        for dt in factor_z.index:
            if dt not in fwd_h.index:
                continue
            a, b = factor_z.loc[dt], fwd_h.loc[dt]
            m = a.notna() & b.notna()
            if m.sum() >= min_names:
                ics.append(a[m].rank().corr(b[m].rank()))
        out[h] = float(np.mean(ics)) if ics else np.nan
    return pd.Series(out)


def decile_long_short(factor_z: pd.DataFrame, fwd: pd.DataFrame, n_deciles: int = 10,
                      min_names: int = 100) -> pd.Series:
    """
    Monthly gross return of a top-minus-bottom decile book (equal-weight legs).
    Index = rebalance month, value = next-month L/S return (before costs).
    """
    out = {}
    for dt in factor_z.index:
        if dt not in fwd.index:
            continue
        a, b = factor_z.loc[dt], fwd.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() < min_names:
            continue
        d = pd.qcut(a[m], n_deciles, labels=False, duplicates="drop")
        out[dt] = b[m][d == d.max()].mean() - b[m][d == 0].mean()
    return pd.Series(out)


def ls_stats(ls: pd.Series) -> dict[str, float]:
    """Annualised mean, Sharpe and t-stat of a monthly long-short return series."""
    mu, sd, n = ls.mean(), ls.std(), len(ls)
    return {
        "ann_return": mu * 12,
        "sharpe": mu / sd * np.sqrt(12) if sd > 0 else np.nan,
        "t_stat": mu / sd * np.sqrt(n) if sd > 0 else np.nan,
    }


# ── Signal combination & a costed long-short book (notebook 03) ──────────────

def rolling_ic_weights(ic_df: pd.DataFrame, lookback: int = 24,
                       min_periods: int = 12) -> pd.DataFrame:
    """
    Sign-aware factor weights with NO look-ahead: the weight used to form the book at month t is
    the trailing-mean IC over months [t-lookback, t-1]. The `shift(1)` is essential — a factor's IC
    at month t is only known at t+1 (it depends on the t→t+1 forward return), so using it to weight
    the t portfolio would leak the future.
    """
    return ic_df.shift(1).rolling(lookback, min_periods=min_periods).mean()


def combine_signals(neutral: dict[str, pd.DataFrame], weights: pd.DataFrame) -> pd.DataFrame:
    """Per month, linear-combine neutralised factors by the (lagged) weight row for that month."""
    names = list(neutral)
    out = {}
    for dt in weights.index:
        w = weights.loc[dt]
        if w.isna().all():
            continue
        w = w.fillna(0.0)
        out[dt] = sum(w[f] * neutral[f].loc[dt] for f in names)
    return pd.DataFrame(out).T


def long_short_book(factor_z: pd.DataFrame, fwd: pd.DataFrame, q: float = 0.2,
                    cost_per_side: float = 0.0010, min_names: int = 100) -> pd.DataFrame:
    """
    Equal-weight top-vs-bottom quantile book with turnover costs.

    Each month: long the top `q` fraction, short the bottom `q`, equal-weight within each leg
    (book is dollar-neutral, gross exposure 2). Returns a DataFrame indexed by rebalance month with
    columns ['gross', 'turnover', 'net'] where turnover is the two-sided sum |Δw| and
    net = gross − turnover · cost_per_side.
    """
    weights = {}
    for dt in factor_z.index:
        a = factor_z.loc[dt].dropna()
        if len(a) < min_names:
            continue
        hi = a[a >= a.quantile(1 - q)].index
        lo = a[a <= a.quantile(q)].index
        row = pd.Series(0.0, index=factor_z.columns)
        row[hi] = 1.0 / len(hi)
        row[lo] = -1.0 / len(lo)
        weights[dt] = row
    W = pd.DataFrame(weights).T
    gross = (W * fwd.reindex(W.index)).sum(axis=1)
    turnover = (W.fillna(0.0) - W.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = gross - turnover * cost_per_side
    return pd.DataFrame({"gross": gross, "turnover": turnover, "net": net})
