"""
Multi-seed experiment harness.

House rule for this project: **no number appears in a conclusion unless it is the mean
over >= 30 seeds, reported with an interval.** Everything here exists to make that easy
and hard to forget.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

N_SEEDS = 30
SEEDS = list(range(N_SEEDS))


def mean_ci(x: Sequence[float], level: float = 0.95) -> dict:
    """Mean with a Student-t confidence interval for the mean."""
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    n = len(a)
    if n == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "sd": np.nan, "sem": np.nan, "n": 0}
    m, sd = a.mean(), a.std(ddof=1) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else 0.0
    half = stats.t.ppf(0.5 + level / 2, n - 1) * sem if n > 1 else 0.0
    return {"mean": m, "lo": m - half, "hi": m + half, "sd": sd, "sem": sem, "n": n}


def fmt_ci(d: dict, digits: int = 3, pct: bool = False) -> str:
    k = 100 if pct else 1
    suf = "%" if pct else ""
    return f"{d['mean']*k:.{digits}f}{suf} [{d['lo']*k:.{digits}f}, {d['hi']*k:.{digits}f}]"


def over_seeds(fn: Callable[[int], dict], seeds: Iterable[int] = SEEDS,
               progress: bool = False) -> pd.DataFrame:
    """Run `fn(seed) -> dict of scalars` for each seed; return one row per seed."""
    rows = []
    for s in seeds:
        r = fn(s)
        if r is None:
            continue
        rows.append({"seed": s, **r})
        if progress:
            print(f"  seed {s} done", end="\r")
    return pd.DataFrame(rows).set_index("seed")


def summarise(df: pd.DataFrame, level: float = 0.95, cols: Sequence[str] | None = None
              ) -> pd.DataFrame:
    """Column-wise mean / CI / sd table for an `over_seeds` result."""
    cols = list(cols or df.columns)
    out = {c: mean_ci(df[c].values, level) for c in cols
           if pd.api.types.is_numeric_dtype(df[c])}
    return pd.DataFrame(out).T[["mean", "lo", "hi", "sd", "n"]]


def paired_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired t-test on per-seed differences (same seeds => paired is the right test)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    t, p = stats.ttest_rel(a[ok], b[ok])
    ci = mean_ci(d)
    return {"diff": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "t": float(t), "p": float(p),
            "n": int(ok.sum())}


def one_sample_test(x: Sequence[float], mu0: float = 0.0) -> dict:
    """Is the per-seed mean different from mu0? (e.g. 'is AUC above 0.5?')"""
    a = np.asarray([v for v in x if np.isfinite(v)], float)
    t, p = stats.ttest_1samp(a, mu0)
    ci = mean_ci(a)
    return {"mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "t": float(t),
            "p": float(p), "n": len(a), "mu0": mu0}


# --------------------------------------------------------------------------------------
# selection bias: what "the best of K tries" is worth
# --------------------------------------------------------------------------------------
EULER = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected *maximum* Sharpe across `n_trials` independent strategies with zero true
    Sharpe (Bailey & Lopez de Prado, the SR0 of the deflated Sharpe ratio).

        E[max SR] ~ sr_std * [ (1-g) Z(1 - 1/K) + g Z(1 - 1/(K e)) ]

    This is the number every "we tried a few hundred alphas and this was the best one"
    result has to beat before it means anything.
    """
    if n_trials < 2 or not np.isfinite(sr_std) or sr_std <= 0:
        return 0.0
    k = float(n_trials)
    z1 = stats.norm.ppf(1.0 - 1.0 / k)
    z2 = stats.norm.ppf(1.0 - 1.0 / (k * np.e))
    return float(sr_std * ((1.0 - EULER) * z1 + EULER * z2))


def sharpe(returns: Sequence[float], periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe of a return series (no risk-free rate — everything here is
    long-short or per-bet excess)."""
    r = np.asarray([v for v in returns if np.isfinite(v)], float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def deflated_sharpe(returns: Sequence[float], n_trials: int, sr_std: float,
                    periods_per_year: float = 252.0) -> dict:
    """Probability that the observed Sharpe is real once you account for having looked at
    `n_trials` strategies, and for the non-normality of the returns (AFML Ch. 8).

    Returns the raw and deflated numbers together so the haircut is visible.
    """
    r = np.asarray([v for v in returns if np.isfinite(v)], float)
    n = len(r)
    sr_ann = sharpe(r, periods_per_year)
    if n < 3 or not np.isfinite(sr_ann):
        return {"sharpe": np.nan, "sr0": np.nan, "dsr": np.nan, "n": n}
    sr = sr_ann / np.sqrt(periods_per_year)                  # per-period
    sr0 = expected_max_sharpe(n_trials, sr_std / np.sqrt(periods_per_year))
    g3, g4 = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    return {"sharpe": sr_ann, "sr0": float(sr0 * np.sqrt(periods_per_year)),
            "dsr": float(stats.norm.cdf(z)), "n": n}
