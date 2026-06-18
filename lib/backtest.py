"""
Backtesting utilities: scoring rules, walk-forward CV, combinatorial purged CV.
Focused on binary prediction markets (probability forecasts).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from sklearn.model_selection import KFold


# ── Scoring rules ────────────────────────────────────────────────────────────

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: MSE between forecast probability and binary outcome."""
    return float(np.mean((y_prob - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """BSS relative to climatological baseline. 1=perfect, 0=no skill, <0=worse than baseline."""
    bs = brier_score(y_true, y_prob)
    bs_ref = brier_score(y_true, np.full_like(y_prob, y_true.mean()))
    return 1 - bs / bs_ref if bs_ref > 0 else np.nan


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


# ── Calibration ──────────────────────────────────────────────────────────────

def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin forecasts and compute mean forecast vs actual resolution rate per bin.
    Returns DataFrame usable for a calibration plot.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_bins)]
    bucket = pd.cut(y_prob, bins=bins, labels=labels, include_lowest=True)

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "bucket": bucket})
    result = (
        df.groupby("bucket", observed=True)
        .agg(
            mean_forecast=("y_prob", "mean"),
            actual_rate=("y_true", "mean"),
            count=("y_true", "size"),
        )
        .reset_index()
    )
    return result


# ── Walk-forward cross-validation ────────────────────────────────────────────

def walk_forward_splits(
    n: int,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Purged walk-forward splits. Each test fold comes strictly after training.
    embargo_pct: fraction of samples to drop between train and test to avoid leakage.
    """
    fold_size = n // (n_splits + 1)
    embargo = max(1, int(n * embargo_pct))
    splits = []
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_start = train_end + embargo
        test_end = test_start + fold_size
        if test_end > n:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    return splits


# ── Purged K-Fold cross-validation (AFML Ch. 7) ──────────────────────────────

def purged_kfold_splits(
    t1: pd.Series,
    n_splits: int = 5,
    embargo_pct: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Purged K-Fold with embargo (AFML Ch. 7, snippet 7.3). Removes from the training set any
    label whose horizon [t0, t1] overlaps the test fold (purge), plus an embargo of observations
    immediately after the test fold. Without this, overlapping labels leak the test outcome.

    t1 : Series indexed by event-start time, values = event-end time (the label horizon).
    Returns positional (train_idx, test_idx) arrays.
    """
    t1 = t1.sort_index()
    n = len(t1)
    indices = np.arange(n)
    mbrg = int(n * embargo_pct)
    test_ranges = [(g[0], g[-1] + 1) for g in np.array_split(indices, n_splits)]

    splits = []
    for i, j in test_ranges:
        test_idx = indices[i:j]
        t0 = t1.index[i]                       # test fold start time
        max_t1 = t1.iloc[i:j].max()            # latest horizon end inside the test fold
        max_t1_idx = t1.index.searchsorted(max_t1)

        # Train = labels that close at/before the test starts (no forward overlap) ...
        train_idx = indices[(t1 <= t0).to_numpy()]
        # ... plus labels that start after the test horizon ends, past the embargo.
        if max_t1_idx < n:
            train_idx = np.concatenate((train_idx, indices[max_t1_idx + mbrg:]))
        splits.append((train_idx, test_idx))
    return splits


# ── Sharpe ratio inference (AFML Ch. 14) ─────────────────────────────────────

def sharpe_ratio(returns: np.ndarray) -> float:
    """Non-annualised Sharpe ratio of a per-observation return series."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (AFML Ch. 14): P(true SR > benchmark) given the estimated SR,
    sample length and the return distribution's skew/kurtosis. Higher = more confident the SR
    is real rather than a small-sample / non-normal artefact.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return np.nan
    sr = sharpe_ratio(r)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))  # non-excess (normal = 3)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int, sr_trials_std: float) -> float:
    """
    Deflated Sharpe Ratio (AFML Ch. 14): PSR against the *expected maximum* SR that `n_trials`
    independent strategies would produce by chance (variance `sr_trials_std**2`). This is the
    honest test after a parameter search — it penalises selection bias / multiple testing.
    """
    if n_trials < 1 or sr_trials_std <= 0:
        return np.nan
    emc = 0.5772156649  # Euler-Mascheroni
    sr0 = sr_trials_std * (
        (1 - emc) * norm.ppf(1 - 1.0 / n_trials)
        + emc * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)


# ── Forward return analysis ──────────────────────────────────────────────────

def forward_returns(
    prices: pd.Series,
    signal_dates: pd.Index,
    horizons: list[int] = [1, 5, 10],
) -> pd.DataFrame:
    """
    For each signal date, compute forward returns at given horizons (in trading days).
    """
    rows = []
    for dt in signal_dates:
        if dt not in prices.index:
            continue
        p0 = prices.at[dt]
        row = {"date": dt}
        for h in horizons:
            idx = prices.index.get_loc(dt)
            if idx + h < len(prices):
                row[f"fwd_{h}d"] = prices.iloc[idx + h] / p0 - 1
            else:
                row[f"fwd_{h}d"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")
