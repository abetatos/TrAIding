"""
Backtesting utilities: scoring rules, walk-forward CV, combinatorial purged CV.
Focused on binary prediction markets (probability forecasts).
"""
import numpy as np
import pandas as pd
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
