"""
Cross-validation for overlapping labels (AFML Ch. 7), from scratch.

The failure mode this file exists for: when the label at t0 is decided by returns over
(t0, t1], two observations sampled close together share returns. A shuffled k-fold puts one
in train and its near-twin in test, and the model is scored on data it has effectively seen.

Two defences, implemented separately so their contributions can be told apart:

  * PURGING  — drop from the training set any observation whose label window overlaps the
               test window in time.
  * EMBARGO  — additionally drop training observations that *start* within a short window
               after the test set ends, to break residual serial correlation in the features
               (purging alone does not remove it: a feature can be a 50-bar rolling mean).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def random_kfold_splits(n: int, n_splits: int = 5, seed: int = 0
                        ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """The leaky baseline: shuffle rows, then cut into folds."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    for test in np.array_split(idx, n_splits):
        yield np.setdiff1d(np.arange(n), test), np.sort(test)


def contiguous_kfold_splits(n: int, n_splits: int = 5
                            ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Time-contiguous folds with NO purging — isolates what purging alone buys."""
    for test in np.array_split(np.arange(n), n_splits):
        yield np.setdiff1d(np.arange(n), test), test


def purged_kfold_splits(
    t0: pd.Series | np.ndarray,
    t1: pd.Series | np.ndarray,
    n_splits: int = 5,
    embargo_bars: int = 0,
    bar_index: pd.DatetimeIndex | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Purged (and optionally embargoed) k-fold over time-contiguous test folds.

    Parameters
    ----------
    t0, t1       : start and end timestamp of each observation's label window, in time order
    n_splits     : number of contiguous test folds
    embargo_bars : after purging, also drop training observations starting within this many
                   *bars* of the end of the test window. Needs `bar_index` to convert bars
                   to timestamps; with 0 the embargo is off.
    """
    t0 = pd.Series(pd.to_datetime(np.asarray(t0)))
    t1 = pd.Series(pd.to_datetime(np.asarray(t1)))
    n = len(t0)
    if embargo_bars > 0 and bar_index is None:
        raise ValueError("embargo_bars > 0 requires bar_index")

    for test in np.array_split(np.arange(n), n_splits):
        test_start, test_end = t0.iloc[test[0]], t1.iloc[test].max()

        # PURGE: keep only training windows entirely before or entirely after the test window
        keep = (t1.values < np.datetime64(test_start)) | (t0.values > np.datetime64(test_end))

        # EMBARGO: push the "entirely after" boundary forward by embargo_bars
        if embargo_bars > 0:
            k = bar_index.searchsorted(test_end)
            emb_end = bar_index[min(k + embargo_bars, len(bar_index) - 1)]
            keep &= (t1.values < np.datetime64(test_start)) | (t0.values > np.datetime64(emb_end))

        train = np.where(keep)[0]
        train = np.setdiff1d(train, test)
        yield train, test


def out_of_time_split(n: int, test_frac: float = 0.3, gap: int = 0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """The honest reference: train on the first part, test on the last part, with a gap of
    `gap` observations between them so no window can straddle the boundary."""
    n_test = int(n * test_frac)
    test = np.arange(n - n_test, n)
    train = np.arange(0, max(n - n_test - gap, 0))
    return train, test


def split_diagnostics(t0, t1, splits) -> pd.DataFrame:
    """Per fold, the quantity that actually measures leakage: for each *test row*, how many
    *training rows* share bars with it. This is the same count notebook 03 measured by
    injecting a shock — here it is counted directly on the split."""
    a0 = pd.to_datetime(np.asarray(t0)).values
    a1 = pd.to_datetime(np.asarray(t1)).values
    rows = []
    for k, (tr, te) in enumerate(splits):
        # windows overlap iff  train_t0 <= test_t1  and  train_t1 >= test_t0
        ov = ((a0[tr][:, None] <= a1[te][None, :]) &
              (a1[tr][:, None] >= a0[te][None, :]))
        per_test = ov.sum(axis=0)
        rows.append({"fold": k, "n_train": len(tr), "n_test": len(te),
                     "train_overlaps_per_test_row": per_test.mean(),
                     "test_rows_contaminated": float((per_test > 0).mean())})
    return pd.DataFrame(rows).set_index("fold")


def walk_forward_gapped(n: int, n_splits: int = 5, gap: int = 0, min_train_frac: float = 0.25
                        ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward with a gap: train on everything up to `gap`
    observations before each test block. No training row can share a bar with a test row,
    and — unlike k-fold — the model never sees anything from the future. This is the
    reference every CV scheme is scored against."""
    start = int(n * min_train_frac)
    edges = np.linspace(start, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        test = np.arange(edges[k], edges[k + 1])
        train_end = max(edges[k] - gap, 0)
        if train_end < 30:
            continue
        yield np.arange(0, train_end), test
