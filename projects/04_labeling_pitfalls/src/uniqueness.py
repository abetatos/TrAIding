"""
Concurrency, uniqueness and sample weights (AFML Ch. 4), from scratch.

The problem: a label at t0 is decided by what the price does over (t0, t1]. If two events
are sampled a few bars apart, their windows overlap and their labels are driven by *the same
returns*. They are not two observations, they are one and a bit. Everything in this file is
about measuring that "and a bit".

Convention note: AFML counts an event as active over the closed interval [t0, t1]. That is
what `num_concurrent_events` does. Using the half-open (t0, t1] — the bars whose returns
actually enter the label — shifts average uniqueness by O(1/N); `include_t0=False` allows it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _positions(bar_index: pd.DatetimeIndex, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(i0, i1) integer positions for each event window."""
    if "i0" in events.columns and "i1" in events.columns:
        return events["i0"].values.astype(int), events["i1"].values.astype(int)
    pos = pd.Series(np.arange(len(bar_index)), index=bar_index)
    return pos.reindex(events.index).values.astype(int), pos.reindex(events["t1"]).values.astype(int)


def indicator_matrix(bar_index: pd.DatetimeIndex, events: pd.DataFrame,
                     include_t0: bool = True) -> pd.DataFrame:
    """Binary (bars x events) matrix: 1 where event j is alive at bar t (AFML 4.3).

    Only build this for plots or small problems — it is O(bars x events) in memory.
    """
    i0, i1 = _positions(bar_index, events)
    m = np.zeros((len(bar_index), len(events)), dtype=np.int8)
    for j, (a, b) in enumerate(zip(i0, i1)):
        m[(a if include_t0 else a + 1): b + 1, j] = 1
    return pd.DataFrame(m, index=bar_index, columns=events.index)


def num_concurrent_events(bar_index: pd.DatetimeIndex, events: pd.DataFrame,
                          include_t0: bool = True) -> pd.Series:
    """How many event windows are alive at each bar (AFML 4.1). O(n) via a difference array."""
    i0, i1 = _positions(bar_index, events)
    d = np.zeros(len(bar_index) + 1)
    np.add.at(d, i0 if include_t0 else i0 + 1, 1.0)
    np.add.at(d, i1 + 1, -1.0)
    return pd.Series(np.cumsum(d)[:-1], index=bar_index, name="concurrency")


def average_uniqueness(bar_index: pd.DatetimeIndex, events: pd.DataFrame,
                       co_events: pd.Series | None = None,
                       include_t0: bool = True) -> pd.Series:
    """Per-event mean of 1/concurrency over its own window (AFML 4.2). In [0, 1];
    1 means "no other event shares a single bar with me"."""
    if co_events is None:
        co_events = num_concurrent_events(bar_index, events, include_t0)
    inv = np.where(co_events.values > 0, 1.0 / np.maximum(co_events.values, 1e-12), 0.0)
    cum = np.concatenate(([0.0], np.cumsum(inv)))
    i0, i1 = _positions(bar_index, events)
    a = i0 if include_t0 else i0 + 1
    span = np.maximum(i1 - a + 1, 1)
    return pd.Series((cum[i1 + 1] - cum[a]) / span, index=events.index, name="uniqueness")


def return_attribution_weights(bar_index: pd.DatetimeIndex, events: pd.DataFrame,
                               close: pd.Series, co_events: pd.Series | None = None,
                               normalise: bool = True) -> pd.Series:
    """AFML 4.10: weight an observation by the absolute log-return attributable to it,
    each bar's return split across the events alive at that bar.

        w_i = | sum_{t in window_i} r_t / c_t |
    """
    if co_events is None:
        co_events = num_concurrent_events(bar_index, events)
    lr = np.log(close).diff().fillna(0.0).values
    c = np.maximum(co_events.values, 1e-12)
    contrib = np.concatenate(([0.0], np.cumsum(lr / c)))
    i0, i1 = _positions(bar_index, events)
    w = np.abs(contrib[i1 + 1] - contrib[i0 + 1])
    w = pd.Series(w, index=events.index, name="w_return")
    return w * len(w) / w.sum() if normalise and w.sum() > 0 else w


def uniqueness_weights(av_uniqueness: pd.Series, normalise: bool = True) -> pd.Series:
    """The simple alternative: weight proportional to average uniqueness."""
    w = av_uniqueness.copy().rename("w_uniqueness")
    return w * len(w) / w.sum() if normalise and w.sum() > 0 else w


def effective_sample_size(av_uniqueness: pd.Series) -> float:
    """n * mean(uniqueness): how many *non-overlapping* observations this sample is worth."""
    return float(len(av_uniqueness) * av_uniqueness.mean())


def kish_ess(weights: pd.Series) -> float:
    """Kish's effective sample size for a weighted sample: (sum w)^2 / sum w^2.
    A second, independent way to ask 'how many observations is this really?'."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0


def sampling_schemes(close: pd.Series, sigma: pd.Series, every: int = 1) -> pd.DatetimeIndex:
    """One observation every `every` bars — the trivial scheme, kept for symmetry."""
    return close.index[::every]
