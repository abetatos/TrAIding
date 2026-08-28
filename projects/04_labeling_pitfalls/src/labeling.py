"""
Labeling schemes, implemented from scratch (numpy/pandas only).

Two families:
  * fixed horizon  — look h bars ahead, compare the return to a threshold
  * triple barrier — first of {profit-take, stop-loss, expiry} decides the label

The triple barrier is implemented with three *touch-detection* modes, which is the
whole point of notebook 02:

    mode="close"  only bar closes are inspected  (the textbook implementation)
    mode="ohlc"   bar highs/lows are inspected   (both barriers can hit in one bar
                  -> `tie` decides, because OHLC does not say which came first)
    mode="path"   the 1-minute path is inspected (exact; only available because we
                  simulated the data ourselves)

`mode="path"` is the ground truth against which the other two are scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# volatility target
# --------------------------------------------------------------------------------------
def ewma_vol(close: pd.Series, span: int = 50) -> pd.Series:
    """Per-bar volatility of log returns, exponentially weighted (AFML's `getDailyVol`
    in spirit, but on a bar index rather than calendar days)."""
    lr = np.log(close).diff()
    return lr.ewm(span=span, min_periods=span // 2).std()


# --------------------------------------------------------------------------------------
# fixed horizon
# --------------------------------------------------------------------------------------
def fixed_horizon_labels(
    close: pd.Series,
    h: int = 20,
    threshold: float | None = None,
    sigma: pd.Series | None = None,
    mult: float = 1.0,
    scale_sqrt_h: bool = True,
) -> pd.DataFrame:
    """Label bar t by the sign of the h-bar-ahead return vs a threshold.

    threshold given  -> fixed absolute threshold (the naive version)
    sigma given      -> threshold = mult * sigma_t * sqrt(h)  (vol-scaled version)
    neither          -> threshold 0, pure sign
    """
    ret = close.shift(-h) / close - 1.0
    if threshold is not None:
        thr = pd.Series(float(threshold), index=close.index)
    elif sigma is not None:
        thr = mult * sigma * (np.sqrt(h) if scale_sqrt_h else 1.0)
    else:
        thr = pd.Series(0.0, index=close.index)

    lab = np.sign(ret).where(ret.abs() > thr, 0.0)
    t1 = pd.Series(close.index[np.minimum(np.arange(len(close)) + h, len(close) - 1)],
                   index=close.index)
    out = pd.DataFrame({"t1": t1, "ret": ret, "thr": thr, "bin": lab})
    return out.iloc[: len(close) - h]          # drop events without a full horizon


# --------------------------------------------------------------------------------------
# triple barrier
# --------------------------------------------------------------------------------------
def triple_barrier(
    bars: pd.DataFrame,
    t_events: pd.DatetimeIndex,
    trgt: pd.Series,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    n_bars: int = 20,
    mode: str = "close",
    fine: np.ndarray | None = None,
    vertical_label: str = "zero",
    tie: str = "conservative",
    min_trgt: float = 1e-6,
) -> pd.DataFrame:
    """First-touch labeling.

    Parameters
    ----------
    bars           : DataFrame with open/high/low/close indexed by bar timestamp
    t_events       : timestamps at which a bet is considered
    trgt           : per-bar volatility target (barrier width unit)
    pt_sl          : (profit-take, stop-loss) multipliers on `trgt`; 0 disables that side
    n_bars         : vertical barrier, in bars after the event
    mode           : "close" | "ohlc" | "path"
    fine           : (n_bars_total, M+1) intra-bar price matrix, required for mode="path"
    vertical_label : "zero" -> expiry gets label 0; "sign" -> sign of the realised return
    tie            : mode="ohlc" only, when both barriers fall inside the same bar:
                     "conservative" -> stop-loss wins, "optimistic" -> profit-take wins,
                     "close" -> whichever side the bar closed on

    Returns
    -------
    DataFrame indexed by event time with columns
    ['t1', 't1_vert', 'ret', 'bin', 'barrier', 'trgt', 'i0', 'i1'].
    """
    if mode not in {"close", "ohlc", "path"}:
        raise ValueError(mode)
    if mode == "path" and fine is None:
        raise ValueError("mode='path' needs the fine intra-bar matrix")

    idx = bars.index
    pos = pd.Series(np.arange(len(idx)), index=idx)
    close = bars["close"].values
    high = bars["high"].values
    low = bars["low"].values
    n = len(idx)
    n_fine = fine.shape[1] if fine is not None else 0

    if mode == "path":
        cmax = np.maximum.accumulate(fine, axis=1)
        neg_cmin = -np.minimum.accumulate(fine, axis=1)

    ev_pos = pos.reindex(t_events).dropna().astype(int).values
    tg = trgt.reindex(idx).values

    rows = []
    for i0 in ev_pos:
        t = tg[i0]
        if not np.isfinite(t) or t < min_trgt:
            continue
        i_vert = min(i0 + n_bars, n - 1)
        if i_vert <= i0:
            continue
        c0 = close[i0]
        pt = pt_sl[0] * t if pt_sl[0] > 0 else np.inf
        sl = -pt_sl[1] * t if pt_sl[1] > 0 else -np.inf
        p_pt, p_sl = c0 * (1 + pt), c0 * (1 + sl)

        i1, barrier, ret = i_vert, "vert", close[i_vert] / c0 - 1.0
        for j in range(i0 + 1, i_vert + 1):
            if mode == "close":
                r = close[j] / c0 - 1.0
                hit_pt, hit_sl = r >= pt, r <= sl
                if hit_pt or hit_sl:          # a single close cannot be on both sides
                    i1, barrier, ret = j, ("pt" if hit_pt else "sl"), r
                    break
            elif mode == "ohlc":
                hit_pt = high[j] >= p_pt
                hit_sl = low[j] <= p_sl
                if hit_pt and hit_sl:
                    if tie == "conservative":
                        win = "sl"
                    elif tie == "optimistic":
                        win = "pt"
                    elif tie == "close":
                        win = "pt" if close[j] >= c0 else "sl"
                    else:
                        raise ValueError(tie)
                    i1, barrier, ret = j, win, (pt if win == "pt" else sl)
                    break
                if hit_pt or hit_sl:
                    i1, barrier = j, ("pt" if hit_pt else "sl")
                    ret = pt if hit_pt else sl
                    break
            else:  # path — exact first crossing inside the bar
                k_pt = np.searchsorted(cmax[j], p_pt, side="left") if pt_sl[0] > 0 else n_fine
                k_sl = np.searchsorted(neg_cmin[j], -p_sl, side="left") if pt_sl[1] > 0 else n_fine
                if k_pt >= n_fine and k_sl >= n_fine:
                    continue
                i1, barrier, ret = (j, "pt", pt) if k_pt <= k_sl else (j, "sl", sl)
                break

        if barrier == "vert":
            lab = 0.0 if vertical_label == "zero" else float(np.sign(ret))
        else:
            lab = 1.0 if barrier == "pt" else -1.0

        rows.append((idx[i0], idx[i1], idx[i_vert], ret, lab, barrier, t, i0, i1))

    out = pd.DataFrame(
        rows, columns=["t0", "t1", "t1_vert", "ret", "bin", "barrier", "trgt", "i0", "i1"]
    ).set_index("t0")
    out.index.name = "t0"
    return out


# --------------------------------------------------------------------------------------
# event sampling
# --------------------------------------------------------------------------------------
def cusum_events(close: pd.Series, h) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter (AFML Ch. 2): fire when the cumulative log-return since
    the last event exceeds +/- h. `h` may be a scalar or a per-bar Series (k * sigma)."""
    lr = np.log(close).diff().fillna(0.0)
    hh = pd.Series(h, index=close.index) if np.isscalar(h) else h.reindex(close.index)
    s_pos = s_neg = 0.0
    events = []
    for t, r in lr.items():
        thr = hh.at[t]
        if not np.isfinite(thr):
            continue
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos > thr:
            s_pos = 0.0
            events.append(t)
        elif s_neg < -thr:
            s_neg = 0.0
            events.append(t)
    return pd.DatetimeIndex(events, name="date")
