"""
Information-driven bar construction from a trade tape (AFML Ch. 2, Lopez de Prado 2018).

Input everywhere is a *trades* DataFrame with columns:
    ts    : datetime64[ns, UTC]   trade timestamp
    price : float                  trade price
    qty   : float                  trade size (base units)
    sign  : int  (+1 / -1)         aggressor side (+1 buy, -1 sell)

`sign` should come from the true aggressor flag (e.g. Binance `is_buyer_maker`) when available,
which is more accurate than the tick rule AFML falls back to.

Bar families
------------
- Standard bars      : tick / volume / dollar   (fixed threshold on a measure)
- Imbalance bars     : TIB / VIB / DIB          (adaptive: sample on cumulative signed imbalance)
- Run bars           : TRB / VRB / DRB          (adaptive: sample on dominant-side run)

The adaptive bars use EWMA estimates of (a) the expected bar length E[T] and (b) the expected
per-tick imbalance. They are known to be sensitive to initialisation and can degenerate into
1-tick or never-closing bars, so `min_ticks` / `max_ticks` guards are exposed and on by default.
"""
import numpy as np
import pandas as pd

# Per-tick "measure" underlying each bar kind
_KINDS = ("tick", "volume", "dollar")


def _measure(trades: pd.DataFrame, kind: str) -> np.ndarray:
    """Non-negative per-tick measure: 1 (tick), qty (volume), price*qty (dollar)."""
    if kind == "tick":
        return np.ones(len(trades), dtype=float)
    if kind == "volume":
        return trades["qty"].to_numpy(float)
    if kind == "dollar":
        return (trades["price"].to_numpy(float) * trades["qty"].to_numpy(float))
    raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")


_BAR_COLS = ["t_open", "open", "high", "low", "close", "volume", "dollar",
             "vwap", "n_ticks", "buy_volume", "sell_volume"]


def _aggregate(trades: pd.DataFrame, ends: list[int]) -> pd.DataFrame:
    """
    Aggregate trades into OHLCV bars given inclusive end positions of each bar.
    Vectorised (cumsum for sums, reduceat for high/low) so it scales to tens of
    millions of trades — a Python per-bar loop here is the main perf trap.
    """
    ends = np.asarray(ends, dtype=np.int64)
    if ends.size == 0:
        return pd.DataFrame(columns=_BAR_COLS, index=pd.DatetimeIndex([], name="t_close"))

    last = int(ends[-1])
    ts = trades["ts"].to_numpy()[:last + 1]
    px = trades["price"].to_numpy(float)[:last + 1]
    qty = trades["qty"].to_numpy(float)[:last + 1]
    sign = trades["sign"].to_numpy(int)[:last + 1]

    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1

    dollar = px * qty
    buyv = np.where(sign > 0, qty, 0.0)
    cum_v = np.concatenate(([0.0], np.cumsum(qty)))
    cum_d = np.concatenate(([0.0], np.cumsum(dollar)))
    cum_b = np.concatenate(([0.0], np.cumsum(buyv)))

    vol = cum_v[ends + 1] - cum_v[starts]
    dol = cum_d[ends + 1] - cum_d[starts]
    bvol = cum_b[ends + 1] - cum_b[starts]

    out = pd.DataFrame(
        {
            "t_open": ts[starts],
            "open": px[starts],
            "high": np.maximum.reduceat(px, starts),
            "low": np.minimum.reduceat(px, starts),
            "close": px[ends],
            "volume": vol,
            "dollar": dol,
            "vwap": np.where(vol > 0, dol / vol, px[ends]),
            "n_ticks": (ends - starts + 1).astype(int),
            "buy_volume": bvol,
            "sell_volume": vol - bvol,
        },
        index=pd.DatetimeIndex(ts[ends], name="t_close"),
    )
    return out


# ── Time bars (baseline to beat) ─────────────────────────────────────────────

def time_bars(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Clock-time bars (the classic baseline). Index is the bin's left edge.
    AFML's point: time bars oversample quiet periods and undersample active ones,
    giving worse statistical properties than information-driven bars.
    """
    t = trades.set_index("ts")
    px, v, s = t["price"], t["qty"], t["sign"]
    dollar = px * v
    g = px.resample(freq)
    out = pd.DataFrame({
        "open": g.first(), "high": g.max(), "low": g.min(), "close": g.last(),
        "volume": v.resample(freq).sum(), "dollar": dollar.resample(freq).sum(),
        "n_ticks": px.resample(freq).count(),
        "buy_volume": v[s > 0].resample(freq).sum().reindex(g.first().index, fill_value=0),
    })
    out = out[out["n_ticks"] > 0].copy()
    out["vwap"] = (out["dollar"] / out["volume"]).where(out["volume"] > 0, out["close"])
    out["sell_volume"] = out["volume"] - out["buy_volume"]
    return out


# ── Standard bars ────────────────────────────────────────────────────────────

def standard_bars(
    trades: pd.DataFrame,
    kind: str = "dollar",
    threshold: float | None = None,
    target_n_bars: int | None = None,
) -> pd.DataFrame:
    """
    Fixed-threshold tick / volume / dollar bars.

    Provide either `threshold` (absolute, in measure units) or `target_n_bars`
    (threshold is then total_measure / target_n_bars). Defaults to ~100 bars.
    """
    m = _measure(trades, kind)
    cum = np.cumsum(m)
    total = cum[-1] if len(cum) else 0.0

    if threshold is None:
        target_n_bars = target_n_bars or 100
        threshold = total / max(target_n_bars, 1)

    n_full = int(total // threshold)
    if n_full < 1:
        ends = [len(trades) - 1] if len(trades) else []
    else:
        levels = np.arange(1, n_full + 1) * threshold
        ends = np.searchsorted(cum, levels, side="left").tolist()
        ends = sorted(set(min(e, len(trades) - 1) for e in ends))
    return _aggregate(trades, ends)


# ── Imbalance bars (TIB / VIB / DIB) ─────────────────────────────────────────

def imbalance_bars(
    trades: pd.DataFrame,
    kind: str = "dollar",
    init_T: int = 1000,
    T_ewma_span: int = 50,
    imb_ewma_span: int = 200,
    min_ticks: int = 10,
    max_ticks: int = 200_000,
) -> pd.DataFrame:
    """
    Imbalance bars (AFML Ch. 2.3.2.1). Sample a bar when the cumulative *signed* imbalance
    |theta_T| exceeds its expectation E[T] * E[|imbalance per tick|], both tracked by EWMA.

    init_T        : seed for E[T] (expected ticks per bar) before the first bar closes
    T_ewma_span   : EWMA span for updating E[T]
    imb_ewma_span : EWMA span for updating expected per-tick imbalance
    min_ticks / max_ticks : guards against degenerate (1-tick) or runaway bars
    """
    signed = trades["sign"].to_numpy(float) * _measure(trades, kind)
    n = len(signed)
    if n == 0:
        return _aggregate(trades, [])

    init = min(init_T, n)
    E_T = float(init_T)
    # Expected *signed* imbalance per tick (AFML: (2P[b=1]-1)*E[v]); ~0 for a balanced tape.
    E_imb = float(np.mean(signed[:init])) or 1e-12
    a_T = 2.0 / (T_ewma_span + 1)
    a_i = 2.0 / (imb_ewma_span + 1)

    ends: list[int] = []
    cum = 0.0
    count = 0
    for i in range(n):
        cum += signed[i]
        count += 1
        threshold = E_T * abs(E_imb)
        if count >= min_ticks and (abs(cum) >= threshold or count >= max_ticks):
            ends.append(i)
            E_T = a_T * count + (1 - a_T) * E_T
            E_imb = a_i * (cum / count) + (1 - a_i) * E_imb  # realized signed per-tick imbalance
            cum, count = 0.0, 0
    return _aggregate(trades, ends)


# ── Run bars (TRB / VRB / DRB) ───────────────────────────────────────────────

def run_bars(
    trades: pd.DataFrame,
    kind: str = "dollar",
    init_T: int = 1000,
    T_ewma_span: int = 50,
    run_ewma_span: int = 200,
    min_ticks: int = 10,
    max_ticks: int = 200_000,
) -> pd.DataFrame:
    """
    Run bars (AFML Ch. 2.3.2.2). Sample when the dominant-side cumulative measure
    max(buy, sell) exceeds E[T] * max(expected buy share, expected sell share).
    """
    m = _measure(trades, kind)
    sign = trades["sign"].to_numpy(int)
    n = len(m)
    if n == 0:
        return _aggregate(trades, [])

    init = min(init_T, n)
    # Expected dominant-side measure *per tick* (AFML: P[b]·E[v|b]); same units as theta.
    E_buy = m[:init][sign[:init] > 0].sum() / init
    E_sell = m[:init][sign[:init] < 0].sum() / init
    E_T = float(init_T)
    a_T = 2.0 / (T_ewma_span + 1)
    a_r = 2.0 / (run_ewma_span + 1)

    ends: list[int] = []
    buy = sell = 0.0
    count = 0
    for i in range(n):
        if sign[i] > 0:
            buy += m[i]
        else:
            sell += m[i]
        count += 1
        theta = max(buy, sell)
        threshold = E_T * max(E_buy, E_sell)
        if count >= min_ticks and (theta >= threshold or count >= max_ticks):
            ends.append(i)
            E_T = a_T * count + (1 - a_T) * E_T
            E_buy = a_r * (buy / count) + (1 - a_r) * E_buy
            E_sell = a_r * (sell / count) + (1 - a_r) * E_sell
            buy = sell = 0.0
            count = 0
    return _aggregate(trades, ends)


# ── Dispatcher ───────────────────────────────────────────────────────────────

def build_bars(trades: pd.DataFrame, method: str = "dollar", **kwargs) -> pd.DataFrame:
    """
    Configurable entry point. `method` is "<kind>" or "<kind>_imbalance" / "<kind>_run",
    e.g. "dollar", "volume_imbalance", "tick_run". Extra kwargs pass through.
    """
    if method == "time":
        return time_bars(trades, **kwargs)
    if method.endswith("_imbalance"):
        return imbalance_bars(trades, kind=method[:-10], **kwargs)
    if method.endswith("_run"):
        return run_bars(trades, kind=method[:-4], **kwargs)
    return standard_bars(trades, kind=method, **kwargs)


# ── Diagnostics: which bar is "best"? ────────────────────────────────────────

def bar_diagnostics(bars: pd.DataFrame) -> dict:
    """
    Statistical quality of a bar series (AFML §2.4 motivation). Better bars yield returns that
    are closer to IID-normal and a more stable sampling frequency:

      ret_autocorr1 : serial correlation of bar log-returns  → closer to 0 is better
      jb_stat       : Jarque-Bera statistic                  → lower is better (more normal)
      ret_skew / ret_kurt_excess : distribution shape
      ticks_cv      : coeff. of variation of ticks/bar       → lower = more uniform bars
    """
    from scipy.stats import jarque_bera

    if len(bars) < 3:
        return {"n_bars": len(bars)}

    r = np.log(bars["close"]).diff().dropna()
    jb = jarque_bera(r) if len(r) > 2 else (np.nan, np.nan)
    ticks = bars["n_ticks"].to_numpy(float)
    return {
        "n_bars": int(len(bars)),
        "ret_autocorr1": float(r.autocorr(1)) if len(r) > 2 else np.nan,
        "jb_stat": float(jb[0]),
        "ret_skew": float(r.skew()),
        "ret_kurt_excess": float(r.kurt()),
        "mean_ticks": float(ticks.mean()),
        "ticks_cv": float(ticks.std() / ticks.mean()) if ticks.mean() else np.nan,
    }
