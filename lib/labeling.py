"""
Label generation for supervised ML on financial time series.
Based on AFML Ch. 3 (Lopez de Prado, 2018).
"""
import numpy as np
import pandas as pd


def daily_volatility(close: pd.Series, lookback: int = 100) -> pd.Series:
    """Annualised daily vol estimate using exponentially weighted std."""
    log_ret = np.log(close / close.shift(1)).dropna()
    return log_ret.ewm(span=lookback).std()


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """
    Triple-barrier labeling (AFML Ch. 3).

    Parameters
    ----------
    close   : price series indexed by datetime
    events  : DataFrame with columns ['t1', 'trgt'] indexed by event timestamps
              t1   = vertical barrier (expiry datetime)
              trgt = target return (e.g. daily_volatility * multiplier)
    pt_sl   : profit-take / stop-loss multipliers applied to trgt
    min_ret : minimum absolute return to assign a non-zero label

    Returns
    -------
    DataFrame with columns ['ret', 'bin'] (return and label -1/0/1).
    """
    out = []
    for t0, ev in events.iterrows():
        t1 = ev["t1"]
        trgt = ev["trgt"]

        pt = trgt * pt_sl[0] if pt_sl[0] > 0 else np.inf
        sl = -trgt * pt_sl[1] if pt_sl[1] > 0 else -np.inf

        path = close.loc[t0:t1]
        if len(path) < 2:
            continue

        path_ret = (path / close.at[t0]) - 1

        # First barrier touched
        sl_hit = path_ret[path_ret <= sl]
        pt_hit = path_ret[path_ret >= pt]

        t_sl = sl_hit.index[0] if not sl_hit.empty else t1
        t_pt = pt_hit.index[0] if not pt_hit.empty else t1
        t_exit = min(t_sl, t_pt, t1)

        ret = path_ret.at[t_exit]
        label = 0 if abs(ret) < min_ret else (1 if ret > 0 else -1)
        out.append({"t0": t0, "t1": t_exit, "ret": ret, "bin": label})

    return pd.DataFrame(out).set_index("t0")


def meta_labels(primary_labels: pd.Series, predicted_labels: pd.Series) -> pd.Series:
    """
    Meta-labeling (AFML Ch. 3): given a primary model's predictions, return
    1 where primary is correct (to size positions) and 0 where wrong.
    """
    return (primary_labels == predicted_labels).astype(int)
