"""
Meta-labeling (AFML Ch. 3.6), from scratch.

The split the chapter proposes: a **primary** model decides the *side* of the bet, and a
**secondary** ("meta") model decides the *size* — in the binary case, whether to take the bet
at all. The meta-label of an event is therefore not "did the price go up" but **"was the
primary model right"**, which is a different, easier and strictly conditional question.

The claim under test is the one the book makes explicitly: that this arrangement "helps
achieve high F1 scores" — build a primary with high recall, then let the secondary fix the
precision. Two things have to be separated to check it:

  * whether the *classification* metric improves  (F1, precision, recall)
  * whether the *money* improves                  (Sharpe / mean return per bet)

They are not the same question, and section 3 of notebook 05 is about how far apart they get.

Barrier convention. With symmetric barriers (pt_sl = (k, k)) a sided barrier set is the same
set of price levels as an unsided one, so the meta-label can be read straight off a symmetric
`triple_barrier` run: the primary is right iff `side * ret > 0`. With asymmetric multipliers
that is no longer true — the profit-take sits at a different distance than the stop — and
`sided_triple_barrier` below does the honest thing instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from labeling import triple_barrier


# --------------------------------------------------------------------------------------
# primary models (the thing that picks the side)
# --------------------------------------------------------------------------------------
def rule_side(features: pd.DataFrame, index: pd.Index, col: str = "mom_5") -> pd.Series:
    """A realistic primary: bet with the sign of a momentum feature. No ground truth is used,
    it is right about as often as a coin unless the generator happens to reward momentum."""
    s = np.sign(features[col].reindex(index).fillna(0.0))
    return s.replace(0.0, 1.0).rename("side")


def oracle_side(truth: pd.DataFrame, index: pd.Index, accuracy: float,
                rng: np.random.Generator, drift_col: str = "drift_bar") -> pd.Series:
    """A primary of *tunable quality*: it points at the true drift with probability
    `accuracy` and the other way otherwise. Where there is no drift its sign is a coin, which
    is the point — the meta model's job is to learn to sit those out."""
    d = truth[drift_col].reindex(index).values
    true_side = np.where(d > 0, 1.0, np.where(d < 0, -1.0, 0.0))
    coin = rng.choice([-1.0, 1.0], size=len(index))
    true_side = np.where(true_side == 0.0, coin, true_side)
    flip = rng.random(len(index)) > accuracy
    return pd.Series(np.where(flip, -true_side, true_side), index=index, name="side")


# --------------------------------------------------------------------------------------
# meta-labels
# --------------------------------------------------------------------------------------
def meta_labels(events: pd.DataFrame, side: pd.Series) -> pd.DataFrame:
    """Attach the primary's side and the binary meta-label to a triple-barrier run.

    `bin_meta` = 1 if the bet made money, 0 otherwise. Events whose realised return is exactly
    zero are impossible here (continuous prices) but would count as 0.
    """
    s = side.reindex(events.index)
    ret_side = events["ret"] * s
    out = events.copy()
    out["side"] = s
    out["ret_side"] = ret_side
    out["bin_meta"] = (ret_side > 0).astype(int)
    return out


def sided_triple_barrier(bars: pd.DataFrame, side: pd.Series, trgt: pd.Series,
                         pt_sl: tuple[float, float] = (1.0, 1.0), n_bars: int = 20,
                         fine: np.ndarray | None = None, mode: str = "path") -> pd.DataFrame:
    """Triple barrier where the profit-take/stop-loss sides follow the primary's direction.

    For a short bet the profit-take is *below* the entry and the stop *above*, so with
    asymmetric `pt_sl` the barrier levels genuinely differ from the unsided case. Implemented
    by running the unsided labeler separately on longs and shorts with the multipliers
    swapped, which is exactly the same set of levels.
    """
    s = side.dropna()
    longs, shorts = s.index[s > 0], s.index[s < 0]
    parts = []
    if len(longs):
        parts.append(triple_barrier(bars, longs, trgt, pt_sl, n_bars, mode=mode, fine=fine,
                                    vertical_label="sign"))
    if len(shorts):
        parts.append(triple_barrier(bars, shorts, trgt, (pt_sl[1], pt_sl[0]), n_bars,
                                    mode=mode, fine=fine, vertical_label="sign"))
    ev = pd.concat(parts).sort_index()
    return meta_labels(ev, s)


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def classification_metrics(y: np.ndarray, take: np.ndarray) -> dict:
    """Precision / recall / F1 of the *filtered* strategy, in AFML's framing.

    positive = "take the bet"; y = "the bet would have made money". Recall is then the share
    of the profitable bets that were actually taken, and precision the hit rate of what was
    taken. The primary alone has recall 1 by construction, because it takes everything.
    """
    y = np.asarray(y).astype(bool)
    take = np.asarray(take).astype(bool)
    tp = float((y & take).sum())
    fp = float((~y & take).sum())
    fn = float((y & ~take).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else np.nan
    rec = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "n_bets": float(take.sum()), "frac_bets": float(take.mean()),
            "base_rate": float(y.mean())}


def f1_of_always_take(base_rate: float) -> float:
    """F1 of the do-nothing filter (take every bet): 2p / (1 + p).

    Kept as a function because it is the floor every reported F1 has to clear, and it is high:
    a base rate of 0.5 already gives 0.667 with zero skill.
    """
    return 2 * base_rate / (1 + base_rate) if base_rate > 0 else 0.0


def bet_metrics(ret_side: np.ndarray, take: np.ndarray, cost: float = 0.0,
                bars_per_year: int = 252, n_hold: int = 20) -> dict:
    """Economic side of the same filter: mean return per bet and an annualised Sharpe.

    Costs are charged per bet taken (round trip). The Sharpe annualisation assumes bets of
    `n_hold` bars, i.e. `bars_per_year / n_hold` non-overlapping bets a year — the honest
    scaling, since overlapping bets are not independent (notebook 03).
    """
    r = np.asarray(ret_side, float)[np.asarray(take).astype(bool)]
    if len(r) < 2:
        return {"mean_ret": np.nan, "sharpe": np.nan, "hit": np.nan, "total": 0.0, "n": len(r)}
    r = r - cost
    sd = r.std(ddof=1)
    per_year = bars_per_year / n_hold
    return {"mean_ret": float(r.mean()),
            "sharpe": float(r.mean() / sd * np.sqrt(per_year)) if sd > 0 else np.nan,
            "hit": float((r > 0).mean()), "total": float(r.sum()), "n": len(r)}


def daily_position(bar_index: pd.DatetimeIndex, events: pd.DataFrame,
                   take: np.ndarray, normalise: str = "mean",
                   size: np.ndarray | None = None) -> pd.Series:
    """Turn a set of accepted bets into a position held through calendar time.

    A bet taken at t0 with side s is held over (t0, t1]. Where bets overlap the position is
    their mean (`normalise="mean"`, constant gross exposure) or their sum (`"sum"`, exposure
    proportional to conviction). O(n) via difference arrays.

    `size` scales each bet in [0, 1] — the difference between using the secondary model as a
    *filter* (a hard 0/1 decision) and as a *sizer*, which is what AFML Ch. 10 actually asks
    for. The mean is taken over the bets that are alive, so sizing changes gross exposure
    while filtering changes only which windows are covered at all.
    """
    take = np.asarray(take).astype(bool)
    i0 = events["i0"].values.astype(int)[take]
    i1 = events["i1"].values.astype(int)[take]
    s = events["side"].values.astype(float)[take]
    if size is not None:
        s = s * np.asarray(size, float)[take]
    num = np.zeros(len(bar_index) + 1)
    cnt = np.zeros(len(bar_index) + 1)
    np.add.at(num, i0 + 1, s); np.add.at(num, i1 + 1, -s)
    np.add.at(cnt, i0 + 1, 1.0); np.add.at(cnt, i1 + 1, -1.0)
    num, cnt = np.cumsum(num)[:-1], np.cumsum(cnt)[:-1]
    pos = num if normalise == "sum" else np.divide(num, np.maximum(cnt, 1e-12),
                                                   out=np.zeros_like(num), where=cnt > 0)
    return pd.Series(pos, index=bar_index, name="position")


def portfolio_metrics(bar_index: pd.DatetimeIndex, close: pd.Series, events: pd.DataFrame,
                      take: np.ndarray, cost_bps: float = 0.0,
                      bars_per_year: int = 252, size: np.ndarray | None = None) -> dict:
    """Annualised Sharpe of the *strategy through time*, not per bet.

    This is the metric that stops selectivity from looking free. Sharpe per bet mechanically
    rewards taking fewer, better bets — it never charges for the days spent in cash. Here the
    position is held in calendar time, so sitting out shows up as return forgone.
    """
    pos = daily_position(bar_index, events, take, size=size)
    r = close.pct_change().fillna(0.0).values
    gross = pos.shift(1).fillna(0.0).values * r
    turn = np.abs(np.diff(np.concatenate(([0.0], pos.values))))
    net = gross - turn * cost_bps * 1e-4
    sd = net.std(ddof=1)
    return {"sharpe_time": float(net.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else np.nan,
            "ann_ret": float(net.mean() * bars_per_year),
            "exposure": float((pos.values != 0).mean()),
            "turnover": float(turn.sum()),
            "equity": float(np.exp(np.log1p(net).sum()) - 1.0)}
