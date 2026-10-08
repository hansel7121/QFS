"""
Intraday acceleration test -- the test Cho et al. actually specify.

    "To test the magnet effect, however, we cannot use daily prices. We must
     examine intraday price changes to see how the price reacts as it gets
     closer to the limits."

THE TEST
--------
For every intraday bar, measure the distance from the bar's OPEN to the day's
up-limit, and the return realised OVER that bar. The magnet predicts the bar
return rises as the distance shrinks.

THE CONFOUND, AND WHY IT IS THE WHOLE PROBLEM
---------------------------------------------
A price sitting near the limit got there by rising. Plain momentum -- trend
continuation with no limit in the picture at all -- produces exactly the same
raw pattern. So the unconditional relationship between distance and bar return
is uninformative. Two controls handle it:

  1. Explicit momentum regressors: the previous bar's return, and the cumulative
     return so far that session. A magnet must survive these.
  2. A pseudo-limit placebo: repeat the whole exercise measuring distance to a
     level with no regulatory standing (e.g. +7% in the 10% regime). Momentum,
     fat tails and volatility clustering all operate there identically; only the
     rule does not. If the real bound shows no more pull than the placebo, the
     result is momentum, not magnetism.

Bars are aligned to the daily panel to get each day's exact tick-rounded limit
and previous close.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_bars(path, daily):
    """Join intraday bars to the daily panel's limit fields."""
    b = pd.read_csv(path, dtype={"stock_id": str})
    b["datetime"] = pd.to_datetime(b.datetime, utc=True, errors="coerce")
    b = b.dropna(subset=["datetime", "open", "high", "low", "close"])
    b["datetime"] = b.datetime.dt.tz_convert("Asia/Taipei")
    b["date"] = pd.to_datetime(b.datetime.dt.date)
    b = b[b.open > 0]

    d = daily[["date", "stock_id", "prev", "up_limit", "dn_limit",
               "close", "lim_r_up"]].rename(columns={"close": "day_close"})
    m = b.merge(d, on=["date", "stock_id"], how="inner").sort_values(
        ["stock_id", "datetime"])

    g = m.groupby(["stock_id", "date"])
    m["bar_n"] = g.cumcount()
    m["day_open"] = g.open.transform("first")

    # Distance from where the bar STARTS to the bound, as a fraction of prev close.
    m["dist_up"] = (m.up_limit - m.open) / m.prev
    m["dist_dn"] = (m.open - m.dn_limit) / m.prev

    # Return realised over the bar, and momentum built before it.
    m["r_bar"] = m.close / m.open - 1
    m["r_prev"] = g.r_bar.shift(1) if "r_bar" in m else np.nan
    m["r_prev"] = m.groupby(["stock_id", "date"]).r_bar.shift(1)
    m["r_sofar"] = m.open / m.prev - 1
    m["absr"] = m.r_bar.abs()
    return m.dropna(subset=["dist_up", "r_bar"])


def band_profile(bars, side="up", bands=(0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.0)):
    """Mean bar return and mean |bar return| by distance-to-bound band.

    |bar return| is the speed of the price; the signed return is its direction.
    A magnet implies both rise as the bound approaches.
    """
    col = "dist_up" if side == "up" else "dist_dn"
    sgn = 1 if side == "up" else -1
    rows = []
    edges = list(bands)
    for lo, hi in zip(edges[1:], edges[:-1]):
        sel = bars[(bars[col] >= lo) & (bars[col] < hi)]
        if len(sel) < 30:
            continue
        r = sel.r_bar * sgn
        rows.append({"band": f"[{lo:.3f},{hi:.3f})", "n": len(sel),
                     "mean_r_bp": r.mean() * 1e4,
                     "t": r.mean() / r.std() * np.sqrt(len(r)),
                     "mean_abs_bp": sel.absr.mean() * 1e4})
    return pd.DataFrame(rows)


def acceleration_regression(bars, side="up", pseudo=None, max_dist=0.05):
    """Bar return on proximity to the bound, controlling for momentum.

    proximity = max_dist - distance, so a POSITIVE coefficient means the price
    moves faster toward the bound as it gets closer -- the magnet prediction.

    `pseudo` replaces the real bound with a fake threshold (as a return level,
    e.g. 0.07) to run the placebo version on identical bars.
    """
    sgn = 1 if side == "up" else -1
    b = bars.copy()

    if pseudo is None:
        dist = b.dist_up if side == "up" else b.dist_dn
    else:
        # distance from the bar open to a fake threshold at +/- `pseudo`
        level = b.prev * (1 + sgn * pseudo)
        dist = sgn * (level - b.open) / b.prev

    b["dist"] = dist
    b = b[(b.dist >= 0) & (b.dist <= max_dist)]
    b = b.dropna(subset=["r_bar", "r_prev", "r_sofar"])
    if len(b) < 200:
        return None

    b["prox"] = max_dist - b.dist
    y = b.r_bar * sgn * 1e4                      # bp
    X = pd.DataFrame({
        "prox": b.prox * 100,                    # per 1pp closer
        "mom_prev": b.r_prev * sgn * 1e4,
        "mom_sofar": b.r_sofar * sgn * 1e4,
        "bar_n": b.bar_n,
    })
    fit = sm.OLS(y, sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": b.stock_id})
    return fit, len(b)
