"""
Taiwan price limits: exact limit prices, distance-to-limit, and limit-hit flags.

WHY THIS MATTERS
----------------
The daily limit is NOT exactly +/-10%. TWSE computes the limit from the previous
close and then rounds it to the tick, and the tick size steps up with the price
level. So a stock's actual limit return varies roughly 9.5-10.0% depending on
where its previous close sat inside a tick band.

That blur lands precisely on the region the magnet test depends on -- the last
percent below the bound. Using a flat 9.5% threshold mixes genuine limit hits
with near-misses and would manufacture (or mask) a magnet effect. Everything
here works from the exact tick-rounded limit instead.

TICK TABLE (TWSE common stocks)
    price < 10        0.01
    10  <= p < 50     0.05
    50  <= p < 100    0.10
    100 <= p < 500    0.50
    500 <= p < 1000   1.00
    p >= 1000         5.00

The tick applicable to the limit price is determined by the LIMIT price's own
band, not the previous close's -- a stock closing at 99 has an up-limit near
108.9, which sits in the 100-500 band and so rounds to 0.5. We resolve this by
computing the raw limit, taking the tick from the raw limit's band, and rounding
toward the previous close (down for the up-limit, up for the down-limit) so the
limit is never wider than the regulation allows.

REGIME NOTE
-----------
The limit widened from 7% to 10% on 2015-06-01. Anything before that date needs
LIMIT_PCT = 0.07.
"""

import numpy as np
import pandas as pd

LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-01")
LIMIT_PCT_OLD = 0.07
LIMIT_PCT_NEW = 0.10

_TICK_EDGES = np.array([0, 10, 50, 100, 500, 1000, np.inf])
_TICK_SIZES = np.array([0.01, 0.05, 0.10, 0.50, 1.00, 5.00])


def tick_size(price):
    """Vectorised TWSE tick size for a price level."""
    p = np.asarray(price, dtype=float)
    idx = np.searchsorted(_TICK_EDGES, p, side="right") - 1
    idx = np.clip(idx, 0, len(_TICK_SIZES) - 1)
    return _TICK_SIZES[idx]


def limit_pct(dates):
    """0.07 before 2015-06-01, 0.10 from then on."""
    d = pd.to_datetime(pd.Series(dates))
    return np.where(d < LIMIT_CHANGE_DATE, LIMIT_PCT_OLD, LIMIT_PCT_NEW)


def limit_prices(prev_close, dates):
    """Exact tick-rounded (up_limit, down_limit) given the previous close.

    Rounding is toward the previous close on both sides, so the realised limit
    is never wider than the regulation permits.
    """
    prev = np.asarray(prev_close, dtype=float)
    pct = limit_pct(dates)

    raw_up = prev * (1 + pct)
    raw_dn = prev * (1 - pct)

    up = np.floor(raw_up / tick_size(raw_up)) * tick_size(raw_up)
    dn = np.ceil(raw_dn / tick_size(raw_dn)) * tick_size(raw_dn)
    return up, dn


def add_limit_fields(px, price_col="close", eps=1e-9):
    """Attach limit prices, distance-to-limit and hit flags to a price panel.

    Expects columns: date, stock_id, open, high, low, close (and prev close is
    computed here). Returns a copy with:

      up_limit, dn_limit      exact limit prices for the day
      r, r_high, r_low        simple returns vs previous close
      d_up, d_dn              distance from the day's HIGH to the up limit, and
                              from the day's LOW to the down limit, as a
                              fraction of the previous close. 0 means touched.
      touch_up, touch_dn      the day's high/low reached the limit
      close_up, close_dn      the day CLOSED at the limit
      locked_up, locked_dn    open = high = low = limit (locked all session)
    """
    d = px.copy()
    d = d.sort_values(["stock_id", "date"])
    d["prev"] = d.groupby("stock_id")[price_col].shift(1)
    d = d.dropna(subset=["prev", "open", "high", "low", "close"])
    d = d[d.prev > 0]

    up, dn = limit_prices(d.prev.values, d.date.values)
    d["up_limit"] = up
    d["dn_limit"] = dn

    d["r"] = d.close / d.prev - 1
    d["r_high"] = d.high / d.prev - 1
    d["r_low"] = d.low / d.prev - 1

    # Distance still to travel to reach the bound, in units of prev close.
    d["d_up"] = (d.up_limit - d.high) / d.prev
    d["d_dn"] = (d.low - d.dn_limit) / d.prev

    d["touch_up"] = d.high >= d.up_limit - eps
    d["touch_dn"] = d.low <= d.dn_limit + eps
    d["close_up"] = d.close >= d.up_limit - eps
    d["close_dn"] = d.close <= d.dn_limit + eps
    d["locked_up"] = d.touch_up & (d.low >= d.up_limit - eps)
    d["locked_dn"] = d.touch_dn & (d.high <= d.dn_limit + eps)

    # Realised limit return, which is what the limit actually binds at.
    d["lim_r_up"] = d.up_limit / d.prev - 1
    d["lim_r_dn"] = d.dn_limit / d.prev - 1
    return d


def load_prices(path="data/prices.csv", min_trade_value=1e6, min_price=1.0):
    """Load the TWSE daily panel and attach limit fields.

    The liquidity floor matters here: on a near-untraded day a single lot can
    print at the limit, which would pollute every statistic in the study.
    """
    px = pd.read_csv(path, dtype={"stock_id": str})
    px["date"] = pd.to_datetime(px.date, format="%Y%m%d")
    px = px.drop_duplicates(["date", "stock_id"])
    px = px[(px.trade_value >= min_trade_value) & (px.close >= min_price)]
    return add_limit_fields(px)
