"""
Build the stock x day panel the GIV estimation runs on.

Loads whatever collect_twse.py has written so far (the loaders are safe to call
mid-pull), joins prices to institutional flows, attaches market caps and index
weights, and returns tidy wide frames indexed by date.

Central object is the demand-shock proxy. For stock i on day t:

    dq_it = (institutional net buy in shares) / (shares issued)

which is the institutional sector's holdings change in stock i as a fraction of
the stock's total shares -- the analogue of Gabaix-Koijen's change in equity
holdings, at the stock level rather than the investor-sector level.

Known data issues, deliberately left visible rather than silently patched:
  * TWSE closes are UNADJUSTED. Ex-dividend and stock-dividend days produce
    large spurious negative returns. flag_ex_days() marks the suspects; a proper
    fix needs TWSE's TWT49U adjustment factors.
  * shares issued comes from a current snapshot, so weights drift backwards
    through the sample. build_shares_history.py is the fix.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

FLOWS_CSV = os.path.join(DATA, "inst_flows.csv")
PRICES_CSV = os.path.join(DATA, "prices.csv")
UNIVERSE_CSV = os.path.join(DATA, "universe.csv")

# Investor categories as TWSE reports them. foreign_dealer is the proprietary
# book of foreign brokers and is tiny; dealer_hedge is mechanical delta-hedging
# against warrant issuance, which is arguably the cleanest exogenous demand
# shock in the file and is worth carrying separately rather than netting away.
FLOW_COLS = ["foreign_net", "foreign_dealer_net", "trust_net",
             "dealer_self_net", "dealer_hedge_net"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_flows():
    df = pd.read_csv(FLOWS_CSV, dtype={"stock_id": str})
    df["date"] = pd.to_datetime(df.date, format="%Y%m%d")
    # TWSE gives foreign and trust as gross buy/sell; net them here.
    df["foreign_net"] = df.foreign_buy - df.foreign_sell
    df["trust_net"] = df.trust_buy - df.trust_sell
    df = df.drop_duplicates(subset=["date", "stock_id"], keep="last")
    return df.sort_values(["date", "stock_id"]).reset_index(drop=True)


def load_prices():
    df = pd.read_csv(PRICES_CSV, dtype={"stock_id": str})
    df["date"] = pd.to_datetime(df.date, format="%Y%m%d")
    df = df.drop_duplicates(subset=["date", "stock_id"], keep="last")
    return df.sort_values(["date", "stock_id"]).reset_index(drop=True)


def load_universe():
    if not os.path.exists(UNIVERSE_CSV):
        raise SystemExit("run build_universe.py first")
    return pd.read_csv(UNIVERSE_CSV, dtype={"stock_id": str})


# --------------------------------------------------------------------------
# panel construction
# --------------------------------------------------------------------------

def build_panel(min_price=5.0, min_trade_value=1e6):
    """Long panel: one row per stock-day, with returns, flows and weights.

    Filters out penny stocks and days with almost no trading, where a single
    lot can produce an enormous 'return' and an enormous scaled flow.
    """
    px = load_prices()
    fl = load_flows()
    uni = load_universe()

    df = px.merge(fl, on=["date", "stock_id"], how="inner")
    df = df.merge(uni[["stock_id", "stock_name", "shares_issued", "industry"]],
                  on="stock_id", how="inner")

    df = df[(df.close >= min_price) & (df.trade_value >= min_trade_value)]
    df = df.sort_values(["stock_id", "date"])

    # Unadjusted log return. See flag_ex_days() before trusting the tails.
    df["ret"] = np.log(df.close / df.groupby("stock_id").close.shift(1))

    df["market_cap"] = df.close * df.shares_issued
    df["inst_net"] = df[FLOW_COLS].sum(axis=1)

    # The demand-shock proxy: holdings change scaled by shares outstanding.
    # In basis points of shares outstanding, so the units are readable.
    df["dq"] = df.inst_net / df.shares_issued * 1e4
    for c in FLOW_COLS:
        df["dq_" + c.replace("_net", "")] = df[c] / df.shares_issued * 1e4

    df["turnover"] = df.trade_value / df.market_cap

    return df.dropna(subset=["ret"]).reset_index(drop=True)


def flag_ex_days(df, ret_floor=-0.08, vol_ratio=1.5):
    """Mark likely ex-dividend / stock-dividend days.

    TWSE closes are unadjusted, so an ex-day shows up as a large negative return
    that is NOT accompanied by unusual volume. A real selloff usually comes with
    elevated turnover; a mechanical price adjustment does not. Crude, but it
    isolates the contaminated observations well enough to test whether results
    depend on them.
    """
    d = df.copy()
    med_turn = d.groupby("stock_id").turnover.transform("median")
    d["ex_suspect"] = (d.ret < ret_floor) & (d.turnover < vol_ratio * med_turn)
    return d


def to_wide(df, value, universe=None):
    """Pivot the long panel to a days x stocks matrix."""
    if universe is not None:
        df = df[df.stock_id.isin(universe)]
    return df.pivot(index="date", columns="stock_id", values=value)


def weight_matrix(df, universe=None):
    """Time-varying index weights S_it, renormalised to sum to 1 each day.

    Weights use each day's own market cap, so they track price moves within the
    sample even though shares issued is a fixed snapshot.
    """
    cap = to_wide(df, "market_cap", universe)
    return cap.div(cap.sum(axis=1), axis=0)


def index_return(df, universe=None):
    """Cap-weighted return of the universe -- our stand-in for the TAIEX move.

    Built from the same weights the GIV uses, so the left and right hand sides
    of the estimating equation are internally consistent. Compare against the
    published TAIEX as a sanity check.
    """
    ret = to_wide(df, "ret", universe)
    w = weight_matrix(df, universe)
    w = w.reindex_like(ret).where(ret.notna())
    w = w.div(w.sum(axis=1), axis=0)
    return (ret * w).sum(axis=1, min_count=1)


def herfindahl(w):
    """Row-wise Herfindahl of a weight matrix, and its reciprocal.

    GIV power needs var(u_S) = H * sigma_u^2 to be large, i.e. a few big names.
    1/H reads as the effective number of independent units.
    """
    H = (w ** 2).sum(axis=1)
    return pd.DataFrame({"H": H, "eff_n": 1.0 / H})
