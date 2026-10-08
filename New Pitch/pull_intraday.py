"""
Intraday bars for the magnet test's acceleration leg.

Cho et al. are explicit that daily prices cannot test the magnet effect -- the claim
is about how the price behaves *within* the session as the bound gets closer. Free
per-stock intraday for Taiwan is thin, so this is a supporting test on a short window
rather than the headline:

    Yahoo 5-minute   ~60 days of history   ~54 bars/session
    Yahoo 60-minute  ~730 days             ~5 bars/session

We pull both for the names that most often approach the limit, since a random sample
of Taiwanese stocks would contain almost no limit approaches in a 60-day window.

That selection is deliberate and is NOT a bias for this test: we condition on distance
to the bound within each stock, so we are comparing a stock against itself at different
distances, not comparing limit-hitters against non-hitters.

Output: data/intraday_5m.csv, data/intraday_60m.csv
        columns: datetime, stock_id, open, high, low, close, volume
"""

import argparse
import os
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def pick_universe(n=60, band=0.02):
    """Stocks that most often bring their high within `band` of the up limit."""
    import limits as L
    px = L.load_prices(os.path.join(DATA, "prices.csv"), min_trade_value=1e6)
    recent = px[px.date >= px.date.max() - pd.Timedelta(days=400)]
    near = recent[recent.d_up <= band]
    counts = near.stock_id.value_counts()
    # Require some liquidity so Yahoo actually has usable intraday bars.
    liq = recent.groupby("stock_id").trade_value.median()
    keep = [s for s in counts.index if liq.get(s, 0) >= 3e7][:n]
    return keep, counts


def fetch(stock_ids, interval, period, out_csv, pause=0.6):
    import yfinance as yf

    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv, usecols=["stock_id"], dtype=str).stock_id.unique())
        print(f"  resuming: {len(done)} stocks already cached")

    for i, sid in enumerate(stock_ids, 1):
        if sid in done:
            continue
        try:
            df = yf.download(f"{sid}.TW", interval=interval, period=period,
                             progress=False, auto_adjust=False, threads=False)
        except Exception as e:
            print(f"  {sid}: {repr(e)[:80]}")
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        tcol = "Datetime" if "Datetime" in df.columns else "Date"
        out = pd.DataFrame({
            "datetime": df[tcol], "stock_id": sid,
            "open": df["Open"], "high": df["High"], "low": df["Low"],
            "close": df["Close"], "volume": df["Volume"]})
        out.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        if i % 10 == 0:
            print(f"  {i}/{len(stock_ids)}  last={sid}  bars={len(out)}")
        time.sleep(pause)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    ids, counts = pick_universe(args.n)
    print(f"universe: {len(ids)} frequent limit-approachers")
    print("  top 12:", ids[:12])

    print("\n5-minute (~60 days):")
    fetch(ids, "5m", "60d", os.path.join(DATA, "intraday_5m.csv"))
    print("\n60-minute (~730 days):")
    fetch(ids, "60m", "730d", os.path.join(DATA, "intraday_60m.csv"))

    for f in ("intraday_5m.csv", "intraday_60m.csv"):
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            d = pd.read_csv(p, dtype={"stock_id": str})
            print(f"  {f}: {len(d):,} bars, {d.stock_id.nunique()} stocks")


if __name__ == "__main__":
    main()
