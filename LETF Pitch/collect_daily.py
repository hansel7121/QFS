"""Daily prices for every ETF in data/universe.csv, their underlying stocks, and SPY.

The ETF close stands in for NAV when converting N-PORT net assets into shares outstanding (leveraged ETFs trade at
NAV to within a few bp). For the stocks, `close` is the official close (closing auction print), which is what
leveraged ETFs rebalance to and what a market-on-close order fills at.

Outputs:
  data/daily.csv  date, ticker, open, high, low, close (split-adjusted), adj_close (dividend-adjusted), volume
"""

import time
from pathlib import Path
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"
START = "2021-06-01"


def main():
    u = pd.read_csv(DATA / "universe.csv")
    tickers = sorted(set(u.etf) | set(u.under.str.replace(".", "-")) | {"SPY"})
    out = DATA / "daily.csv"
    parts = [pd.read_csv(out, parse_dates=["date"])] if out.exists() else []   # resume: Yahoo rate-limits hard
    have = set(parts[0].ticker.str.replace(".", "-")) if parts else set()
    todo = [t for t in tickers if t not in have]
    for i in range(0, len(todo), 40):
        raw = yf.download(todo[i:i + 40], start=START, auto_adjust=False, progress=False, threads=False)
        df = raw[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].stack(future_stack=True).reset_index()
        df.columns = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        df["ticker"] = df.ticker.str.replace("-", ".")
        parts.append(df.dropna(subset=["close"]))
        print(f"{min(i + 40, len(todo))}/{len(todo)}", flush=True)
        time.sleep(15)
    d = pd.concat(parts).drop_duplicates(["date", "ticker"])
    d.to_csv(DATA / "daily.csv", index=False)
    missing = sorted(set(t.replace("-", ".") for t in tickers) - set(d.ticker))
    print(f"{d.ticker.nunique()} tickers, {d.date.min().date()}..{d.date.max().date()}; no data: {len(missing)} {missing[:20]}")


if __name__ == "__main__":
    main()
