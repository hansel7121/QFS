"""Daily prices and volumes (Yahoo) for every stock that appears in data/fit.csv with at least $50m of fund holdings.

Resumable (Yahoo throttles): tickers already in data/prices.csv are skipped.
Output: data/prices.csv  date, ticker, close, adj_close, volume
"""

import time
from pathlib import Path
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"


def main():
    f = pd.read_csv(DATA / "fit.csv")
    tick = sorted(set(f[f.fund_value_usd >= 5e7].ticker.str.replace(".", "-")) | {"SPY", "IWM"})
    out = DATA / "prices.csv"
    have = set(pd.read_csv(out, usecols=["ticker"]).ticker) if out.exists() else set()
    todo = [t for t in tick if t not in have]
    print(len(tick), "tickers;", len(todo), "to download", flush=True)
    for i in range(0, len(todo), 100):
        raw = yf.download(todo[i:i + 100], start="2019-01-01", auto_adjust=False, progress=False, threads=False)
        df = raw[["Close", "Adj Close", "Volume"]].stack(future_stack=True).reset_index()
        df.columns = ["date", "ticker", "close", "adj_close", "volume"]
        df.dropna(subset=["close"]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"{min(i + 100, len(todo))}/{len(todo)}", flush=True)
        time.sleep(12)


if __name__ == "__main__":
    main()
