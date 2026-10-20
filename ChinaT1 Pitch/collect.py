"""China A-share ETFs and indices, daily OHLC (Yahoo).

Output: data/china_etfs.csv  date, ticker, open, close, adj_close
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"
TICK = ["510300.SS", "510500.SS", "510050.SS", "159919.SZ", "159915.SZ", "512100.SS", "510330.SS", "159922.SZ", "000001.SS", "399001.SZ"]

if __name__ == "__main__":
    raw = yf.download(TICK, start="2004-01-01", auto_adjust=False, progress=False, threads=False)
    d = raw[["Open", "Close", "Adj Close"]].stack(future_stack=True).reset_index()
    d.columns = ["date", "ticker", "open", "close", "adj_close"]
    d = d[(d.open > 0) & (d.close > 0)]
    d.to_csv(DATA / "china_etfs.csv", index=False)
    print(d.groupby("ticker").date.agg(["min", "max", "size"]))
