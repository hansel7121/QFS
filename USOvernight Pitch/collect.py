"""Daily open/close for US index ETFs (Yahoo), 1993-2026.

Output: data/etfs.csv  date, ticker, open, close, adj_close
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"
TICK = ["SPY", "QQQ", "IWM", "DIA", "MDY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]

if __name__ == "__main__":
    raw = yf.download(TICK, start="1993-01-01", auto_adjust=False, progress=False, threads=False)
    d = raw[["Open", "Close", "Adj Close"]].stack(future_stack=True).reset_index()
    d.columns = ["date", "ticker", "open", "close", "adj_close"]
    d[(d.open > 0) & (d.close > 0)].to_csv(DATA / "etfs.csv", index=False)
    print(d.groupby("ticker").date.agg(["min", "max", "size"]))
