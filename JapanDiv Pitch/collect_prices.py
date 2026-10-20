"""Nikkei 225 index, TOPIX ETF (1306.T), and a few large Japanese stocks, daily OHLC (Yahoo).

Output: data/prices.csv  date, ticker, open, close
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"

if __name__ == "__main__":
    raw = yf.download(["^N225", "1306.T", "1321.T", "EWJ"], start="1995-01-01", auto_adjust=False, progress=False)
    d = raw[["Open", "Close"]].stack(future_stack=True).reset_index()
    d.columns = ["date", "ticker", "open", "close"]
    d.dropna().to_csv(DATA / "prices.csv", index=False)
    print(d.groupby("ticker").date.agg(["min", "max", "size"]))
