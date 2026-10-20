"""Daily total-return prices: SPY, S&P 500 index (pre-1993), IWM, EFA, QQQ (Yahoo).

Output: data/etfs.csv  date, ticker, close, adj_close
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"

if __name__ == "__main__":
    raw = yf.download(["SPY", "^GSPC", "IWM", "EFA", "QQQ"], start="1989-01-01", auto_adjust=False, progress=False)
    e = pd.concat([raw["Close"].stack().rename("close"), raw["Adj Close"].stack().rename("adj_close")], axis=1).reset_index()
    e.columns = ["date", "ticker", "close", "adj_close"]
    e.dropna().to_csv(DATA / "etfs.csv", index=False)
