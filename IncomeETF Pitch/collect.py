"""Income / covered-call ETFs: daily prices and dividend (ex-date, amount) history, plus their underlyings (Yahoo).

Outputs:
  data/prices.csv     date, ticker, close, adj_close
  data/dividends.csv  ticker, ex_date, amount
"""

import time
from pathlib import Path
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"
PAIRS = {"JEPI": "SPY", "JEPQ": "QQQ", "QYLD": "QQQ", "XYLD": "SPY", "RYLD": "IWM", "DIVO": "SPY", "SPYI": "SPY", "QQQI": "QQQ",
         "TSLY": "TSLA", "NVDY": "NVDA", "CONY": "COIN", "MSTY": "MSTR", "AMDY": "AMD", "APLY": "AAPL", "AMZY": "AMZN",
         "GOOY": "GOOGL", "FBY": "META", "NFLY": "NFLX", "MSFO": "MSFT", "PLTY": "PLTR", "ULTY": "SPY", "YMAX": "SPY"}

if __name__ == "__main__":
    tick = sorted(set(PAIRS) | set(PAIRS.values()))
    raw = yf.download(tick, start="2012-01-01", auto_adjust=False, progress=False, threads=False)
    e = pd.concat([raw["Close"].stack().rename("close"), raw["Adj Close"].stack().rename("adj_close")], axis=1).reset_index()
    e.columns = ["date", "ticker", "close", "adj_close"]
    e.to_csv(DATA / "prices.csv", index=False)
    rows = []
    for t in PAIRS:
        d = yf.Ticker(t).dividends
        rows += [(t, i.tz_localize(None).normalize(), v) for i, v in d.items()]
        time.sleep(1)
    pd.DataFrame(rows, columns=["ticker", "ex_date", "amount"]).to_csv(DATA / "dividends.csv", index=False)
    pd.Series(PAIRS).rename_axis("etf").rename("under").to_csv(DATA / "pairs.csv")
    print(len(rows), "dividends")
