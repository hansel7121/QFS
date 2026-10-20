"""Daily stock and bond prices for the month-end rebalancing pitch.

Outputs:
  data/etfs.csv    date, ticker, close, adj_close   (SPY, IEF, TLT, AGG, ^GSPC; Yahoo)
  data/dgs10.csv   date, DGS10 (FRED 10-year CMT yield, for a pre-2002 bond return proxy)
"""

import io
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"


def main():
    raw = yf.download(["SPY", "IEF", "TLT", "AGG", "^GSPC"], start="1989-01-01", auto_adjust=False, progress=False)
    e = pd.concat([raw["Close"].stack().rename("close"), raw["Adj Close"].stack().rename("adj_close")], axis=1).reset_index()
    e.columns = ["date", "ticker", "close", "adj_close"]
    e.dropna().to_csv(DATA / "etfs.csv", index=False)
    print(e.groupby("ticker").date.agg(["min", "max", "size"]))
    t = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10", timeout=60).text
    d = pd.read_csv(io.StringIO(t)); d.columns = ["date", "DGS10"]
    d.to_csv(DATA / "dgs10.csv", index=False)


if __name__ == "__main__":
    main()
