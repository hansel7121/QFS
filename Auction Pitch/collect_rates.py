"""Daily Treasury yields (FRED constant-maturity) and Treasury ETF prices (Yahoo).

Yields are the clean measure of the auction effect (no roll, no fund flows). ETFs are what you would trade at
student scale (futures are the institutional version: ZT / ZF / ZN / TN / ZB track the same points of the curve).

Outputs:
  data/yields.csv  date, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30 (percent)
  data/etfs.csv    date, ticker, close, adj_close
"""

import io
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
SERIES = ["DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
ETFS = ["SHY", "IEI", "IEF", "TLH", "TLT", "SPY"]


def main():
    ys = []
    for s in SERIES:
        t = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", timeout=60).text
        df = pd.read_csv(io.StringIO(t))
        df.columns = ["date", s]
        df["date"] = pd.to_datetime(df.date)
        df[s] = pd.to_numeric(df[s], errors="coerce")
        ys.append(df.set_index("date"))
    y = pd.concat(ys, axis=1).loc["1989-01-01":]
    y.to_csv(DATA / "yields.csv")
    print("yields:", y.index.min().date(), "..", y.index.max().date(), y.notna().sum().to_dict())
    raw = yf.download(ETFS, start="2002-01-01", auto_adjust=False, progress=False)
    e = pd.concat([raw["Close"].stack().rename("close"), raw["Adj Close"].stack().rename("adj_close")], axis=1).reset_index()
    e.columns = ["date", "ticker", "close", "adj_close"]
    e.dropna().to_csv(DATA / "etfs.csv", index=False)
    print("etfs:", e.groupby("ticker").date.min().dt.date.to_dict())


if __name__ == "__main__":
    main()
