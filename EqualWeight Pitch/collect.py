"""Equal-weight SPDR industry ETFs (XBI, XME, XOP, XRT, KRE): current holdings (SSGA daily file) and daily
prices/volumes of every current holding and the ETFs themselves (Yahoo).

The S&P Select Industry indices reset to equal weight each quarter, effective after the close of the third Friday of
March / June / September / December, so the ETFs trade the reset at that close.
Survivorship caveat: only today's holdings are known; past members that left the index are missing.

Outputs:
  data/holdings.csv  etf, ticker, name, weight, shares_held
  data/prices.csv    date, ticker, close, adj_close, volume
"""

import time
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
ETFS = ["XBI", "XME", "XOP", "XRT", "KRE"]
URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{}.xlsx"


def main():
    rows = []
    for e in ETFS:
        f = DATA / f"holdings_{e.lower()}.xlsx"
        f.write_bytes(requests.get(URL.format(e.lower()), headers={"User-Agent": "Mozilla/5.0"}, timeout=60).content)
        h = pd.read_excel(f, skiprows=4).dropna(subset=["Ticker"])
        h = h[h.Ticker.str.match(r"^[A-Z.]{1,6}$") & (h.Ticker != "CASH")]
        rows.append(pd.DataFrame({"etf": e, "ticker": h.Ticker.str.replace(".", "-"), "name": h.Name, "weight": h.Weight, "shares_held": h["Shares Held"]}))
    H = pd.concat(rows)
    H.to_csv(DATA / "holdings.csv", index=False)
    print(H.groupby("etf").size().to_dict())
    tick = sorted(set(H.ticker)) + ETFS
    parts = []
    for i in range(0, len(tick), 60):
        raw = yf.download(tick[i:i + 60], start="2012-01-01", auto_adjust=False, progress=False, threads=False)
        df = raw[["Close", "Adj Close", "Volume"]].stack(future_stack=True).reset_index()
        df.columns = ["date", "ticker", "close", "adj_close", "volume"]
        parts.append(df.dropna(subset=["close"]))
        print(f"{min(i + 60, len(tick))}/{len(tick)}", flush=True)
        time.sleep(8)
    pd.concat(parts).to_csv(DATA / "prices.csv", index=False)


if __name__ == "__main__":
    main()
