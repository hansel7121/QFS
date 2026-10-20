"""US IPOs 2012-2026 (Nasdaq IPO calendar API) and daily prices of the IPO stocks + IWM (Yahoo).

SPACs (blank-check "Acquisition" vehicles priced at $10) are dropped: their lockups work differently.

Outputs:
  data/ipos.csv    ticker, name, exchange, price, shares, value_usd, priced
  data/prices.csv  date, ticker, adj_close, volume
Survivorship caveat: IPOs later acquired/delisted have no Yahoo history and drop out.
"""

import time
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def main():
    rows = []
    for per in pd.period_range("2012-01", "2026-09", freq="M"):
        for k in range(3):
            try:
                d = requests.get(f"https://api.nasdaq.com/api/ipo/calendar?date={per}", headers=H, timeout=30).json()["data"]
                rows += (d.get("priced") or {}).get("rows") or []
                break
            except Exception:
                time.sleep(5)
        time.sleep(0.5)
    i = pd.DataFrame(rows)
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    out = pd.DataFrame({"ticker": i.proposedTickerSymbol, "name": i.companyName, "exchange": i.proposedExchange,
                        "price": num(i.proposedSharePrice), "shares": num(i.sharesOffered),
                        "value_usd": num(i.dollarValueOfSharesOffered), "priced": pd.to_datetime(i.pricedDate, errors="coerce")})
    spac = out.name.str.contains("Acquisition|Capital Corp|SPAC", case=False) & out.price.between(9.9, 10.1)
    out = out[~spac].dropna(subset=["ticker", "priced"]).drop_duplicates("ticker", keep="last")
    out.to_csv(DATA / "ipos.csv", index=False)
    print(len(out), "IPOs (ex-SPAC)")
    tick = sorted(out.ticker.str.replace(".", "-").unique()) + ["IWM"]
    parts = []
    for j in range(0, len(tick), 80):
        raw = yf.download(tick[j:j + 80], start="2012-01-01", auto_adjust=False, progress=False, threads=False)
        df = raw[["Adj Close", "Volume"]].stack(future_stack=True).reset_index()
        df.columns = ["date", "ticker", "adj_close", "volume"]
        parts.append(df.dropna(subset=["adj_close"]))
        print(f"{min(j + 80, len(tick))}/{len(tick)}", flush=True)
        time.sleep(10)
    pd.concat(parts).to_csv(DATA / "prices.csv", index=False)


if __name__ == "__main__":
    main()
