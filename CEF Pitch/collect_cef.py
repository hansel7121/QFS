"""US closed-end funds: universe (CEFConnect) and daily share price + NAV history (Yahoo; NAVs trade as X...X tickers).

Survivorship caveat: the universe is today's listed funds; funds merged or liquidated before today are missing.

Outputs:
  data/universe.csv  ticker, name, category, nav_ticker
  data/prices.csv    date, ticker, close, adj_close, nav
"""

import time
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


def main():
    u = pd.DataFrame(requests.get("https://www.cefconnect.com/api/v3/DailyPricing?props=Ticker,Name,CategoryName,NAVTicker",
                                  headers={"User-Agent": "Mozilla/5.0"}, timeout=60).json())
    u.columns = ["ticker", "name", "category", "nav_ticker"]
    u = u.dropna(subset=["nav_ticker"])
    u.to_csv(DATA / "universe.csv", index=False)
    print(len(u), "funds;", u.category.str.split("-").str[0].value_counts().to_dict())
    out = DATA / "prices.csv"
    have = set(pd.read_csv(out).ticker) if out.exists() else set()
    parts = [pd.read_csv(out, parse_dates=["date"])] if out.exists() else []
    todo = u[~u.ticker.isin(have)]
    for i in range(0, len(todo), 40):
        b = todo.iloc[i:i + 40]
        raw = yf.download(list(b.ticker) + list(b.nav_ticker), start="2004-01-01", auto_adjust=False, progress=False, threads=False)
        cl, ad = raw["Close"], raw["Adj Close"]
        for t, n in zip(b.ticker, b.nav_ticker):
            if t not in cl or n not in cl:
                continue
            df = pd.DataFrame({"close": cl[t], "adj_close": ad[t], "nav": cl[n]}).dropna()
            if len(df):
                parts.append(df.assign(ticker=t).reset_index().rename(columns={"Date": "date"}))
        print(f"{min(i + 40, len(todo))}/{len(todo)}", flush=True)
        time.sleep(10)
    p = pd.concat(parts).drop_duplicates(["date", "ticker"])
    p.to_csv(out, index=False)
    print(p.ticker.nunique(), "funds with price + NAV;", p.date.min(), "..", p.date.max())


if __name__ == "__main__":
    main()
