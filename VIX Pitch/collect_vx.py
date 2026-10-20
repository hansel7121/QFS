"""Monthly VIX futures (VX) daily settlements from Cboe's free historical files, plus VIX ETP prices (Yahoo).

Expiry = the Wednesday 30 days before the third Friday of the following month (one business day earlier if that
Wednesday or the Friday is a holiday; the probe below tries the day before when the file is missing).

Outputs:
  data/vx.csv    date, expiry, settle, volume, oi
  data/etps.csv  date, ticker, close, adj_close  (VXX, UVXY, SVXY, ^VIX, SPY)
"""

import io
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
RAW = DATA / "vx"; RAW.mkdir(parents=True, exist_ok=True)
URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{}.csv"


def expiry(y, m):
    nm = pd.Timestamp(y, m, 1) + pd.offsets.MonthBegin(1)
    fridays = pd.date_range(nm, nm + pd.offsets.MonthEnd(0), freq="W-FRI")
    return fridays[2] - pd.Timedelta(days=30)


def main():
    parts = []
    for per in pd.period_range("2013-01", "2027-06", freq="M"):
        e = expiry(per.year, per.month)
        for cand in [e, e - pd.Timedelta(days=1)]:
            f = RAW / f"VX_{cand.date()}.csv"
            if not f.exists():
                r = requests.get(URL.format(cand.date()), timeout=60)
                if r.status_code != 200 or "Trade Date" not in r.text[:50]:
                    continue
                f.write_text(r.text)
            d = pd.read_csv(f)
            d = d.rename(columns={"Trade Date": "date", "Settle": "settle", "Total Volume": "volume", "Open Interest": "oi"})
            d["expiry"] = cand
            parts.append(d[["date", "expiry", "settle", "volume", "oi"]])
            break
    v = pd.concat(parts); v["date"] = pd.to_datetime(v.date)
    v = v[v.settle > 0]
    v.to_csv(DATA / "vx.csv", index=False)
    print(v.expiry.nunique(), "contracts", v.date.min().date(), "..", v.date.max().date())
    raw = yf.download(["VXX", "UVXY", "SVXY", "^VIX", "SPY", "VIXY"], start="2012-01-01", auto_adjust=False, progress=False)
    e = pd.concat([raw["Close"].stack().rename("close"), raw["Adj Close"].stack().rename("adj_close")], axis=1).reset_index()
    e.columns = ["date", "ticker", "close", "adj_close"]
    e.to_csv(DATA / "etps.csv", index=False)


if __name__ == "__main__":
    main()
