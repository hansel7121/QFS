"""Nasdaq Reg SHO threshold-security lists (daily, 2010-), and prices of names that stay on the list 13+ days.

Rule 203(b)(3): a fail to deliver in a threshold security that persists for 13 consecutive settlement days must be
closed out by purchasing shares, a forced buy whose timing follows from the list itself.

Outputs:
  data/threshold.csv  date, symbol            (one row per symbol per list day)
  data/prices.csv     date, ticker, adj_close, volume
"""

import io, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"; RAW.mkdir(parents=True, exist_ok=True)


def day(d):
    f = RAW / f"nasdaqth{d:%Y%m%d}.txt"
    if not f.exists():
        time.sleep(0.2)
        try:
            r = requests.get(f"https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{d:%Y%m%d}.txt", timeout=30)
        except requests.RequestException:
            return None
        if r.status_code != 200 or "Symbol|" not in r.text[:50]:
            return None
        f.write_text(r.text)
    t = pd.read_csv(io.StringIO(f.read_text()), sep="|")
    t = t[t.Symbol.notna() & ~t.Symbol.astype(str).str.startswith("File")]
    return pd.DataFrame({"date": d, "symbol": t.Symbol.astype(str)})


def main():
    days = pd.bdate_range("2010-01-01", pd.Timestamp.today() - pd.Timedelta(days=1))
    with ThreadPoolExecutor(1) as ex:
        parts = [p for p in ex.map(day, days) if p is not None]
    th = pd.concat(parts); th.to_csv(DATA / "threshold.csv", index=False)
    print(th.date.nunique(), "list days;", th.symbol.nunique(), "symbols")
    # names with a 13+ day consecutive run
    th = th.sort_values(["symbol", "date"]); dts = sorted(th.date.unique()); pos = {d: i for i, d in enumerate(dts)}
    th["k"] = th.date.map(pos)
    th["run"] = th.groupby("symbol").k.diff().ne(1).cumsum()
    runs = th.groupby(["symbol", "run"]).size()
    names = sorted(runs[runs >= 13].reset_index().symbol.unique())
    print(len(names), "symbols with a 13+ day run")
    out = []
    for i in range(0, len(names), 80):
        raw = yf.download([n.replace(".", "-") for n in names[i:i + 80]] + ["IWM"], start="2009-10-01", auto_adjust=False, progress=False, threads=False)
        df = raw[["Adj Close", "Volume"]].stack(future_stack=True).reset_index(); df.columns = ["date", "ticker", "adj_close", "volume"]
        out.append(df.dropna(subset=["adj_close"]))
        time.sleep(8)
    pd.concat(out).drop_duplicates(["date", "ticker"]).to_csv(DATA / "prices.csv", index=False)


if __name__ == "__main__":
    main()
