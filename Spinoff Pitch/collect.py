"""Spin-offs: companies that filed a Form 10-12B registration (SEC full-text search, 2005-2025), mapped to their ticker
(SEC company_tickers.json, today's tickers only, so delisted spincos are missing: survivorship), with Yahoo prices.
A registrant counts as a spinco if its Yahoo history starts within 18 months after the first 10-12B filing.

Outputs:
  data/filers.csv   cik, name, first_filed, ticker
  data/prices.csv   date, ticker, adj_close, volume  (+ IWM)
"""

import time
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "QFS research hanselliu206@gmail.com"}


def main():
    rows = []
    for y in range(2005, 2026):
        for half in [("01-01", "06-30"), ("07-01", "12-31")]:
            frm = 0
            while True:
                r = requests.get("https://efts.sec.gov/LATEST/search-index", headers=UA, timeout=60,
                                 params={"forms": "10-12B", "startdt": f"{y}-{half[0]}", "enddt": f"{y}-{half[1]}", "from": frm}).json()
                hits = r["hits"]["hits"]
                for h in hits:
                    s = h["_source"]
                    if s.get("form") != "10-12B":
                        continue
                    rows.append((int(s["ciks"][0]), s["display_names"][0], s["file_date"]))
                frm += len(hits)
                if not hits or frm >= r["hits"]["total"]["value"] or frm >= 9900:
                    break
                time.sleep(0.15)
    f = pd.DataFrame(rows, columns=["cik", "name", "filed"]).groupby("cik").agg(name=("name", "first"), first_filed=("filed", "min")).reset_index()
    tick = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=60).json()
    m = {v["cik_str"]: v["ticker"] for v in tick.values()}
    f["ticker"] = f.cik.map(m)
    f = f.dropna(subset=["ticker"])
    f.to_csv(DATA / "filers.csv", index=False)
    print(len(f), "10-12B filers with a current ticker")
    tk = sorted(f.ticker.str.replace(".", "-").unique()) + ["IWM"]
    parts = []
    for i in range(0, len(tk), 80):
        raw = yf.download(tk[i:i + 80], start="2005-01-01", auto_adjust=False, progress=False, threads=False)
        df = raw[["Adj Close", "Volume"]].stack(future_stack=True).reset_index(); df.columns = ["date", "ticker", "adj_close", "volume"]
        parts.append(df.dropna(subset=["adj_close"]))
        time.sleep(8)
    pd.concat(parts).to_csv(DATA / "prices.csv", index=False)


if __name__ == "__main__":
    main()
