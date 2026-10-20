"""Buybacks and earnings dates for current S&P 500 members, from SEC EDGAR (free, 10 requests/s).

* Earnings announcement dates = filing dates of 8-Ks with Item 2.02 (Results of Operations), from the submissions API
  (recent + older filing pages).
* Annual buybacks = PaymentsForRepurchaseOfCommonStock from 10-K XBRL facts (companyfacts API), with the 10-K filing
  date (when it became public). Shares outstanding = dei:EntityCommonStockSharesOutstanding (cover page).

Outputs:
  data/earnings.csv  ticker, cik, date (8-K 2.02 filing date)
  data/buybacks.csv  ticker, fy_end, filed, repurchases (USD), shares
"""

import json, time
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "QFS research hanselliu206@gmail.com"}


def get(url):
    for k in range(4):
        time.sleep(0.12)
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 ** k)
    return None


def main():
    tickers = pd.read_csv(DATA.parent.parent / "Overnight Pitch" / "data" / "panel.csv", usecols=["ticker"]).ticker.unique()
    m = {v["ticker"].replace(".", "-"): v["cik_str"] for v in get("https://www.sec.gov/files/company_tickers.json").values()}
    earn, bb = [], []
    for i, t in enumerate(sorted(tickers)):
        if t == "SPY" or t not in m:
            continue
        cik = m[t]
        sub = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        if sub:
            pages = [sub["filings"]["recent"]] + [get("https://data.sec.gov/submissions/" + f["name"]) for f in sub["filings"].get("files", [])]
            for pg in pages:
                if not pg:
                    continue
                for form, d, items in zip(pg["form"], pg["filingDate"], pg.get("items", [""] * len(pg["form"]))):
                    if form == "8-K" and "2.02" in (items or ""):
                        earn.append((t, cik, d))
        cf = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        if cf:
            g = cf.get("facts", {}).get("us-gaap", {})
            rep = g.get("PaymentsForRepurchaseOfCommonStock", {}).get("units", {}).get("USD", [])
            sh = cf.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
            shd = pd.DataFrame(sh)
            for x in rep:
                if x.get("form") == "10-K" and x.get("fp") == "FY" and x.get("start") and \
                        (pd.Timestamp(x["end"]) - pd.Timestamp(x["start"])).days > 300:
                    s = shd[shd.accn == x["accn"]].val.max() if len(shd) else None
                    bb.append((t, x["end"], x["filed"], x["val"], s))
        if i % 50 == 0:
            print(i, t, len(earn), len(bb), flush=True)
    pd.DataFrame(earn, columns=["ticker", "cik", "date"]).drop_duplicates().to_csv(DATA / "earnings.csv", index=False)
    b = pd.DataFrame(bb, columns=["ticker", "fy_end", "filed", "repurchases", "shares"])
    b = b.sort_values("filed").drop_duplicates(["ticker", "fy_end"], keep="first")   # first report of each fiscal year
    b.to_csv(DATA / "buybacks.csv", index=False)
    print(len(earn), "earnings 8-Ks;", len(b), "firm-years of buybacks")


if __name__ == "__main__":
    main()
