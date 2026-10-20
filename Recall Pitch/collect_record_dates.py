"""Annual-meeting record dates from DEF 14A proxy statements (SEC EDGAR), for the stock universe in
FundFlow Pitch/data/prices.csv. Only the first ~150KB of each proxy is fetched (HTTP Range); the record date is the
"close of business on <date>" / "record date ... <date>" sentence near the meeting notice.

Output: data/record_dates.csv  ticker, cik, filed, record_date
"""

import re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "QFS research hanselliu206@gmail.com"}
MON = "January|February|March|April|May|June|July|August|September|October|November|December"
PAT = [re.compile(rf"record date[^.]{{0,160}}?((?:{MON})\s+\d{{1,2}},\s+20\d\d)", re.I),
       re.compile(rf"close of business on\s+(?:\w+,\s+)?((?:{MON})\s+\d{{1,2}},\s+20\d\d)", re.I)]


def get(url, rng=None):
    h = dict(UA)
    if rng: h["Range"] = rng
    for k in range(4):
        try:
            time.sleep(0.11)
            r = requests.get(url, headers=h, timeout=60)
            if r.status_code in (200, 206):
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    return None


def one(t, cik):
    sub = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if sub is None:
        return []
    rec = sub.json()["filings"]["recent"]
    out = []
    for form, filed, acc, doc in zip(rec["form"], rec["filingDate"], rec["accessionNumber"], rec["primaryDocument"]):
        if form != "DEF 14A" or filed < "2019-01-01":
            continue
        r = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}", "bytes=0-150000")
        if r is None:
            continue
        txt = re.sub(r"\s+", " ", re.sub(r"&#160;|&nbsp;", " ", re.sub(r"<[^>]+>", " ", r.text)))
        for p in PAT:
            m = p.search(txt)
            if m:
                d = pd.to_datetime(m.group(1), errors="coerce")
                if pd.notna(d) and pd.Timestamp(filed) - pd.Timedelta(days=120) <= d <= pd.Timestamp(filed) + pd.Timedelta(days=30):
                    out.append((t, cik, filed, d.date()))
                    break
    return out


def main():
    tick = pd.read_csv(DATA.parent.parent / "FundFlow Pitch" / "data" / "prices.csv", usecols=["ticker"]).ticker.unique()
    m = {v["ticker"].replace(".", "-"): v["cik_str"] for v in get("https://www.sec.gov/files/company_tickers.json").json().values()}
    todo = [(t, m[t]) for t in tick if t in m]
    print(len(todo), "tickers with CIK", flush=True)
    rows = []
    with ThreadPoolExecutor(1) as ex:
        for i, res in enumerate(ex.map(lambda a: one(*a), todo)):
            rows += res
            if i % 200 == 0:
                print(i, len(rows), flush=True)
    pd.DataFrame(rows, columns=["ticker", "cik", "filed", "record_date"]).to_csv(DATA / "record_dates.csv", index=False)
    print(len(rows), "proxy record dates")


if __name__ == "__main__":
    main()
