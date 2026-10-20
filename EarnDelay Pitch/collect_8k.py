"""Earnings-release dates (8-K Item 2.02, "Results of Operations") from SEC EDGAR submissions JSON, for every
ticker in the Insider Pitch price set. Uses filing date and acceptance time (after 16:00 ET = after the close).

Output: data/earn_8k.csv  ticker, cik, filed, accepted, items
"""

import json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
INS = Path(__file__).parent.parent / "Insider Pitch" / "data"
UA = {"User-Agent": "QFS research hanselliu206@gmail.com"}


def get(url):
    for k in range(4):
        try:
            time.sleep(0.12)
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * (k + 1))
    return None


def rows(block, t, cik):
    out = []
    for form, filed, acc, items in zip(block["form"], block["filingDate"], block["acceptanceDateTime"], block["items"]):
        if form in ("8-K", "8-K/A") and "2.02" in str(items) and form == "8-K":
            out.append((t, cik, filed, acc, items))
    return out


def one(t, cik):
    sub = get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
    if sub is None:
        return []
    out = rows(sub["filings"]["recent"], t, cik)
    for f in sub["filings"].get("files", []):
        old = get(f"https://data.sec.gov/submissions/{f['name']}")
        if old:
            out += rows(old, t, cik)
    return out


def main():
    tick = set(pd.read_parquet(INS / "prices.parquet", columns=["ticker"]).ticker.unique())
    m = {v["ticker"].replace(".", "-"): v["cik_str"] for v in json.load(open(INS / "company_tickers.json")).values()}
    todo = [(t, m[t]) for t in sorted(tick) if t in m]
    print(len(todo), "tickers", flush=True)
    res = []
    with ThreadPoolExecutor(1) as ex:
        for i, r in enumerate(ex.map(lambda a: one(*a), todo)):
            res += r
            if i % 500 == 0:
                print(i, len(res), flush=True)
    pd.DataFrame(res, columns=["ticker", "cik", "filed", "accepted", "items"]).to_csv(DATA / "earn_8k.csv", index=False)


if __name__ == "__main__":
    main()
