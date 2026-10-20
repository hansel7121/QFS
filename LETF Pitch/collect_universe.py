"""Universe of US-listed single-stock leveraged / inverse ETFs, from SEC filings only (no scraping, no rate limits).

Source: SEC "Investment Company Series and Class Information" (every registered fund series with its name and ticker).
A series is kept if its name states a leverage multiple ("2X", "1.5x", "Bear 1X" ...) and the ticker it names is a
listed operating company in SEC company_tickers.json. That drops index (QQQ, SPY), crypto (BTC, ETH) and sector products.

Outputs:
  data/universe.csv  etf, series_id, cik, entity, name, lev (signed: +2 = 2x long, -1 = 1x inverse), under
"""

import re
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
UA = {"User-Agent": "QFS research hanselliu206@gmail.com"}
SERIES_URL = ("https://www.sec.gov/files/investment/data/other/investment-company-series-class-information/"
              "investment-company-series-class-2026.csv")
NOT_UNDER = {"ETF", "DAILY", "LONG", "SHORT", "BULL", "BEAR", "TARGET", "SHARES", "TRUST", "FUND", "ULTRA", "REX",
             "INVERSE", "ADR", "USD", "II",
             # exchange-traded funds that also appear in company_tickers.json
             "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "IBIT", "ETHA", "SMH", "SOXX", "XLK", "XLF", "ARKK"}


def parse(name):
    """(signed leverage, candidate underlying tickers) from a fund name, or (None, [])."""
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*x\b", name, re.I)
    if not m:
        return None, []
    lev = abs(float(m.group(1)))
    if re.search(r"\b(bear|short|inverse)\b", name, re.I) or m.group(1).startswith("-"):
        lev = -lev
    # the underlying sits next to the direction word: "Daily TSLA Bull", "2X Long TSLA", "1x Short LCID"
    # tickers are written in capitals, which keeps out "2X Long Pure Quantum"
    toks = re.findall(r"\b(?i:Daily)\s+([A-Z.]{1,6})\s+(?i:Bull|Bear)\b", name)
    toks += re.findall(r"\b(?i:Long|Short|Inverse)\s+([A-Z.]{1,6})\b", name)
    return lev, [t for t in toks if t not in NOT_UNDER]


def main():
    path = DATA / "sec_series_class_2026.csv"
    if not path.exists():
        path.write_bytes(requests.get(SERIES_URL, headers=UA, timeout=60).content)
    s = pd.read_csv(path, dtype=str, encoding="latin1")
    s.columns = [c.lstrip("﻿ï»¿") for c in s.columns]
    s = s.dropna(subset=["Class Ticker"]).drop_duplicates("Series ID")
    stocks = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=30).json()
    stocks = {v["ticker"].replace("-", ".") for v in stocks.values()}

    rows = []
    for r in s.itertuples(index=False):
        name = r[5]  # Series Name
        lev, toks = parse(name)
        if lev is None or lev == 1 or re.search(r"S&P|NASDAQ|Index|Strategy|MSCI|Treasury|Weekly|Monthly", name, re.I):
            # index products, and weekly/monthly-reset funds (they do not rebalance every close)
            continue
        under = next((t for t in toks if t in stocks), None)
        if under is None:
            continue
        rows.append({"etf": r[8], "series_id": r[4], "cik": int(r[1]), "entity": r[2], "name": name, "lev": lev,
                     "under": under})
    u = pd.DataFrame(rows).sort_values(["under", "lev", "etf"])
    u.to_csv(DATA / "universe.csv", index=False)
    print(f"{len(u)} single-stock leveraged/inverse ETF series on {u.under.nunique()} stocks, "
          f"{u.entity.nunique()} trusts")
    print(u.lev.value_counts().sort_index().to_string())
    print("most covered:", u.under.value_counts().head(15).to_dict())


if __name__ == "__main__":
    main()
