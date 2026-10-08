"""
MOPS ex-rights / ex-dividend announcements (t108sb27), with the time each was posted.

suspension.csv gives the forced-covering deadline D but not when it became public. For ex-dividend and
ex-rights events, D = ex-date - 4 trading days, and the ex-date is public from the moment the company posts it
on MOPS. This pulls every announcement for TWSE-listed (sii) and OTC (otc) companies, one request per market
per year, so a backtest can trade only events that were announced before the entry.

The query year is the ROC year the announcement falls in; 2017 is included so that announcements made in late
2017 for early-2018 ex-dates are not missed.

Output:
  data/mops_exdiv.csv  stock_id, name, div_year, record_date, ex_rights_date, ex_div_date, pay_date,
                       cash_increase_shares, announce_date, announce_time, market

Raw responses are cached under data/mops_exdiv_cache/, so the job can be killed and resumed.

Usage:
    python collect_mops_exdiv.py
"""

import io
import os
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "mops_exdiv.csv")
URL = "https://mopsov.twse.com.tw/mops/web/ajax_t108sb27"
YEARS = range(2017, 2027)
INTERVAL = 4.0   # MOPS blocks clients that query too fast ("查詢過於頻繁")

CACHE = os.path.join(HERE, "data", "mops_exdiv_cache")
# columns by header name: the table gains a column in 2019, so positions are not stable.
# Ex-dates sit under a two-level header (股票股利 / 除權交易日, 現金股利 / 除息交易日).
COLS = {"公司代號": "stock_id", "公司名稱": "name", "股利所屬年度": "div_year", "權利分派基準日": "record_date",
        "除權交易日": "ex_rights_date", "除息交易日": "ex_div_date", "現金股利發放日": "pay_date",
        "現金增資總股數(股)": "cash_increase_shares", "公告日期": "announce_date", "公告時間": "announce_time"}


REQUIRED = ["stock_id", "ex_rights_date", "ex_div_date", "announce_date", "announce_time"]   # the rest is optional


def roc(s):
    """'107/07/26' -> Timestamp('2018-07-26'); anything else -> NaT."""
    s = str(s).strip()
    try:
        y, m, d = s.split("/")
        return pd.Timestamp(int(y) + 1911, int(m), int(d))
    except ValueError:
        return pd.NaT


def parse(html):
    t = pd.read_html(io.StringIO(html))[-1]
    names = [c[-1] if isinstance(c, tuple) else c for c in t.columns]
    keep = {i: COLS[n] for i, n in enumerate(names) if n in COLS}
    missing = set(REQUIRED) - set(keep.values())
    if missing:
        raise ValueError(f"columns not found: {missing}")
    t = t.iloc[:, list(keep)]
    t.columns = list(keep.values())
    return t.reindex(columns=list(COLS.values()))


def fetch(market, year):
    path = os.path.join(CACHE, f"{market}_{year}.html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return parse(f.read()), True
    data = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "TYPEK": market, "year": year - 1911, "type": ""}
    for attempt in range(5):
        try:
            r = requests.post(URL, data=data, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.encoding = "utf-8"
            if "查詢過於頻繁" in r.text:
                raise RuntimeError("rate limited")
            if "查無" in r.text or "<table" not in r.text:
                return None, False
            with open(path, "w", encoding="utf-8") as f:
                f.write(r.text)
            return parse(r.text), False
        except Exception as e:
            print(f"  {market} {year}: {e}; retry {attempt + 1}")
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"failed {market} {year}")


def main():
    os.makedirs(CACHE, exist_ok=True)
    frames = []
    for year in YEARS:
        for market in ("sii", "otc"):
            t, cached = fetch(market, year)
            n = 0 if t is None else len(t)
            print(f"{market} {year}: {n} rows")
            if t is not None:
                t["market"] = market
                frames.append(t)
            if not cached:
                time.sleep(INTERVAL)
    df = pd.concat(frames, ignore_index=True)
    df = df[df.stock_id.astype(str).str.fullmatch(r"\w{4,6}")]     # drop repeated header rows
    df["stock_id"] = df.stock_id.astype(str)
    for c in ("record_date", "ex_rights_date", "ex_div_date", "pay_date", "announce_date"):
        df[c] = df[c].map(roc)
    df = df.drop_duplicates().sort_values(["stock_id", "announce_date", "announce_time"])
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df):,} rows to {OUT}")


if __name__ == "__main__":
    main()
