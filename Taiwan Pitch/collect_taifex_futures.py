"""
TAIFEX daily futures data for every single-stock future, including delisted contracts.

data/futures_map.csv is a snapshot of TAIFEX's *current* underlying list, so it misses futures delisted before the
snapshot and says nothing about when a future started trading. This collector rebuilds both from TAIFEX's public
daily download (https://www.taifex.com.tw/cht/3/futDataDown, all contracts, one month per request, no login). The
file lists every listed contract month every day, traded or not, so a futures code's first row is its listing date
and its last row its delisting / last expiry.

Underlying stock:
  * codes in futures_map.csv: TAIFEX's own mapping
  * codes not in it (delisted): the TWSE stock whose close matches the futures settlement price (median absolute
    log gap < 1% over up to 60 dates spread across the code's life; matched stock futures sit at 0.0-0.5%, the next
    closest non-match at 4%)

Output:
  data/taifex_cache/YYYY-MM.csv       raw monthly downloads (Big5 decoded to UTF-8)
  data/futures_listing_pit.csv        futures_id, stock_id, first, last, mapped_by
  data/taifex_stock_futures.csv       day-session rows of single-month contracts for the mapped codes:
                                      date, futures_id, stock_id, contract_date, close, volume, settle, oi

Usage:
    python collect_taifex_futures.py
"""

import glob
import os
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE = os.path.join(DATA, "taifex_cache")
URL = "https://www.taifex.com.tw/cht/3/futDataDown"
START, END = "2017-12", "2026-09"
MATCH_TOL = 0.01


def download():
    os.makedirs(CACHE, exist_ok=True)
    for m in pd.period_range(START, END, freq="M"):
        p = os.path.join(CACHE, f"{m}.csv")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            continue
        a, b = m.start_time.strftime("%Y/%m/%d"), m.end_time.strftime("%Y/%m/%d")
        for _ in range(4):
            try:
                r = requests.post(URL, timeout=120, data=dict(down_type=1, commodity_id="all",
                                                              queryStartDate=a, queryEndDate=b))
                txt = r.content.decode("big5", errors="replace")
                if txt.startswith("交易日期"):
                    open(p, "w", encoding="utf-8").write(txt)
                    print(m, f"{len(txt):,} chars", flush=True)
                    break
                print(m, "unexpected response:", txt[:80].replace("\n", " "), flush=True)
            except Exception as e:
                print(m, e, flush=True)
            time.sleep(10)
        time.sleep(2)


def load():
    fs = []
    for p in sorted(glob.glob(os.path.join(CACHE, "*.csv"))):
        d = pd.read_csv(p, dtype=str, index_col=False).iloc[:, [0, 1, 2, 6, 9, 10, 11, 17]]
        d.columns = ["date", "futures_id", "contract_date", "close", "volume", "settle", "oi", "session"]
        fs.append(d)
    f = pd.concat(fs).apply(lambda s: s.str.strip())
    f = f[(f.session == "一般") & f.futures_id.str.fullmatch(r"[A-Z0-9]{2}F") & f.contract_date.str.fullmatch(r"\d{6}")]
    f["date"] = pd.to_datetime(f.date)
    for c in ["close", "volume", "settle", "oi"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    return f[f.date >= "2018-01-01"].drop(columns="session").drop_duplicates(["date", "futures_id", "contract_date"])


def price_match(f, fids):
    px = pd.read_csv(os.path.join(DATA, "prices.csv"), usecols=["date", "stock_id", "close"],
                     dtype={"stock_id": str}, parse_dates=["date"])
    lp = np.log(px[px.close > 0].pivot_table(index="date", columns="stock_id", values="close"))
    near = f[f.settle > 0].sort_values("contract_date").drop_duplicates(["futures_id", "date"])   # nearest month
    out = {}
    for fid in fids:
        s = np.log(near[near.futures_id == fid].set_index("date").settle).reindex(lp.index).dropna()
        if len(s) < 5:
            continue
        s = s.iloc[np.linspace(0, len(s) - 1, min(len(s), 60)).astype(int)]
        d = lp.loc[s.index].sub(s, axis=0).abs()
        med = d.median()[d.notna().mean() > 0.8]
        if len(med) and med.min() < MATCH_TOL:
            out[fid] = med.idxmin()
    return out


def main():
    download()
    f = load()
    fm = pd.read_csv(os.path.join(DATA, "futures_map.csv"), dtype=str)
    fmap = dict(zip(fm.prefix + "F", fm.stock_id))
    life = f.groupby("futures_id").date.agg(first="min", last="max")
    unmapped = [x for x in life.index if x not in fmap]
    matched = price_match(f, unmapped)
    lst = life.reset_index()
    lst["stock_id"] = lst.futures_id.map(fmap).fillna(lst.futures_id.map(matched))
    lst["mapped_by"] = np.where(lst.futures_id.isin(fmap), "taifex list", "price match")
    lst = lst.dropna(subset=["stock_id"]).sort_values(["stock_id", "first"])
    lst[["futures_id", "stock_id", "first", "last", "mapped_by"]].to_csv(os.path.join(DATA, "futures_listing_pit.csv"), index=False)
    print(f"stock / ETF futures 2018+: {len(lst)}, of which delisted and price-matched: {(lst.mapped_by == 'price match').sum()}")

    g = f[f.futures_id.isin(lst.futures_id)].merge(lst[["futures_id", "stock_id"]], on="futures_id")
    g = g[g.stock_id.str.fullmatch(r"[1-9]\d{3}")]
    g[["date", "futures_id", "stock_id", "contract_date", "close", "volume", "settle", "oi"]] \
        .sort_values(["futures_id", "contract_date", "date"]) \
        .to_csv(os.path.join(DATA, "taifex_stock_futures.csv"), index=False)
    print(f"taifex_stock_futures.csv: {len(g):,} rows, {g.futures_id.nunique()} codes, {g.stock_id.nunique()} stocks")


if __name__ == "__main__":
    main()
