"""
Third TWSE pull: MI_QFIIS (外資及陸資投資持股統計), daily, whole cross-section.

This endpoint solves two problems the first pull left open.

1. 發行股數 -- shares issued, AS OF THAT DAY. The company-profile file is a
   current snapshot, so index weights built from it drift backwards through the
   sample (Taiwanese firms issue stock dividends heavily). This gives the real
   time-varying share count.

2. 全體外資及陸資持有股數 -- total foreign holdings in shares. This is a
   HOLDINGS level, not a flow. Gabaix-Koijen's estimating equation is written in
   holdings growth dq_it = (Q_it - Q_it-1)/Q_it-1, and market clearing is an
   identity on holdings. With this we can build one sector's demand properly
   instead of proxying everything with turnover.

Output: data/holdings.csv  (date, stock_id, shares_issued, foreign_held, foreign_ratio)
Checkpointed per day in data/_done_qfiis.txt, same as collect_twse.py.
"""

import argparse
import os

import pandas as pd

from twse_client import TWSEClient

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

START = "2020-03-02"
END = "2026-09-23"

QFIIS_URL = "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"

COLS = {
    "證券代號": "stock_id",
    "發行股數": "shares_issued",
    "全體外資及陸資持有股數": "foreign_held",
    "全體外資及陸資持股比率": "foreign_ratio",
}


def to_num(x):
    s = str(x).replace(",", "").strip()
    if s in ("", "--", "---", "X", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def is_common_stock(code):
    code = str(code).strip()
    return len(code) == 4 and code.isdigit() and not code.startswith("0")


def parse(payload, datestr):
    if not payload or payload.get("stat") != "OK":
        return []
    fields = payload.get("fields", [])
    idx = {zh: fields.index(zh) for zh in COLS if zh in fields}
    if "證券代號" not in idx:
        return []
    out = []
    for row in payload.get("data", []):
        code = str(row[idx["證券代號"]]).strip()
        if not is_common_stock(code):
            continue
        rec = {"date": datestr, "stock_id": code}
        for zh, en in COLS.items():
            if zh == "證券代號" or zh not in idx:
                continue
            rec[en] = to_num(row[idx[zh]])
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    out_csv = os.path.join(DATA, "holdings.csv")
    done_f = os.path.join(DATA, "_done_qfiis.txt")

    # Only ask for days we already know are trading days.
    done_t86 = os.path.join(DATA, "_done_t86.txt")
    if os.path.exists(done_t86):
        px = pd.read_csv(os.path.join(DATA, "prices.csv"), usecols=["date"])
        days = sorted(px.date.unique())
        days = [str(d) for d in days]
    else:
        days = [d.strftime("%Y%m%d") for d in pd.bdate_range(args.start, args.end)]

    done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
    todo = [d for d in days if d not in done]

    client = TWSEClient(min_interval=args.interval)
    print(f"{len(days)} trading days, {len(done)} done, {len(todo)} to go")

    for n, ds in enumerate(todo, 1):
        rows = parse(client.get_json(QFIIS_URL, {"date": ds,
                                                 "selectType": "ALLBUT0999",
                                                 "response": "json"}), ds)
        if rows:
            pd.DataFrame(rows).to_csv(out_csv, mode="a",
                                      header=not os.path.exists(out_csv), index=False)
        with open(done_f, "a") as f:
            f.write(ds + "\n")
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  {ds}  rows={len(rows)}  ({client.n_calls} calls)")

    print(f"\ndone. {client.n_calls} requests.")
    if os.path.exists(out_csv):
        print(f"  holdings.csv {sum(1 for _ in open(out_csv))-1:,} rows")


if __name__ == "__main__":
    main()
