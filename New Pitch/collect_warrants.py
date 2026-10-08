"""
Warrant issuance history for the dealer-hedging pitch.

THE IDEA
--------
Taiwanese warrant issuers must run delta-neutral books, so when a new call
warrant is listed the issuer has to BUY the underlying to hedge. That flow is
mechanical, sizeable, and its timing is known in advance -- which is exactly
what the Taiwan multiplier study lacked. Chan et al. (JBF 2014) document
+0.90% cumulative abnormal return on the underlying in the 20 days around
warrant issuance, and negative pressure at expiry when cash-settled hedges
are unwound.

You already hold the other half: T86's 自營商避險 (Dealer_Hedging) column is the
issuers' hedging flow, per stock, per day. This script supplies the event dates.

DATA DESIGN
-----------
TWSE's MI_INDEX serves every listed warrant's daily quote (type 0999 for calls,
0999P for puts) -- about 32,000 rows a day, so a naive full pull would be ~50m
rows. We do not need the panel. We need each warrant's FIRST appearance, which
is its listing date. So we walk the days forward keeping a set of codes already
seen and emit only codes we have not seen before. That collapses the job to one
row per warrant ever listed (~250k rows) while still being survivorship-free:
warrants that listed and expired inside the sample are captured, unlike the
current master-file snapshot which only holds warrants still alive.

The underlying is recovered from the warrant's short name, which is prefixed
with the underlying's short name (台積電凱基3B購01 -> 台積電). The mapping from
short name to stock id is learned from the live master file, which states the
underlying explicitly; 89% of warrants map to a listed common stock, and the
remainder are index and ETF warrants we exclude anyway.

BURN-IN WARNING
---------------
The FIRST day of the scan emits every warrant alive on that day (~32,000 of
them), because none has been seen before. Those are the standing stock, not new
listings. Drop the first sample date before running any event study -- from day
two onward the ~300-400 codes appearing each day are genuine new listings.
The output carries an `is_burnin` column so this cannot be forgotten.

Output (data/):
  warrant_listings.csv   first_date, warrant_id, warrant_name, kind, volume,
                         underlying_name, stock_id, is_burnin
  warrant_master.csv     current snapshot: underlying, issue size, strike, dates
  _done_warrants.txt     checkpoint
  _warrant_seen.txt      codes already emitted (needed to resume correctly)
"""

import argparse
import os

import pandas as pd
import requests

from twse_client import TWSEClient, HEADERS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

MI_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
MASTER_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap37_L"
PROFILE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"

KINDS = {"0999": "call", "0999P": "put"}


def to_num(x):
    s = str(x).replace(",", "").strip()
    if s in ("", "--", "---", "X", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_master():
    """Live warrant master: underlying, issue size, strike, exercise dates."""
    m = pd.DataFrame(requests.get(MASTER_URL, headers=HEADERS, timeout=120).json())
    m = m.rename(columns={
        "權證代號": "warrant_id", "權證簡稱": "warrant_name", "權證類型": "kind_zh",
        "標的證券/指數": "underlying_name", "履約開始日": "start_roc",
        "最後交易日": "last_trade_roc", "履約截止日": "expiry_roc",
        "發行單位數量(仟單位)": "units_k", "最新履約價格(元)/履約指數": "strike",
    })
    prof = pd.DataFrame(requests.get(PROFILE_URL, headers=HEADERS, timeout=120).json())
    prof = prof.rename(columns={"公司代號": "stock_id", "公司簡稱": "short"})
    name2id = dict(zip(prof.short, prof.stock_id))
    m["stock_id"] = m.underlying_name.map(name2id)
    m["units_k"] = pd.to_numeric(m.units_k, errors="coerce")
    m["strike"] = pd.to_numeric(m.strike, errors="coerce")
    return m, name2id


def build_prefix_map(master, name2id):
    """underlying short name -> stock_id, longest names first so that a name
    which is a prefix of another (e.g. 台積 vs 台積電) cannot shadow it."""
    names = sorted({n for n in master.underlying_name.dropna().unique()
                    if n in name2id}, key=len, reverse=True)
    return [(n, name2id[n]) for n in names]


def underlying_from_name(wname, prefix_map):
    for name, sid in prefix_map:
        if str(wname).startswith(name):
            return name, sid
    return None, None


def parse_day(payload, datestr, kind):
    if not payload:
        return []
    rows = []
    for t in payload.get("tables", []):
        if not t.get("data") or "每日收盤行情" not in (t.get("title") or ""):
            continue
        fields = t["fields"]
        try:
            i_code = fields.index("證券代號")
            i_name = fields.index("證券名稱")
            i_vol = fields.index("成交股數")
        except ValueError:
            continue
        for r in t["data"]:
            rows.append({"first_date": datestr,
                         "warrant_id": str(r[i_code]).strip(),
                         "warrant_name": str(r[i_name]).strip(),
                         "kind": kind,
                         "volume": to_num(r[i_vol])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    out_csv = os.path.join(DATA, "warrant_listings.csv")
    seen_f = os.path.join(DATA, "_warrant_seen.txt")
    done_f = os.path.join(DATA, "_done_warrants.txt")

    print("fetching warrant master + name map")
    master, name2id = fetch_master()
    prefix_map = build_prefix_map(master, name2id)
    master.to_csv(os.path.join(DATA, "warrant_master.csv"), index=False)
    print(f"  master {len(master):,} live warrants, {master.stock_id.nunique()} mapped underlyings")

    days = sorted(str(d) for d in
                  pd.read_csv(os.path.join(DATA, "prices.csv"), usecols=["date"]).date.unique())
    seen = set(open(seen_f).read().split()) if os.path.exists(seen_f) else set()
    done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
    todo = [d for d in days if d not in done]

    client = TWSEClient(min_interval=args.interval)
    print(f"{len(days)} trading days, {len(done)} done, {len(todo)} to go; "
          f"{len(seen):,} warrant codes already seen")

    first_day = days[0] if days else None

    for n, ds in enumerate(todo, 1):
        new_rows = []
        for type_code, kind in KINDS.items():
            payload = client.get_json(MI_URL, {"date": ds, "type": type_code,
                                               "response": "json"})
            for row in parse_day(payload, ds, kind):
                if row["warrant_id"] in seen:
                    continue
                seen.add(row["warrant_id"])
                uname, sid = underlying_from_name(row["warrant_name"], prefix_map)
                row["underlying_name"] = uname
                row["stock_id"] = sid
                # Day one emits the entire standing universe, not new listings.
                row["is_burnin"] = (ds == first_day)
                new_rows.append(row)

        if new_rows:
            pd.DataFrame(new_rows).to_csv(out_csv, mode="a",
                                          header=not os.path.exists(out_csv), index=False)
            with open(seen_f, "a") as f:
                f.write("\n".join(r["warrant_id"] for r in new_rows) + "\n")
        with open(done_f, "a") as f:
            f.write(ds + "\n")

        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  {ds}  new={len(new_rows):>4}  "
                  f"total seen={len(seen):,}  ({client.n_calls} calls)")

    print(f"\ndone. {client.n_calls} requests, {len(seen):,} warrants.")


if __name__ == "__main__":
    main()
