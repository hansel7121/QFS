"""
Taiwan GIV data build: the stock x day panel, straight from TWSE.

Two endpoints, each returning the whole cross-section for one trading day:

  T86       三大法人買賣超日報  -> per-stock institutional buy/sell, in shares,
                                  split into the five investor categories
  MI_INDEX  每日收盤行情(全部)  -> per-stock OHLC, volume and trade value

Looping over days rather than over stocks is what makes this feasible: ~1,700
weekdays x 2 endpoints instead of one request per stock per dataset. It also
gives us the entire market rather than a preselected universe, which matters
because the GIV needs the residual (retail) sector to make the sectors sum to
market clearing.

Holidays are detected by the endpoint returning a non-OK payload; those dates
are recorded as done so a restart does not retry them.

Output (data/):
  inst_flows.csv   date, stock_id, and net/gross flows by investor category
  prices.csv       date, stock_id, OHLC, volume, trade value
  _done_t86.txt    dates already pulled, one per line (restart checkpoint)
  _done_px.txt

Usage:
    python collect_twse.py                      # full sample, resumable
    python collect_twse.py --start 2024-01-01   # narrower window
"""

import argparse
import os

import pandas as pd

from twse_client import TWSEClient

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

START = "2020-03-02"     # matches the sample already pulled for the Gotobi pitch
END = "2026-09-23"

T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
MI_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"

# T86 column order is stable; map the Chinese headers we keep to English.
T86_COLS = {
    "證券代號": "stock_id",
    "外陸資買進股數(不含外資自營商)": "foreign_buy",
    "外陸資賣出股數(不含外資自營商)": "foreign_sell",
    "外資自營商買賣超股數": "foreign_dealer_net",
    "投信買進股數": "trust_buy",
    "投信賣出股數": "trust_sell",
    "自營商買賣超股數(自行買賣)": "dealer_self_net",
    "自營商買賣超股數(避險)": "dealer_hedge_net",
    "三大法人買賣超股數": "total_net",
}

PX_COLS = {
    "證券代號": "stock_id",
    "成交股數": "volume",
    "成交金額": "trade_value",
    "開盤價": "open",
    "最高價": "high",
    "最低價": "low",
    "收盤價": "close",
}


def to_num(x):
    """TWSE numbers arrive as comma-separated strings; '--' means no trade."""
    if x is None:
        return None
    s = str(x).replace(",", "").strip()
    if s in ("", "--", "---", "X", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def is_common_stock(code):
    """TAIEX constituents are 4-digit codes in 1101..9999. Drop warrants, TDRs and
    preferreds (letter suffixes) and ETFs (4 digits but leading zero, e.g. 0050)."""
    code = str(code).strip()
    return len(code) == 4 and code.isdigit() and not code.startswith("0")


def load_done(path):
    if os.path.exists(path):
        return set(open(path).read().split())
    return set()


def mark_done(path, datestr):
    with open(path, "a") as f:
        f.write(datestr + "\n")


def append_rows(path, rows):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def parse_t86(payload, datestr):
    if not payload or payload.get("stat") != "OK":
        return []
    fields = payload.get("fields", [])
    idx = {name: fields.index(name) for name in T86_COLS if name in fields}
    if "證券代號" not in idx:
        return []
    out = []
    for row in payload.get("data", []):
        code = str(row[idx["證券代號"]]).strip()
        if not is_common_stock(code):
            continue
        rec = {"date": datestr, "stock_id": code}
        for zh, en in T86_COLS.items():
            if zh == "證券代號" or zh not in idx:
                continue
            rec[en] = to_num(row[idx[zh]])
        out.append(rec)
    return out


def parse_mi_index(payload, datestr):
    if not payload or payload.get("stat") != "OK":
        return []
    # The daily close table is the one whose title contains 每日收盤行情.
    table = None
    for t in payload.get("tables", []):
        if "每日收盤行情" in (t.get("title") or ""):
            table = t
            break
    if table is None:
        return []
    fields = table.get("fields", [])
    idx = {name: fields.index(name) for name in PX_COLS if name in fields}
    if "證券代號" not in idx:
        return []
    out = []
    for row in table.get("data", []):
        code = str(row[idx["證券代號"]]).strip()
        if not is_common_stock(code):
            continue
        rec = {"date": datestr, "stock_id": code}
        for zh, en in PX_COLS.items():
            if zh == "證券代號" or zh not in idx:
                continue
            rec[en] = to_num(row[idx[zh]])
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--interval", type=float, default=2.0,
                    help="minimum seconds between TWSE requests")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    flows_csv = os.path.join(DATA, "inst_flows.csv")
    px_csv = os.path.join(DATA, "prices.csv")
    done_t86_f = os.path.join(DATA, "_done_t86.txt")
    done_px_f = os.path.join(DATA, "_done_px.txt")

    days = pd.bdate_range(args.start, args.end)
    done_t86 = load_done(done_t86_f)
    done_px = load_done(done_px_f)

    client = TWSEClient(min_interval=args.interval)
    print(f"{len(days)} weekdays in {args.start}..{args.end}")
    print(f"  T86   already done: {len(done_t86)}")
    print(f"  price already done: {len(done_px)}")

    holidays = 0
    for n, day in enumerate(days, 1):
        ds = day.strftime("%Y%m%d")

        if ds not in done_t86:
            rows = parse_t86(client.get_json(T86_URL, {"date": ds, "selectType": "ALL",
                                                       "response": "json"}), ds)
            append_rows(flows_csv, rows)
            mark_done(done_t86_f, ds)
            if not rows:
                holidays += 1

        if ds not in done_px:
            rows = parse_mi_index(client.get_json(MI_URL, {"date": ds, "type": "ALLBUT0999",
                                                           "response": "json"}), ds)
            append_rows(px_csv, rows)
            mark_done(done_px_f, ds)

        if n % 25 == 0:
            print(f"  {n}/{len(days)}  {day.date()}  "
                  f"({client.n_calls} calls, {holidays} non-trading days so far)")

    print(f"\ndone. {client.n_calls} requests.")
    for path in (flows_csv, px_csv):
        if os.path.exists(path):
            n = sum(1 for _ in open(path)) - 1
            print(f"  {os.path.basename(path):16s} {n:>12,} rows")


if __name__ == "__main__":
    main()
