"""
FinMind collector for the forced short-covering pitch.

  1. TaiwanStockMarginShortSaleSuspension  (one call per stock)
       -> the event calendar: every window where new short sales are banned
          ahead of a shareholder meeting / ex-dividend / ex-rights book closure.
          Shorts must be bought back before the window ends. This is the
          "Gotobi calendar" of this pitch.
  2. TaiwanFuturesDaily for every single-stock future  (one call per contract)
       -> the tax-cheap way to trade the effect (0.002% vs 0.3% on stock sells).

The stock universe is every 4-digit TWSE code that shows up in the margin
panel (only margin-eligible stocks can have a covering deadline). The TWSE
collector fills that panel concurrently, so this script re-reads it after each
pass and keeps going until the TWSE job has finished and nothing is left.

Output:
  data/suspension.csv     stock_id, date, end_date, reason
  data/stock_futures.csv  day-session rows with volume or open interest
  data/futures_map.csv    TAIFEX prefix -> underlying stock_id

Usage:
    python collect_finmind.py
"""

import csv
import os
import time

import pandas as pd

from finmind_client import FinMindClient

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

START, END = "2018-01-01", "2026-09-23"

MARGIN = os.path.join(DATA, "margin.csv")
TWSE_FINISHED = os.path.join(DATA, "_twse_finished.flag")
SUSP = os.path.join(DATA, "suspension.csv")
FUT = os.path.join(DATA, "stock_futures.csv")
FUTMAP = os.path.join(DATA, "futures_map.csv")
DONE_SUSP = os.path.join(DATA, "_done_susp.txt")
DONE_FUT = os.path.join(DATA, "_done_fut.txt")

SUSP_COLS = ["stock_id", "date", "end_date", "reason"]
FUT_COLS = ["date", "futures_id", "contract_date", "open", "max", "min", "close",
            "volume", "settlement_price", "open_interest", "trading_session"]


def get_patiently(cli, dataset, **kw):
    """Retry forever on rate limits so a throttled call is never recorded as 'no data'."""
    while True:
        try:
            return cli.get(dataset, **kw)
        except RuntimeError as e:
            if "level" in str(e).lower():
                raise
            print(f"    {e}; sleeping 15 min", flush=True)
            time.sleep(900)


def done_set(path):
    return set(open(path).read().split()) if os.path.exists(path) else set()


def appender(path, cols):
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(cols)
    return f, w


def margin_codes():
    if not os.path.exists(MARGIN):
        return set()
    ids = pd.read_csv(MARGIN, usecols=["stock_id"], dtype=str)["stock_id"].str.strip()
    return set(ids[ids.str.fullmatch(r"[1-9]\d{3}")])


def futures_map():
    if os.path.exists(FUTMAP):
        return pd.read_csv(FUTMAP, dtype=str)
    t = [x for x in pd.read_html("https://www.taifex.com.tw/cht/2/stockLists") if x.shape[1] > 10][0]
    t = t.iloc[:, [0, 2, 3, 4]]
    t.columns = ["prefix", "stock_id", "name", "has_future"]
    t = t.astype(str)
    t = t[t["has_future"].str.contains("是股票期貨")]
    t.to_csv(FUTMAP, index=False)
    return t


def pull_futures(cli):
    fmap = futures_map()
    done = done_set(DONE_FUT)
    f, w = appender(FUT, FUT_COLS)
    fd = open(DONE_FUT, "a")
    todo = [p for p in fmap["prefix"] if p not in done]
    print(f"futures: {len(todo)} contracts to pull", flush=True)
    for k, p in enumerate(todo):
        rows = get_patiently(cli, "TaiwanFuturesDaily", data_id=p + "F", start_date=START, end_date=END)
        for r in rows:
            if r.get("trading_session") != "position":
                continue
            if not r.get("volume") and not r.get("open_interest"):
                continue
            w.writerow([r.get(c, "") for c in FUT_COLS])
        f.flush()
        fd.write(p + "\n"); fd.flush()
        if k % 25 == 0:
            print(f"  fut {p}F: {len(rows)} rows [{k+1}/{len(todo)}]", flush=True)
    f.close()


def pull_suspension(cli, wait=True):
    f, w = appender(SUSP, SUSP_COLS)
    fd = open(DONE_SUSP, "a")
    while True:
        finished = os.path.exists(TWSE_FINISHED)
        todo = sorted(margin_codes() - done_set(DONE_SUSP))
        if not todo:
            if finished or not wait:
                break
            print("  suspension: waiting for more margin codes", flush=True)
            time.sleep(600)
            continue
        print(f"suspension: {len(todo)} stocks to pull", flush=True)
        for k, sid in enumerate(todo):
            for r in get_patiently(cli, "TaiwanStockMarginShortSaleSuspension", data_id=sid,
                             start_date=START, end_date=END):
                w.writerow([sid, r["date"], r["end_date"], r["reason"]])
            f.flush()
            fd.write(sid + "\n"); fd.flush()
            if k % 50 == 0:
                print(f"  susp {sid} [{k+1}/{len(todo)}]", flush=True)
    f.close()


def main():
    cli = FinMindClient()
    # Wait for the TWSE job to have some days in so the stock universe is populated.
    while len(margin_codes()) < 500:
        print("waiting for margin panel to seed the universe...", flush=True)
        time.sleep(60)
    pull_suspension(cli, wait=False)   # everything listed now
    pull_futures(cli)
    pull_suspension(cli, wait=True)    # top up codes only seen in older margin files
    print("finished", flush=True)


if __name__ == "__main__":
    main()
