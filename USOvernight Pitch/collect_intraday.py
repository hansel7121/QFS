"""5-minute regular-session bars for the underlying stocks, from Twelve Data (free tier: 8 calls/min, 800/day).

Pages backwards from today in 5,000-bar calls (~64 trading days each) to FLOOR (first single-stock leveraged ETFs
listed mid-2022). One CSV per stock in data/intraday/, checkpointed after every call, so a rerun resumes where the last
one stopped (oldest bar saved), including after the daily credit limit.

Bars are labelled by their start time, New York time: the 15:55 bar ends at 16:00. Its close is the last continuous
trade, not the closing-auction print; the official close comes from data/daily.csv.

Usage: python3 collect_intraday.py TSLA NVDA ...   (default: data/intraday_list.txt, one ticker per line)
"""

import sys, time
from collections import deque
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
OUT = DATA / "intraday"
OUT.mkdir(exist_ok=True)
KEY = (Path(__file__).parent.parent / "Gotobi" / "twelvedata.key").read_text().strip()
FLOOR = pd.Timestamp("2020-01-01")
PER_MIN = 8
calls = deque()


class DailyLimit(Exception):
    pass


def call(params):
    while True:
        now = time.time()
        while calls and now - calls[0] >= 60:
            calls.popleft()
        if len(calls) >= PER_MIN:
            time.sleep(60 - (now - calls[0]) + 0.5)
            continue
        calls.append(time.time())
        try:
            r = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=60).json()
        except Exception as e:
            print("   network:", e); time.sleep(5); continue
        if r.get("status") == "error":
            msg = r.get("message", "")
            if "run out of API credits for the day" in msg:
                raise DailyLimit(msg)
            if "API credits" in msg:          # per-minute limit hit anyway
                time.sleep(20); continue
            if "No data is available" in msg:
                return []
            print("   error:", msg); return []
        return r.get("values", [])


def pull(sym):
    f = OUT / f"{sym.replace('.', '-')}.csv"
    have = pd.read_csv(f, parse_dates=["datetime"]) if f.exists() else pd.DataFrame()
    if len(have) and have.datetime.min() <= FLOOR + pd.Timedelta(days=3):
        return "done"
    end = have.datetime.min() - pd.Timedelta(minutes=1) if len(have) else None
    while True:
        p = dict(symbol=sym, interval="5min", outputsize=5000, timezone="America/New_York", apikey=KEY)
        if end is not None:
            p["end_date"] = end.strftime("%Y-%m-%d %H:%M:%S")
        v = call(p)
        if not v:
            return "no more data"
        page = pd.DataFrame(v)
        page["datetime"] = pd.to_datetime(page.datetime)
        page = page[(page.datetime.dt.strftime("%H:%M") >= "09:30") & (page.datetime.dt.strftime("%H:%M") <= "15:55")]
        have = pd.concat([page, have]).drop_duplicates("datetime").sort_values("datetime")
        have.to_csv(f, index=False)
        end = page.datetime.min() - pd.Timedelta(minutes=1) if len(page) else pd.to_datetime(v[-1]["datetime"])
        print(f"   {sym}: back to {have.datetime.min().date()} ({len(have):,} bars)", flush=True)
        if have.datetime.min() <= FLOOR or len(v) < 5000:
            return "done"


def main():
    syms = sys.argv[1:] or (DATA / "intraday_list.txt").read_text().split()
    for s in syms:
        try:
            print(s, pull(s), flush=True)
        except DailyLimit as e:
            print("daily credit limit reached; rerun tomorrow to resume.", e)
            return


if __name__ == "__main__":
    main()
