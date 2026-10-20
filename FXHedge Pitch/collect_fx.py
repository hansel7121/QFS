"""5-minute FX bars (London time) from Twelve Data, back to 2020: EUR/USD, GBP/USD.

Output: data/{EURUSD,GBPUSD}_5m.csv  datetime (Europe/London), open, high, low, close
"""

import sys, time
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
KEY = (Path(__file__).parent.parent / "Gotobi" / "twelvedata.key").read_text().strip()
FLOOR = pd.Timestamp("2020-01-01")


def pull(sym):
    f = DATA / f"{sym.replace('/', '')}_5m.csv"
    have = pd.read_csv(f, parse_dates=["datetime"]) if f.exists() else pd.DataFrame()
    end = have.datetime.min() - pd.Timedelta(minutes=1) if len(have) else None
    while True:
        p = dict(symbol=sym, interval="5min", outputsize=5000, timezone="Europe/London", apikey=KEY)
        if end is not None:
            p["end_date"] = end.strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(8)                                     # 8 credits/min
        r = requests.get("https://api.twelvedata.com/time_series", params=p, timeout=60).json()
        if r.get("status") == "error":
            print(sym, r.get("message")); return
        v = pd.DataFrame(r["values"]); v["datetime"] = pd.to_datetime(v.datetime)
        have = pd.concat([v, have]).drop_duplicates("datetime").sort_values("datetime"); have.to_csv(f, index=False)
        end = v.datetime.min() - pd.Timedelta(minutes=1)
        print(sym, "back to", end.date(), flush=True)
        if end < FLOOR or len(v) < 5000:
            return


if __name__ == "__main__":
    for s in sys.argv[1:] or ["EUR/USD", "GBP/USD"]:
        pull(s)
