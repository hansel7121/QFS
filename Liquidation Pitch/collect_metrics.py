"""Binance USD-M futures 5-minute metrics (open interest, long/short ratios) from data.binance.vision daily files.

A simultaneous large price move and large drop in open interest marks forced liquidations (Binance's own
liquidation snapshot files are no longer published).

Output: data/metrics_{SYM}.csv  create_time (UTC), sum_open_interest, sum_open_interest_value
Daily zips cached in data/zips/.
"""

import io, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
ZIPS = DATA / "zips"; ZIPS.mkdir(parents=True, exist_ok=True)
BASE = "https://data.binance.vision/data/futures/um/daily/metrics"


def one(sym, day):
    f = ZIPS / f"{sym}-metrics-{day}.zip"
    if not f.exists():
        r = requests.get(f"{BASE}/{sym}/{sym}-metrics-{day}.zip", timeout=60)
        if r.status_code != 200:
            return None
        f.write_bytes(r.content)
    with zipfile.ZipFile(f) as z:
        return pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), usecols=["create_time", "sum_open_interest", "sum_open_interest_value"])


def main():
    days = [d.date() for d in pd.date_range("2021-12-01", pd.Timestamp.today() - pd.Timedelta(days=2))]
    for s in sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]:
        with ThreadPoolExecutor(8) as ex:
            parts = [p for p in ex.map(lambda d: one(s, d), days) if p is not None]
        m = pd.concat(parts); m["create_time"] = pd.to_datetime(m.create_time)
        m.sort_values("create_time").drop_duplicates("create_time").to_csv(DATA / f"metrics_{s}.csv", index=False)
        print(s, len(m), m.create_time.min(), m.create_time.max(), flush=True)


if __name__ == "__main__":
    main()
