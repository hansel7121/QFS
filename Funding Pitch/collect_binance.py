"""Binance USD-M perpetual futures: 5-minute klines and funding-rate history, from data.binance.vision (free bulk files).

Funding settles every 8 hours (00:00, 08:00, 16:00 UTC; some coins moved to 4h/1h later; BTC/ETH stayed at 8h).

Outputs:
  data/klines_{SYM}.csv   open_time (UTC), open, high, low, close, volume, quote_volume, taker_buy_quote
  data/funding_{SYM}.csv  calc_time (UTC), funding_interval_hours, last_funding_rate
Zips are cached in data/zips/.
"""

import io, sys, zipfile
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
ZIPS = DATA / "zips"; ZIPS.mkdir(parents=True, exist_ok=True)
BASE = "https://data.binance.vision/data/futures/um/monthly"
SYMS = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
MONTHS = pd.period_range("2020-01", pd.Timestamp.today().to_period("M") - 1, freq="M")


def fetch(url, f):
    if not f.exists():
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            return None
        f.write_bytes(r.content)
    with zipfile.ZipFile(f) as z:
        return pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None)


def main():
    for s in SYMS:
        kl, fu = [], []
        for m in MONTHS:
            k = fetch(f"{BASE}/klines/{s}/5m/{s}-5m-{m}.zip", ZIPS / f"{s}-5m-{m}.zip")
            if k is not None:
                if isinstance(k.iloc[0, 0], str):
                    k = k.iloc[1:]
                kl.append(k.iloc[:, [0, 1, 2, 3, 4, 5, 7, 10]].astype(float))
            f = fetch(f"{BASE}/fundingRate/{s}/{s}-fundingRate-{m}.zip", ZIPS / f"{s}-fundingRate-{m}.zip")
            if f is not None:
                if isinstance(f.iloc[0, 0], str):
                    f = f.iloc[1:]
                fu.append(f.astype(float))
        k = pd.concat(kl); k.columns = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]
        k["open_time"] = pd.to_datetime(k.open_time.astype("int64"), unit="ms")
        k.drop_duplicates("open_time").sort_values("open_time").to_csv(DATA / f"klines_{s}.csv", index=False)
        f = pd.concat(fu); f.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"]
        f["calc_time"] = pd.to_datetime(f.calc_time.astype("int64"), unit="ms")
        f.drop_duplicates("calc_time").sort_values("calc_time").to_csv(DATA / f"funding_{s}.csv", index=False)
        print(s, len(k), "bars", k.open_time.min(), "..", k.open_time.max(), "|", len(f), "funding rows", flush=True)


if __name__ == "__main__":
    main()
