"""Hourly BTC on Coinbase (BTC-USD and USDT-USD, public candles API) and Binance spot BTCUSDT (data.binance.vision).

Coinbase premium = Coinbase BTC-USD / (Binance BTC-USDT x USDT-USD) - 1: US-dollar spot demand vs offshore.

Outputs: data/coinbase_{BTC-USD,USDT-USD}.csv, data/binance_BTCUSDT_1h.csv  (time UTC, close, volume)
"""

import io, time, zipfile
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


def coinbase(product, start="2020-01-01"):
    rows, t = [], pd.Timestamp(start)
    end = pd.Timestamp.utcnow().tz_localize(None)
    while t < end:
        t1 = min(t + pd.Timedelta(hours=300), end)
        for k in range(5):
            try:
                r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                                 params={"granularity": 3600, "start": t.isoformat() + "Z", "end": t1.isoformat() + "Z"},
                                 headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                if r.status_code == 200:
                    rows += r.json(); break
            except requests.RequestException:
                pass
            time.sleep(2 * (k + 1))
        t = t1
        time.sleep(0.35)
    d = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    d["time"] = pd.to_datetime(d.time, unit="s")
    d.drop_duplicates("time").sort_values("time")[["time", "close", "volume"]].to_csv(DATA / f"coinbase_{product}.csv", index=False)
    print(product, len(d))


def binance():
    parts = []
    for m in pd.period_range("2020-01", pd.Timestamp.today().to_period("M") - 1, freq="M"):
        r = requests.get(f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{m}.zip", timeout=60)
        if r.status_code != 200: continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            k = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None)
        t = k[0].astype("int64"); t = t.where(t < 1e14, t // 1000)          # 2025+ files are in microseconds
        parts.append(pd.DataFrame({"time": pd.to_datetime(t, unit="ms"), "close": k[4], "volume": k[7]}))
    pd.concat(parts).to_csv(DATA / "binance_BTCUSDT_1h.csv", index=False)


if __name__ == "__main__":
    binance()
    coinbase("BTC-USD")
    coinbase("USDT-USD", "2021-05-01")
