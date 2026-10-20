"""Daily Cboe index history: implied correlation (COR*), VIX family, dispersion index (DSPX).

Output data/cboe.csv: date, index, close.
COR1M / COR3M = implied correlation of the top 50 S&P 500 stocks, from SPX and single-stock option prices
(30 / 90 calendar days). In correlation points: 40 = average pairwise correlation 0.40.
"""

import io
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
INDICES = ["COR10D", "COR1M", "COR3M", "COR6M", "COR1Y", "VIX", "VIX3M", "VVIX", "DSPX"]
URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"


def main():
    out = []
    for ix in INDICES:
        r = requests.get(URL.format(ix), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().upper() for c in df.columns]
        col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
        df = pd.DataFrame({"date": pd.to_datetime(df["DATE"]), "index": ix, "close": pd.to_numeric(df[col], errors="coerce")})
        out.append(df.dropna())
        print(f"{ix:7s} {df.date.min().date()} .. {df.date.max().date()} rows={len(df)}")
    pd.concat(out).to_csv(DATA / "cboe.csv", index=False)


if __name__ == "__main__":
    main()
