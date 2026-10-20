"""Daily circulating supply of all USD stablecoins and of USDT / USDC (DefiLlama stablecoins API, free).

Output: data/stables.csv  date, total_usd, usdt_usd, usdc_usd
"""

from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"


def series(url):
    j = requests.get(url, timeout=60).json()
    return pd.Series({pd.to_datetime(int(r["date"]), unit="s"): r["totalCirculatingUSD"].get("peggedUSD", 0) for r in j})


if __name__ == "__main__":
    tot = series("https://stablecoins.llama.fi/stablecoincharts/all")
    usdt = series("https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1")
    usdc = series("https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=2")
    d = pd.DataFrame({"total_usd": tot, "usdt_usd": usdt, "usdc_usd": usdc}).rename_axis("date")
    d.to_csv(DATA / "stables.csv")
    print(d.dropna().iloc[[0, -1]])
