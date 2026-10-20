"""Daily open/close for current S&P 500 members (Wikipedia list) + SPY, from Yahoo.

Survivorship caveat: today's members only; names that left the index are missing.

Output: data/panel.csv  date, ticker, open, close, adj_close, volume   (open/close split-adjusted, not dividend-adjusted)
"""

import io, time
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"


def main():
    h = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "Mozilla/5.0"}, timeout=60).text
    tick = pd.read_html(io.StringIO(h))[0].Symbol.str.replace(".", "-").tolist() + ["SPY"]
    out = DATA / "panel.csv"
    parts = [pd.read_csv(out, parse_dates=["date"])] if out.exists() else []
    have = set(parts[0].ticker) if parts else set()
    todo = [t for t in tick if t not in have]
    for i in range(0, len(todo), 50):
        raw = yf.download(todo[i:i + 50], start="2004-01-01", auto_adjust=False, progress=False, threads=False)
        df = raw[["Open", "Close", "Adj Close", "Volume"]].stack(future_stack=True).reset_index()
        df.columns = ["date", "ticker", "open", "close", "adj_close", "volume"]
        parts.append(df.dropna(subset=["close", "open"]))
        print(f"{min(i + 50, len(todo))}/{len(todo)}", flush=True)
        time.sleep(10)
    p = pd.concat(parts).drop_duplicates(["date", "ticker"])
    p.to_csv(out, index=False)
    print(p.ticker.nunique(), "tickers", p.date.min(), "..", p.date.max())


if __name__ == "__main__":
    main()
