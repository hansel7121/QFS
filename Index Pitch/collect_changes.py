"""S&P 500 index changes (Wikipedia change log) and daily prices of the added / removed stocks (Yahoo).

The current Wikipedia article dropped its change table in 2026, so the table comes from the February 2026 revision.
Dates are effective dates (the change happens before the open; index funds trade at the previous close).
S&P announces changes after the close, typically 2-7 trading days before the effective date.

Outputs:
  data/changes.csv  date, added, removed, reason
  data/prices.csv   date, ticker, open, close, adj_close, volume   (+ SPY, MDY as benchmarks)
Survivorship caveat: removed stocks that were later acquired / delisted have no Yahoo history and drop out.
"""

import io, time
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
REVID = 1340186477      # 2026-02-24 revision of "List of S&P 500 companies"


def main():
    h = requests.get("https://en.wikipedia.org/w/index.php", params={"oldid": REVID},
                     headers={"User-Agent": "QFS research hanselliu206@gmail.com"}, timeout=60).text
    t = pd.read_html(io.StringIO(h))[1]
    t.columns = ["date", "added", "added_name", "removed", "removed_name", "reason"]
    t["date"] = pd.to_datetime(t.date, errors="coerce")
    ch = t.dropna(subset=["date"])[["date", "added", "removed", "reason"]]
    ch.to_csv(DATA / "changes.csv", index=False)
    print(len(ch), "changes", ch.date.min().date(), "..", ch.date.max().date())
    tick = sorted(set(ch.added.dropna()) | set(ch.removed.dropna()))
    tick = [x.replace(".", "-") for x in tick] + ["SPY", "MDY", "IJR"]
    parts = []
    for i in range(0, len(tick), 50):
        raw = yf.download(tick[i:i + 50], start="1999-01-01", auto_adjust=False, progress=False, threads=False)
        df = raw[["Open", "Close", "Adj Close", "Volume"]].stack(future_stack=True).reset_index()
        df.columns = ["date", "ticker", "open", "close", "adj_close", "volume"]
        parts.append(df.dropna(subset=["close"]))
        print(f"{min(i + 50, len(tick))}/{len(tick)}", flush=True)
        time.sleep(10)
    p = pd.concat(parts)
    p.to_csv(DATA / "prices.csv", index=False)
    print(p.ticker.nunique(), "tickers with prices")


if __name__ == "__main__":
    main()
