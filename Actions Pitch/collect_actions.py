"""Stock splits and cash dividends (ex-dates) from Yahoo for every ticker in the Insider Pitch price set, 2005 onward.
Resumable: batches of 50 tickers, checkpointed; sleeps on rate limits.

Output: data/actions.csv  date, ticker, dividend, split
"""

import time
from pathlib import Path
import pandas as pd
import yfinance as yf
import yfinance.shared as yfs

DATA = Path(__file__).parent / "data"
INS = Path(__file__).parent.parent / "Insider Pitch" / "data"


def main():
    tick = sorted(pd.read_parquet(INS / "prices.parquet", columns=["ticker"]).ticker.unique())
    out = DATA / "actions.csv"
    tried_f = DATA / "_tried.txt"
    tried = set(tried_f.read_text().split()) if tried_f.exists() else set()
    todo = [t for t in tick if t not in tried]
    print(len(todo), "to do", flush=True)
    for i in range(0, len(todo), 50):
        b = todo[i:i + 50]
        for k in range(6):
            h = yf.download(b, start="2005-01-01", actions=True, auto_adjust=False, progress=False, threads=False, group_by="column")
            if not any("Rate" in str(v) for v in dict(yfs._ERRORS).values()) and not h.empty:
                break
            print("rate limited, sleeping", flush=True); time.sleep(60 * (k + 1))
        if not h.empty:
            a = h[["Dividends", "Stock Splits"]].stack(level=1, future_stack=True).reset_index()
            a.columns = ["date", "ticker", "dividend", "split"]
            a = a[(a.dividend.fillna(0) > 0) | (a.split.fillna(0) > 0)]
            a.to_csv(out, mode="a", header=not out.exists(), index=False)
        tried.update(b); tried_f.write_text("\n".join(sorted(tried)))
        print(len(tried), "tried", flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
