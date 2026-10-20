"""Yahoo daily prices (open, close, adjusted close, volume) since 2005 for every ticker with an insider open-market
purchase, plus SPY and IWM. Yahoo only serves currently listed tickers, so the sample is survivor-biased; event
returns are measured over short windows against the same surviving universe.

Output: data/prices.parquet  date, ticker, open, close, adj_close, volume
"""

import time
from pathlib import Path
import pandas as pd
import yfinance as yf
import yfinance.shared as yfs

DATA = Path(__file__).parent / "data"


def main():
    tr = pd.read_csv(DATA / "insider_trades.csv", usecols=["ticker", "code"], dtype=str)
    tick = sorted({t for t in tr[tr.code == "P"].ticker.dropna() if t.replace("-", "").replace(".", "").isalpha() and len(t) <= 5})
    tick = [t.replace(".", "-") for t in tick] + ["SPY", "IWM"]
    print(len(tick), "tickers", flush=True)
    part_dir = DATA / "prices_parts"; part_dir.mkdir(exist_ok=True)
    tried_f = DATA / "_tried_tickers.txt"
    tried = set(tried_f.read_text().split()) if tried_f.exists() else set()
    todo = [t for t in tick if t not in tried]
    for i in range(0, len(todo), 50):
        b = todo[i:i + 50]
        for k in range(6):
            h = yf.download(b, start="2005-01-01", auto_adjust=False, progress=False, threads=False, group_by="column")
            errs = dict(yfs._ERRORS)
            if not any("Rate" in str(v) for v in errs.values()) and not (h.empty or h["Close"].notna().sum().sum() == 0 and len(errs) == len(b)):
                break
            print("rate limited, sleeping", flush=True); time.sleep(60 * (k + 1))
        if not h.empty:
            s = h[["Open", "Close", "Adj Close", "Volume"]].stack(level=1, future_stack=True).dropna(subset=["Close"]).reset_index()
            s.columns = ["date", "ticker", "open", "close", "adj_close", "volume"]
            s.to_parquet(part_dir / f"{b[0]}_{len(tried)}.parquet", index=False)
        tried.update(b); tried_f.write_text("\n".join(sorted(tried)))
        print(len(tried), "tried", flush=True)
        time.sleep(3)
    pd.concat(pd.read_parquet(f) for f in part_dir.glob("*.parquet")).drop_duplicates(["date", "ticker"]).to_parquet(DATA / "prices.parquet", index=False)


if __name__ == "__main__":
    main()
