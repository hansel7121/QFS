"""Yahoo daily OHLC, adjusted close and volume for every ETF in data/themes.csv plus SPY, QQQ, IWM (benchmarks),
2015 onward; plus Yahoo's shares-outstanding history where available (a proxy for ETF creations / redemptions).

Output: data/prices.csv  date, ticker, open, close, adj_close, volume
        data/shares.csv  date, ticker, shares
"""

import time
from pathlib import Path
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"


def main():
    themes = pd.read_csv(DATA / "themes.csv")
    tick = sorted({e for s in themes.etfs for e in s.split(";")} | {"SPY", "QQQ", "IWM"})
    h = yf.download(tick, start="2015-01-01", auto_adjust=False, progress=False, threads=False, group_by="column")
    p = h[["Open", "Close", "Adj Close", "Volume"]].stack(level=1, future_stack=True).dropna(subset=["Close"]).reset_index()
    p.columns = ["date", "ticker", "open", "close", "adj_close", "volume"]
    p.to_csv(DATA / "prices.csv", index=False)
    print(p.groupby("ticker").date.min().dt.date.to_dict(), flush=True)
    rows = []
    for t in tick:
        try:
            s = yf.Ticker(t).get_shares_full(start="2015-01-01")
        except Exception as e:
            print(t, "shares error", str(e)[:60]); s = None
        if s is not None and len(s):
            s = s[~s.index.duplicated(keep="last")]
            rows.append(pd.DataFrame({"date": s.index.tz_localize(None).normalize(), "ticker": t, "shares": s.values}))
        print(t, 0 if s is None else len(s), flush=True)
        time.sleep(1)
    if rows:
        pd.concat(rows).to_csv(DATA / "shares.csv", index=False)


if __name__ == "__main__":
    main()
