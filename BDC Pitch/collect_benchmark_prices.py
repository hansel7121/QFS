"""Daily prices for two comparison groups used to ask "is reversal special in BDCs?":
mortgage REITs (leveraged, high-dividend, retail-held, trade vs book value, like BDCs) and small/mid-cap regional banks.
Same survivorship caveat as the BDC list: tickers listed today.
Output: data/benchmark_prices.csv (date, ticker, group, close, adj_close, volume).
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"
GROUPS = {
    "mREIT": "NLY AGNC STWD BXMT ABR TWO ARR IVR MFA NYMT CIM PMT RWT DX ORC EFC LADR KREF TRTX ACRE GPMT ARI MITT CMTG RITM".split(),
    "regional bank": ("CVBF FFIN UBSI HOMB ONB FULT FNB WSFS SFNC TRMK CBU NBTB FRME FCF FBK HTLF BANR WAFD PPBI COLB "
                      "IBOC CATY HOPE BOH FIBK INDB UCBI SBCF TOWN").split(),
}

frames = []
for grp, tickers in GROUPS.items():
    raw = yf.download(tickers, start="2012-01-01", auto_adjust=False, progress=False, group_by="column", threads=True)
    df = raw[["Close", "Adj Close", "Volume"]].stack(level=1, future_stack=True).reset_index()
    df.columns = ["date", "ticker", "close", "adj_close", "volume"]
    df["group"] = grp
    frames.append(df.dropna(subset=["close"]))
out = pd.concat(frames)
out.to_csv(DATA / "benchmark_prices.csv", index=False)
print(out.groupby("group").ticker.nunique(), out.date.min(), out.date.max())
