"""A-share ETF daily OHLC from Eastmoney via AkShare (exchange-quality opening/closing auction prices).

Unadjusted prices: intraday (open -> close of the same day) returns are unaffected by splits/dividends; for
overnight returns the forward-adjusted ('qfq') series is also saved.

Output: data/em_etfs.csv  date, code, name, open, close, open_qfq, close_qfq, amount
"""

import time
from pathlib import Path
import pandas as pd
import akshare as ak

DATA = Path(__file__).parent / "data"
ETFS = {"510300": "CSI300 (Huatai)", "159919": "CSI300 (Harvest)", "510330": "CSI300 (ChinaAMC)", "510050": "SSE50",
        "510500": "CSI500 (Southern)", "159922": "CSI500 (E Fund)", "512500": "CSI500 (ChinaAMC)",
        "512100": "CSI1000 (Southern)", "159845": "CSI1000 (Huaxia)", "159915": "ChiNext (E Fund)", "588000": "STAR50 (ChinaAMC)",
        "510210": "SSE Composite (Fullgoal)"}

def fetch(code, adjust):
    for k in range(6):
        try:
            return ak.fund_etf_hist_em(symbol=code, period="daily", start_date="20100101", end_date="20261015", adjust=adjust)
        except Exception as e:
            print("   retry", code, adjust, type(e).__name__, flush=True); time.sleep(10 * (k + 1))
    raise RuntimeError(code)


if __name__ == "__main__":
    out = DATA / "em_etfs.csv"
    done = set(pd.read_csv(out, dtype={"code": str}).code) if out.exists() else set()
    for code, name in ETFS.items():
        if code in done:
            continue
        raw = fetch(code, ""); time.sleep(3)
        qfq = fetch(code, "qfq")
        m = raw[["日期", "开盘", "收盘", "成交额"]].merge(qfq[["日期", "开盘", "收盘"]], on="日期", suffixes=("", "_q"))
        m.columns = ["date", "open", "close", "amount", "open_qfq", "close_qfq"]
        m["code"], m["name"] = code, name
        m.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(code, name, len(m), m.date.min(), flush=True)
        time.sleep(5)
