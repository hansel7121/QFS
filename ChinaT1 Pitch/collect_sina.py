"""A-share ETF daily OHLC from Sina via AkShare (unadjusted; intraday open->close returns need no adjustment).

Output: data/sina_etfs.csv  date, code, name, open, close, amount
"""

import time
from pathlib import Path
import pandas as pd
import akshare as ak

DATA = Path(__file__).parent / "data"
ETFS = {"sh510300": "CSI300 (Huatai)", "sz159919": "CSI300 (Harvest)", "sh510330": "CSI300 (ChinaAMC)", "sh510050": "SSE50",
        "sh510500": "CSI500 (Southern)", "sz159922": "CSI500 (E Fund)", "sh512500": "CSI500 (ChinaAMC)",
        "sh512100": "CSI1000 (Southern)", "sz159845": "CSI1000 (Huaxia)", "sz159915": "ChiNext (E Fund)", "sh588000": "STAR50 (ChinaAMC)",
        "sh510210": "SSE Composite (Fullgoal)"}

if __name__ == "__main__":
    rows = []
    for code, name in ETFS.items():
        for k in range(4):
            try:
                g = ak.fund_etf_hist_sina(symbol=code); break
            except Exception as e:
                print("retry", code, e); time.sleep(10 * (k + 1))
        g["code"], g["name"] = code, name
        rows.append(g[["date", "code", "name", "open", "close", "amount"]]); print(code, len(g), g.date.min(), flush=True)
        time.sleep(3)
    pd.concat(rows).to_csv(DATA / "sina_etfs.csv", index=False)
