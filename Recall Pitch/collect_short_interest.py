"""FINRA consolidated short interest (all US equities, twice-monthly settlement dates), 2019-2026, via FINRA's public
Query API. Published about 8 business days after each settlement date.

Output: data/short_interest.csv  settlement_date, ticker, short_qty, adv_qty, days_to_cover
"""

import time
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
H = {"Content-Type": "application/json", "Accept": "application/json"}


def settlement_dates():
    """FINRA settlement dates: mid-month and month-end; probe with a known liquid symbol."""
    body = {"limit": 5000, "fields": ["settlementDate"], "compareFilters": [{"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": "AAPL"}]}
    r = requests.post(URL, json=body, headers=H, timeout=60).json()
    return sorted({x["settlementDate"] for x in r if x["settlementDate"] >= "2019-01-01"})


def main():
    out = DATA / "short_interest.csv"
    done = set(pd.read_csv(out).settlement_date) if out.exists() else set()
    for d in settlement_dates():
        if d in done:
            continue
        rows, off = [], 0
        while True:
            body = {"limit": 5000, "offset": off, "fields": ["settlementDate", "symbolCode", "currentShortPositionQuantity", "averageDailyVolumeQuantity", "daysToCoverQuantity"],
                    "compareFilters": [{"compareType": "EQUAL", "fieldName": "settlementDate", "fieldValue": d}]}
            for k in range(4):
                try:
                    r = requests.post(URL, json=body, headers=H, timeout=120).json(); break
                except Exception:
                    time.sleep(5 * (k + 1)); r = []
            rows += r; off += len(r)
            if len(r) < 5000:
                break
        df = pd.DataFrame(rows).rename(columns={"settlementDate": "settlement_date", "symbolCode": "ticker", "currentShortPositionQuantity": "short_qty",
                                                "averageDailyVolumeQuantity": "adv_qty", "daysToCoverQuantity": "days_to_cover"})
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(d, len(df), flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
