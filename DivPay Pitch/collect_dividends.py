"""Cash dividend history (declaration, ex, record, payment dates, amount) for current S&P 500 members, from the
Nasdaq quote API. Used to rebuild aggregate dividend payments per day (Hartzmark & Solomon's predictable buying).

Output: data/dividends.csv  ticker, declared, ex_date, pay_date, amount
"""

import time
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def main():
    tick = pd.read_csv(DATA.parent.parent / "Overnight Pitch" / "data" / "panel.csv", usecols=["ticker"]).ticker.unique()
    out = DATA / "dividends.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["ticker"])
    done = set(old.ticker)
    rows = []
    for i, t in enumerate(sorted(tick)):
        if t == "SPY" or t in done:
            continue
        for k in range(3):                              # the API throttles: back off and retry
            try:
                d = requests.get(f"https://api.nasdaq.com/api/quote/{t.replace('-', '.')}/dividends?assetclass=stocks", headers=H, timeout=30).json()
                got = (d.get("data") or {}).get("dividends", {}).get("rows", []) or []
                for r in got:
                    if r.get("type") == "Cash":
                        rows.append((t, r["declarationDate"], r["exOrEffDate"], r["paymentDate"], r["amount"]))
                break
            except Exception as e:
                time.sleep(5 * (k + 1))
        time.sleep(1.0)
        if i % 100 == 0:
            print(i, len(rows), flush=True)
    d = pd.DataFrame(rows, columns=["ticker", "declared", "ex_date", "pay_date", "amount"])
    for c in ["declared", "ex_date", "pay_date"]:
        d[c] = pd.to_datetime(d[c], format="%m/%d/%Y", errors="coerce")
    d["amount"] = pd.to_numeric(d.amount.str.replace(r"[$,]", "", regex=True), errors="coerce")
    if len(old):
        old_ = old.copy()
        for c in ["declared", "ex_date", "pay_date"]:
            old_[c] = pd.to_datetime(old_[c])
        d = pd.concat([old_, d])
    d.to_csv(out, index=False)
    print(len(d), "dividends,", d.ticker.nunique(), "firms; with pay date:", d.pay_date.notna().mean().round(2),
          "with declaration date:", d.declared.notna().mean().round(2))


if __name__ == "__main__":
    main()
