"""Daily individual tax refunds paid by the IRS (Daily Treasury Statement, fiscaldata.treasury.gov) and prices of
retail-heavy assets (Yahoo).

The DTS category name changed over time ("IRS Tax Refunds Individual (EFT)", "Taxes - Individual Tax Refunds (EFT)",
cheque refunds listed separately): every withdrawal line containing "Individual" and "Refund" is summed.

Outputs:
  data/refunds.csv  date, refunds_musd  (published the next business day at about 16:00 ET)
  data/prices.csv   date, ticker, adj_close
"""

from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/deposits_withdrawals_operating_cash"


def main():
    rows, page = [], 1
    while True:
        r = requests.get(URL, params={"filter": "transaction_type:eq:Withdrawals", "fields": "record_date,transaction_catg,transaction_today_amt",
                                      "page[size]": 10000, "page[number]": page}, timeout=120).json()
        rows += r["data"]
        if page >= r["meta"]["total-pages"]:
            break
        page += 1
        print(page, len(rows), flush=True)
    d = pd.DataFrame(rows)
    m = d.transaction_catg.str.contains("Individual", case=False) & d.transaction_catg.str.contains("Refund", case=False)
    d = d[m]
    d["amt"] = pd.to_numeric(d.transaction_today_amt, errors="coerce")
    out = d.groupby("record_date").amt.sum().rename("refunds_musd").reset_index().rename(columns={"record_date": "date"})
    out.to_csv(DATA / "refunds.csv", index=False)
    print(out.date.min(), out.date.max(), len(out), d.transaction_catg.unique())
    raw = yf.download(["SPY", "IWM", "ARKK", "QQQ", "BTC-USD", "XBI", "GME"], start="2005-01-01", auto_adjust=False, progress=False)
    raw["Adj Close"].stack().rename("adj_close").reset_index().rename(columns={"Date": "date", "Ticker": "ticker"}).to_csv(DATA / "prices.csv", index=False)


if __name__ == "__main__":
    main()
