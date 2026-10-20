"""Flow-induced trading (Lou 2012) from SEC Form N-PORT bulk data sets, 2019Q4 onwards.

For every fund report (series, report date): net flow over its 3 reported months = sum(sales + reinvestment -
redemptions) / start-of-period net assets (net assets minus those flows). Each US-dollar common-stock holding
contributes value x net-flow fraction: the dollars that fund had to buy (inflow) or sell (outflow) in that stock if it
scaled its portfolio pro rata. Summed by stock ticker and report quarter.

Each quarterly zip (~400MB) is downloaded, processed in chunks, then deleted.

Output: data/fit.csv  quarter (report period), ticker, fit_usd, fund_value_usd, n_funds, filed_by (latest filing date
in the group: when the information was public)
"""

import io, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
TMP = DATA / "tmp"; TMP.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "QFS research hanselliu206@gmail.com"}
URL = "https://www.sec.gov/files/dera/data/form-n-port-data-sets/{}_nport.zip"


def process(z):
    sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t", usecols=["ACCESSION_NUMBER", "FILING_DATE", "REPORT_DATE", "SUB_TYPE"], dtype=str)
    sub = sub[sub.SUB_TYPE == "NPORT-P"]
    flow_cols = [f"{k}_FLOW_MON{m}" for k in ("SALES", "REINVESTMENT", "REDEMPTION") for m in (1, 2, 3)]
    info = pd.read_csv(z.open("FUND_REPORTED_INFO.tsv"), sep="\t", usecols=["ACCESSION_NUMBER", "SERIES_ID", "NET_ASSETS"] + flow_cols, dtype={"ACCESSION_NUMBER": str, "SERIES_ID": str})
    info[flow_cols] = info[flow_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    net = sum(info[f"SALES_FLOW_MON{m}"] + info[f"REINVESTMENT_FLOW_MON{m}"] - info[f"REDEMPTION_FLOW_MON{m}"] for m in (1, 2, 3))
    na = pd.to_numeric(info.NET_ASSETS, errors="coerce")
    start = na - net
    info["flow_frac"] = (net / start).where(start > 1e6).clip(-0.9, 2.0)
    info = info.merge(sub, on="ACCESSION_NUMBER")[["ACCESSION_NUMBER", "flow_frac", "FILING_DATE", "REPORT_DATE"]].dropna()
    acc = set(info.ACCESSION_NUMBER)
    # holdings: US-dollar common equity
    keep = []
    for ch in pd.read_csv(z.open("FUND_REPORTED_HOLDING.tsv"), sep="\t", chunksize=2_000_000, dtype=str,
                          usecols=["ACCESSION_NUMBER", "HOLDING_ID", "CURRENCY_CODE", "CURRENCY_VALUE", "ASSET_CAT", "PAYOFF_PROFILE"]):
        ch = ch[(ch.ASSET_CAT == "EC") & (ch.CURRENCY_CODE == "USD") & (ch.PAYOFF_PROFILE == "Long") & ch.ACCESSION_NUMBER.isin(acc)]
        keep.append(ch[["ACCESSION_NUMBER", "HOLDING_ID", "CURRENCY_VALUE"]])
    h = pd.concat(keep)
    ids = set(h.HOLDING_ID)
    tk = []
    for ch in pd.read_csv(z.open("IDENTIFIERS.tsv"), sep="\t", chunksize=3_000_000, dtype=str, usecols=["HOLDING_ID", "IDENTIFIER_TICKER"]):
        ch = ch[ch.HOLDING_ID.isin(ids) & ch.IDENTIFIER_TICKER.notna()]
        tk.append(ch)
    tk = pd.concat(tk).drop_duplicates("HOLDING_ID")
    h = h.merge(tk, on="HOLDING_ID").merge(info, on="ACCESSION_NUMBER")
    h["value"] = pd.to_numeric(h.CURRENCY_VALUE, errors="coerce")
    h["ticker"] = h.IDENTIFIER_TICKER.str.upper().str.strip()
    h = h[h.ticker.str.match(r"^[A-Z]{1,5}([.\-][A-Z])?$", na=False)]
    h["quarter"] = pd.to_datetime(h.REPORT_DATE, format="%d-%b-%Y").dt.to_period("Q")
    h["filed"] = pd.to_datetime(h.FILING_DATE, format="%d-%b-%Y")
    h["fit"] = h.value * h.flow_frac
    return h.groupby(["quarter", "ticker"]).agg(fit_usd=("fit", "sum"), fund_value_usd=("value", "sum"),
                                                 n_funds=("ACCESSION_NUMBER", "nunique"), filed_by=("filed", "max")).reset_index()


def main():
    qs = sys.argv[1:] or [f"{y}q{q}" for y in range(2019, 2027) for q in range(1, 5) if (y, q) >= (2019, 4) and (y, q) <= (2026, 3)]
    out = DATA / "fit.csv"
    done = set(pd.read_csv(out).src) if out.exists() else set()
    for q in qs:
        if q in done:
            continue
        f = TMP / f"{q}.zip"
        r = requests.get(URL.format(q), headers=UA, timeout=600, stream=True)
        if r.status_code != 200:
            print(q, r.status_code); continue
        with open(f, "wb") as fh:
            for b in r.iter_content(1 << 22):
                fh.write(b)
        with zipfile.ZipFile(f) as z:
            g = process(z)
        g["src"] = q
        g.to_csv(out, mode="a", header=not out.exists(), index=False)
        f.unlink()
        print(q, len(g), "stock-quarters", g.quarter.astype(str).value_counts().head(3).to_dict(), flush=True)


if __name__ == "__main__":
    main()
