"""iShares Russell 2000 (IWM) and Russell 1000 (IWB) holdings from SEC N-PORT bulk data sets, every report 2019-2026.

Comparing the last report before the June reconstitution (March 31) with the first one after (June 30) gives the
names that entered / left each index at the recon.

Output: data/holdings.csv  series (IWM/IWB), report_date, ticker, value_usd, shares
"""

import sys, zipfile
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
TMP = DATA / "tmp"; TMP.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "QFS research hanselliu206@gmail.com"}
SERIES = {"S000004344": "IWM", "S000004347": "IWB"}


def process(z):
    info = pd.read_csv(z.open("FUND_REPORTED_INFO.tsv"), sep="\t", usecols=["ACCESSION_NUMBER", "SERIES_ID"], dtype=str)
    info = info[info.SERIES_ID.isin(SERIES)]
    sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t", usecols=["ACCESSION_NUMBER", "REPORT_DATE"], dtype=str)
    info = info.merge(sub, on="ACCESSION_NUMBER")
    acc = set(info.ACCESSION_NUMBER)
    if not acc:
        return pd.DataFrame()
    # iShares reports CUSIPs without tickers: map CUSIP -> ticker from every other fund in the file that reports both
    ec = pd.concat(ch[(ch.ASSET_CAT == "EC") & ch.ISSUER_CUSIP.str.match(r"^[0-9A-Z]{9}$", na=False)] for ch in pd.read_csv(
        z.open("FUND_REPORTED_HOLDING.tsv"), sep="\t", chunksize=2_000_000, dtype=str,
        usecols=["ACCESSION_NUMBER", "HOLDING_ID", "ISSUER_CUSIP", "BALANCE", "CURRENCY_VALUE", "ASSET_CAT"]))
    tk = pd.concat(ch.dropna(subset=["IDENTIFIER_TICKER"]) for ch in pd.read_csv(
        z.open("IDENTIFIERS.tsv"), sep="\t", chunksize=3_000_000, dtype=str, usecols=["HOLDING_ID", "IDENTIFIER_TICKER"]))
    m = ec[["HOLDING_ID", "ISSUER_CUSIP"]].merge(tk, on="HOLDING_ID")
    m["IDENTIFIER_TICKER"] = m.IDENTIFIER_TICKER.str.upper().str.strip()
    cmap = m.groupby("ISSUER_CUSIP").IDENTIFIER_TICKER.agg(lambda s: s.value_counts().index[0])
    h = ec[ec.ACCESSION_NUMBER.isin(acc)].copy()
    h["IDENTIFIER_TICKER"] = h.ISSUER_CUSIP.map(cmap)
    h = h.dropna(subset=["IDENTIFIER_TICKER"]).merge(info, on="ACCESSION_NUMBER")
    return pd.DataFrame({"series": h.SERIES_ID.map(SERIES), "report_date": pd.to_datetime(h.REPORT_DATE, format="%d-%b-%Y"),
                         "ticker": h.IDENTIFIER_TICKER.str.upper().str.replace(".", "-"),
                         "value_usd": pd.to_numeric(h.CURRENCY_VALUE, errors="coerce"), "shares": pd.to_numeric(h.BALANCE, errors="coerce")})


def main():
    out = DATA / "holdings.csv"
    for q in sys.argv[1:] or [f"{y}q{k}" for y in range(2019, 2027) for k in range(1, 5) if (y, k) >= (2019, 4) and (y, k) <= (2026, 3)]:
        f = TMP / f"{q}.zip"
        r = requests.get(f"https://www.sec.gov/files/dera/data/form-n-port-data-sets/{q}_nport.zip", headers=UA, timeout=900, stream=True)
        if r.status_code != 200:
            print(q, r.status_code); continue
        with open(f, "wb") as fh:
            for b in r.iter_content(1 << 22): fh.write(b)
        with zipfile.ZipFile(f) as z:
            g = process(z)
        f.unlink()
        if len(g):
            g.to_csv(out, mode="a", header=not out.exists(), index=False)
            print(q, g.groupby(["series", "report_date"]).size().to_dict(), flush=True)


if __name__ == "__main__":
    main()
