"""SEC Insider Transactions Data Sets (Forms 3/4/5, structured, quarterly), 2006 onward.

Keeps open-market purchases (code P) and sales (code S) of non-derivative securities, joined to the filing date,
issuer and the reporting owner's relationship / title.

Output: data/insider_trades.csv  accession, filing_date, trans_date, issuer_cik, ticker, owner_cik, relationship,
        title, code, shares, price, owned_after, direct
"""

import io, time, zipfile
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
UA = {"User-Agent": "QFS research hanselliu206@gmail.com"}
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"


def quarter(q):
    for k in range(4):
        try:
            r = requests.get(URL.format(q=q), headers=UA, timeout=300)
            if r.status_code == 404:
                return None
            r.raise_for_status(); break
        except requests.RequestException:
            time.sleep(10 * (k + 1))
    else:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    rd = lambda f, cols: pd.read_csv(z.open(f), sep="\t", usecols=cols, dtype=str, quoting=3, on_bad_lines="skip")
    sub = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"])
    own = rd("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
    own = own.drop_duplicates("ACCESSION_NUMBER")
    tr = rd("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                                   "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS", "DIRECT_INDIRECT_OWNERSHIP"])
    tr = tr[tr.TRANS_CODE.isin(["P", "S"])]
    m = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    m = m[m.DOCUMENT_TYPE.isin(["4", "4/A"])]
    out = pd.DataFrame({"accession": m.ACCESSION_NUMBER, "filing_date": pd.to_datetime(m.FILING_DATE, format="%d-%b-%Y", errors="coerce"),
                        "trans_date": pd.to_datetime(m.TRANS_DATE, format="%d-%b-%Y", errors="coerce"), "issuer_cik": m.ISSUERCIK.str.lstrip("0"),
                        "ticker": m.ISSUERTRADINGSYMBOL.str.upper().str.strip(), "owner_cik": m.RPTOWNERCIK.str.lstrip("0"),
                        "relationship": m.RPTOWNER_RELATIONSHIP, "title": m.RPTOWNER_TITLE, "code": m.TRANS_CODE,
                        "shares": pd.to_numeric(m.TRANS_SHARES, errors="coerce"), "price": pd.to_numeric(m.TRANS_PRICEPERSHARE, errors="coerce"),
                        "owned_after": pd.to_numeric(m.SHRS_OWND_FOLWNG_TRANS, errors="coerce"), "direct": m.DIRECT_INDIRECT_OWNERSHIP})
    return out


def main():
    out = DATA / "insider_trades.csv"
    done_f = DATA / "_done_quarters.txt"
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    for y in range(2006, 2027):
        for k in range(1, 5):
            q = f"{y}q{k}"
            if q in done:
                continue
            df = quarter(q)
            if df is None:
                print(q, "missing", flush=True); continue
            df.to_csv(out, mode="a", header=not out.exists(), index=False)
            with open(done_f, "a") as fh: fh.write(q + "\n")
            print(q, len(df), "P", (df.code == "P").sum(), flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
