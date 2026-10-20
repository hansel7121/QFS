"""Weekly at-the-market (ATM) share sales and bitcoin purchases disclosed by Strategy (MSTR) in 8-K filings, 2024-2026.

Strategy files an 8-K most Mondays: "sold an aggregate of N shares ... for aggregate net proceeds of approximately $X"
and "acquired approximately K bitcoins for approximately $Y". The script sums every "net proceeds of approximately $X
million/billion" figure in the filing as ATM proceeds (common + preferred programs) and reads the bitcoin purchase.

Output: data/mstr_8k.csv  filed (date), accession, atm_musd, btc_bought_musd
"""

import re, time
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "QFS research hanselliu206@gmail.com"}
CIK = 1050446   # MicroStrategy / Strategy


def get(url):
    for k in range(5):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    raise RuntimeError("failed: " + url)


def amt(num, unit):
    v = float(num.replace(",", ""))
    return v * 1000 if unit.lower().startswith("b") else v


def main():
    sub = get(f"https://data.sec.gov/submissions/CIK{CIK:010d}.json").json()["filings"]["recent"]
    f = pd.DataFrame({k: sub[k] for k in ["form", "filingDate", "accessionNumber", "primaryDocument"]})
    f = f[(f.form == "8-K") & (f.filingDate >= "2024-01-01")]
    rows = []
    for r in f.itertuples():
        url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{r.accessionNumber.replace('-', '')}/{r.primaryDocument}"
        time.sleep(0.15)
        h = get(url).text
        t = re.sub(r"<[^>]+>", " ", h); t = re.sub(r"&#160;|&nbsp;", " ", t); t = re.sub(r"\s+", " ", t)
        atm = sum(amt(n, u) for n, u in re.findall(r"net proceeds[^$]{0,60}\$\s?([\d,.]+)\s*(million|billion)", t, re.I))
        btc = re.findall(r"acquired (?:approximately |an aggregate of )?[\d,]+ bitcoins? for (?:approximately |an aggregate of )?\$\s?([\d,.]+)\s*(million|billion)", t, re.I)
        rows.append({"filed": r.filingDate, "accession": r.accessionNumber, "atm_musd": atm, "btc_bought_musd": sum(amt(n, u) for n, u in btc)})
    d = pd.DataFrame(rows).sort_values("filed")
    d.to_csv(DATA / "mstr_8k.csv", index=False)
    print(len(d), "8-Ks;", (d.atm_musd > 0).sum(), "with ATM proceeds; total $bn", round(d.atm_musd.sum() / 1e3, 1))


if __name__ == "__main__":
    main()
