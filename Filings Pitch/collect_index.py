"""EDGAR quarterly full index (form.idx), 2004 onward: every SC 13D (activist / >5% stake with intent),
SC 13D/A, NT 10-K and NT 10-Q filing with the subject company's CIK and filing date.

Output: data/filings.csv  form, company, cik, date_filed, path
"""

import re, time
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
UA = {"User-Agent": "QFS research hanselliu206@gmail.com"}
KEEP = {"SC 13D", "SC 13D/A", "NT 10-K", "NT 10-Q", "SCHEDULE 13D", "SCHEDULE 13D/A"}


def main():
    rows = []
    for y in range(2004, 2027):
        for q in range(1, 5):
            r = requests.get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx", headers=UA, timeout=120)
            if r.status_code != 200:
                print(y, q, r.status_code, flush=True); continue
            n0 = len(rows)
            for line in r.text.splitlines():
                form = line[:12].strip()
                if form not in KEEP:
                    continue
                m = re.match(r"(.{12})(.+?)\s{2,}(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\S+)", line)
                if m:
                    rows.append((form, m.group(2).strip(), m.group(3), m.group(4), m.group(5)))
            print(y, q, len(rows) - n0, flush=True)
            time.sleep(0.2)
    pd.DataFrame(rows, columns=["form", "company", "cik", "date_filed", "path"]).to_csv(DATA / "filings.csv", index=False)


if __name__ == "__main__":
    main()
