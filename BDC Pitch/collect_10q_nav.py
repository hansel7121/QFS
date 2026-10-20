"""NAV per share for each quarter from the BDC's own 10-Q / 10-K (statement of assets and liabilities, current period
first), 2012 onward. Only the first ~800KB of the main document is read (the balance sheet comes early).

Used for the 2012-2022 history, where XBRL NAV tags are sparse. Press releases (collect_press_nav.py) give the
earlier public timestamp; this gives the reliable value.

Output: data/filing_nav.csv  ticker, cik, form, filed, accepted, period, nav_ps, snippet
"""

import re
import pandas as pd
from collect_sec_bdc import DATA, get
from collect_press_nav import text, find_nav


def main():
    uni = pd.read_csv(DATA / "universe.csv")
    uni = uni[uni.listed]
    out_f = DATA / "filing_nav.csv"
    done = set(pd.read_csv(out_f).accession) if out_f.exists() else set()
    for _, u in uni.iterrows():
        j = get(f"https://data.sec.gov/submissions/CIK{int(u.cik):010d}.json").json()
        blocks = [j["filings"]["recent"]]
        for f in j["filings"].get("files", []):
            r = get(f"https://data.sec.gov/submissions/{f['name']}")
            if r is not None:
                blocks.append(r.json())
        rows = []
        for b in blocks:
            for form, fd, acc, acc_t, rd, doc in zip(b["form"], b["filingDate"], b["accessionNumber"], b["acceptanceDateTime"],
                                                     b["reportDate"], b["primaryDocument"]):
                if form not in ("10-Q", "10-K") or fd < "2012-01-01" or acc in done or not doc:
                    continue
                r = get(f"https://www.sec.gov/Archives/edgar/data/{int(u.cik)}/{acc.replace('-', '')}/{doc}",
                        headers={"Range": "bytes=0-800000"})
                f = find_nav(text(r.text)) if r is not None else None
                rows.append({"ticker": u.ticker, "cik": u.cik, "accession": acc, "form": form, "filed": fd, "accepted": acc_t,
                             "period": rd, "nav_ps": f["nav_ps"] if f else None, "cands": f["cands"] if f else None,
                             "snippet": f["snippet"] if f else None})
        if rows:
            pd.DataFrame(rows).to_csv(out_f, mode="a", header=not out_f.exists(), index=False)
        print(u.ticker, len(rows), sum(r["nav_ps"] is not None for r in rows), flush=True)


if __name__ == "__main__":
    main()
