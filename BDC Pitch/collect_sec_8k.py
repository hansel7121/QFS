"""Earnings-release dates for each BDC: 8-K filings with Item 2.02 (Results of Operations), from EDGAR submissions.

Output data/earnings_8k.csv: ticker, cik, filed, accepted (EDGAR acceptance time, US Eastern), items.
"""

import pandas as pd
from collect_sec_bdc import DATA, get


def main():
    uni = pd.read_csv(DATA / "universe.csv")
    rows = []
    for _, u in uni.iterrows():
        j = get(f"https://data.sec.gov/submissions/CIK{int(u.cik):010d}.json").json()
        blocks = [j["filings"]["recent"]]
        for f in j["filings"].get("files", []):
            r = get(f"https://data.sec.gov/submissions/{f['name']}")
            if r is not None:
                blocks.append(r.json())
        for b in blocks:
            for form, fd, acc_t, items in zip(b["form"], b["filingDate"], b["acceptanceDateTime"], b["items"]):
                if form == "8-K" and "2.02" in (items or "") and fd >= "2022-01-01":
                    rows.append({"ticker": u.ticker, "cik": u.cik, "filed": fd, "accepted": acc_t, "items": items})
        print(u.ticker, sum(r["ticker"] == u.ticker for r in rows), flush=True)
    pd.DataFrame(rows).to_csv(DATA / "earnings_8k.csv", index=False)


if __name__ == "__main__":
    main()
