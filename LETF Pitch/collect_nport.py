"""Fund size history for every ETF in data/universe.csv, from SEC Form N-PORT-P.

N-PORT is filed monthly but only the quarter-end report is public. Each public report still carries the month-end
net assets of the quarter's last month plus, for all three months, the total return (rtn1..3) and the flows
(sales, redemptions). That is enough to walk net assets back to each of the three month ends.

Outputs:
  data/nport.csv  series_id, accession, rep_date, net_assets, tot_assets,
                  sales1..3, redemption1..3, reinvest1..3, rtn1..3 (% per month, month 3 = rep_date's month)
Raw XML is cached in data/nport_cache/ so reruns cost nothing.
"""

import re, time
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
CACHE = DATA / "nport_cache"
CACHE.mkdir(exist_ok=True)
UA = {"User-Agent": "QFS research hanselliu206@gmail.com"}
GAP = 0.12  # SEC fair-access limit is 10 requests/second


def get(url):
    for k in range(5):
        time.sleep(GAP)
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    return None


def filings(series_id):
    atom = get("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=" + series_id +
               "&type=NPORT-P&dateb=&owner=include&count=100&output=atom") or ""
    hrefs = re.findall(r"<filing-href>([^<]*)", atom)
    types = re.findall(r"<filing-type>([^<]*)", atom)
    return [h.rsplit("/", 1)[0] for h, t in zip(hrefs, types) if t == "NPORT-P"]


def parse(xml):
    one = lambda tag: (re.search(rf"<{tag}>([^<]*)</{tag}>", xml) or [None, None])[1]
    row = {"series_id": one("seriesId"), "rep_date": one("repPdDate"), "net_assets": one("netAssets"),
           "tot_assets": one("totAssets")}
    for k in (1, 2, 3):
        m = re.search(rf"<mon{k}Flow([^/]*)/>", xml)
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1))) if m else {}
        row[f"sales{k}"], row[f"redemption{k}"], row[f"reinvest{k}"] = (attrs.get(x) for x in ("sales", "redemption", "reinvestment"))
    m = re.search(r"<monthlyTotReturn[^>]*?/>", xml)
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(0))) if m else {}
    for k in (1, 2, 3):
        row[f"rtn{k}"] = attrs.get(f"rtn{k}")
    return row


def main():
    u = pd.read_csv(DATA / "universe.csv")
    rows = []
    for i, sid in enumerate(u.series_id):
        for folder in filings(sid):
            acc = folder.rsplit("/", 1)[1]
            f = CACHE / f"{sid}_{acc}.xml"
            if not f.exists():
                xml = get(folder + "/primary_doc.xml")
                if xml is None:
                    continue
                f.write_text(xml)
            r = parse(f.read_text())
            if r["series_id"] != sid:  # trust-level filing for another series
                continue
            r["accession"] = acc
            rows.append(r)
        if i % 25 == 0:
            print(f"{i}/{len(u)} series, {len(rows)} reports", flush=True)
    d = pd.DataFrame(rows)
    num = [c for c in d.columns if c not in ("series_id", "rep_date", "accession")]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    d = d.sort_values(["series_id", "rep_date"]).drop_duplicates(["series_id", "rep_date"], keep="last")
    d.to_csv(DATA / "nport.csv", index=False)
    print(f"{len(d)} quarterly reports, {d.series_id.nunique()} series, {d.rep_date.min()}..{d.rep_date.max()}")


if __name__ == "__main__":
    main()
