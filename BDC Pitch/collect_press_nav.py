"""Point-in-time NAV per share from BDC earnings press releases (8-K Item 2.02, exhibit 99.x), 2012 onward.

XBRL NAV tags only cover BDCs fully from about 2022, so for the long history this reads the earnings release itself:
"net asset value per share was $19.80 as of September 30, 2024". The release's EDGAR acceptance time is the moment the
NAV became public, which is exactly the event a trade can react to.

Output: data/press_nav.csv  ticker, cik, accession, accepted, doc, nav_ps (first per-share NAV quoted), cands (up to
        four numbers following the first NAV-per-share phrase that is followed by a number, e.g. a table row
        "Net asset value per share 27.27 27.15"), estimate (release gives a preliminary estimate / range),
        adjusted ("adjusted NAV"), snippet
"""

import re
import pandas as pd
from collect_sec_bdc import DATA, get

KEY = re.compile(r"(?:net asset value|net assets|NAV)(?:\s*\([^)]{0,12}\))?\s+per\s+(?:common\s+)?share", re.I)
NUM = re.compile(r"\$?\s?(\d{1,3}\.\d{2})(?!\d)")


def find_nav(t):
    """First number >= 2.00 within 150 characters after a 'NAV per share' phrase (skips dividend amounts), plus up to
    four candidates from that spot (table rows list several periods) and flags for estimates / adjusted NAV."""
    for k in KEY.finditer(t):
        win = t[k.end(): k.end() + 150]
        vals = [(float(m.group(1)), m) for m in NUM.finditer(win) if float(m.group(1)) >= 2.0]
        if vals:
            v, m = vals[0]
            ctx = t[max(0, k.start() - 120): k.end() + 150].lower()
            return {"nav_ps": v, "cands": ";".join(str(x) for x, _ in vals[:4]),
                    "estimate": bool(re.search(r"estimat|range of|between \$", ctx)), "adjusted": "adjusted" in ctx,
                    "snippet": t[max(0, k.start() - 40): k.end() + m.end() + 40]}
    return None


def text(html):
    t = re.sub(r"<[^>]+>", " ", html)
    t = re.sub(r"&#160;|&nbsp;|&#xa0;", " ", t)
    t = re.sub(r"&#8217;|&rsquo;", "'", t)
    return re.sub(r"\s+", " ", t)


def one(cik, acc):
    nod = acc.replace("-", "")
    idx = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nod}/index.json")
    if idx is None:
        return None, {}
    items = idx.json()["directory"]["item"]
    docs = [i["name"] for i in items if re.search(r"(ex|exhibit)[-_]?99|ex99|dex99", i["name"], re.I) and i["name"].lower().endswith((".htm", ".html", ".txt"))]
    if not docs:
        docs = [i["name"] for i in items if i["name"].lower().endswith((".htm", ".html")) and "index" not in i["name"].lower()]
    for d in docs:
        r = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nod}/{d}")
        if r is None:
            continue
        t = text(r.text)
        f = find_nav(t)
        if f is not None:
            return d, f
    return (docs[0] if docs else None), {}


def main():
    uni = pd.read_csv(DATA / "universe.csv")
    uni = uni[uni.listed]
    out_f = DATA / "press_nav.csv"
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
            for form, fd, acc, acc_t, items in zip(b["form"], b["filingDate"], b["accessionNumber"], b["acceptanceDateTime"], b["items"]):
                if form == "8-K" and "2.02" in (items or "") and fd >= "2012-01-01" and acc not in done:
                    d, f = one(u.cik, acc)
                    rows.append({"ticker": u.ticker, "cik": u.cik, "accession": acc, "accepted": acc_t, "doc": d, **f})
        if rows:
            pd.DataFrame(rows).to_csv(out_f, mode="a", header=not out_f.exists(), index=False)
        print(u.ticker, len(rows), sum(r.get("nav_ps") is not None for r in rows), flush=True)


if __name__ == "__main__":
    main()
