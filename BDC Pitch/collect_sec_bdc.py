"""Pull BDC fundamentals from SEC EDGAR.

Outputs (data/):
  universe.csv        ticker, cik, name, listed
  bdc_nav.csv         quarterly NAV/share, total fair value, total cost, net assets, with the date each was first filed
  bdc_positions.csv.gz  one row per investment per filing (10-Q / 10-K, periods from 2022-06 on, when BDC
                        schedules of investments started being XBRL-tagged): fair value, cost, principal, rate, maturity

Raw XBRL zips are cached in data/sec_cache/ so the script can be stopped and re-run.
SEC asks for <= 10 requests/s and a User-Agent with contact info; set SEC_UA to override.
"""

import io, json, os, re, sys, time, zipfile
from pathlib import Path
import pandas as pd
import requests
from lxml import etree

DATA = Path(__file__).parent / "data"
CACHE = DATA / "sec_cache"
CACHE.mkdir(parents=True, exist_ok=True)
UA = os.environ.get("SEC_UA", "QFS BDC Pitch research admin@qfs-bdc-pitch.org")
S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
POS_START = "2022-06-30"

# Listed BDCs (tradable) and large non-traded BDCs (no price, but their loan marks are peers for shared loans)
LISTED = ("ARCC OBDC FSK BXSL GBDC PSEC MAIN HTGC OCSL TSLX GSBD NMFC BBDC CGBD SLRC CSWC TCPC PFLT PNNT MFIC "
          "BCSF TRIN FDUS GAIN GLAD CCAP KBDC MSDL NCDL PSBD OFS SAR WHF HRZN TPVG RWAY GECC ICMB OXSQ SCM CION "
          "BCIC").split()
NONTRADED = {1803498: "BCRED", 1838126: "HLEND", 1812554: "OCIC", 1918712: "ASIF", 1837532: "ADS",
             1930087: "GCRED", 1872371: "OSCF", 1973476: "NHPIF"}

CONCEPTS = {
    "InvestmentOwnedAtFairValue": "fv",
    "InvestmentOwnedAtCost": "cost",
    "InvestmentOwnedBalancePrincipalAmount": "principal",
    "InvestmentInterestRate": "rate",
    "InvestmentBasisSpreadVariableRate": "spread",
    "InvestmentMaturityDate": "maturity",
}
POS_AXES = ("InvestmentIdentifierAxis", "InvestmentIssuerNameAxis")
_last = [0.0]


def get(url, **kw):
    for attempt in range(6):
        wait = 0.15 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = S.get(url, timeout=60, **kw)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (200, 206):      # 206 = partial content (HTTP Range requests)
            return r
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(2 ** attempt * 2)
            continue
        return None
    return None


def universe():
    tick = get("https://www.sec.gov/files/company_tickers.json").json()
    m = {v["ticker"]: v["cik_str"] for v in tick.values()}
    rows = [{"ticker": t, "cik": m[t], "listed": True} for t in LISTED if t in m]
    rows += [{"ticker": t, "cik": c, "listed": False} for c, t in NONTRADED.items()]
    return pd.DataFrame(rows)


def facts_nav(cik):
    r = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    if r is None:
        return pd.DataFrame(), None
    j = r.json()
    g = j.get("facts", {}).get("us-gaap", {})
    out = []
    for concept, col in [("NetAssetValuePerShare", "nav_ps"), ("InvestmentOwnedAtFairValue", "fv_total"),
                         ("InvestmentOwnedAtCost", "cost_total"), ("AssetsNet", "net_assets"),
                         ("StockholdersEquity", "equity")]:
        for unit in g.get(concept, {}).get("units", {}).values():
            for f in unit:
                if f.get("form") in ("10-Q", "10-K", "10-Q/A", "10-K/A") and "end" in f and "start" not in f:
                    out.append({"period_end": f["end"], "filed": f["filed"], "col": col, "val": f["val"]})
    if not out:
        return pd.DataFrame(), j.get("entityName")
    df = pd.DataFrame(out)
    # first filing that reported each value = when the market could know it
    df = df.sort_values("filed").drop_duplicates(["period_end", "col"])
    wide = df.pivot(index="period_end", columns="col", values="val")
    filed = df[df.col == "nav_ps"].set_index("period_end")["filed"] if (df.col == "nav_ps").any() else None
    wide["filed"] = filed
    return wide.reset_index(), j.get("entityName")


def filings(cik):
    r = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if r is None:
        return []
    j = r.json()
    blocks = [j["filings"]["recent"]]
    for f in j["filings"].get("files", []):
        rr = get(f"https://data.sec.gov/submissions/{f['name']}")
        if rr is not None:
            blocks.append(rr.json())
    out = []
    for b in blocks:
        for form, fd, rd, acc in zip(b["form"], b["filingDate"], b["reportDate"], b["accessionNumber"]):
            if form in ("10-Q", "10-K") and rd and rd >= POS_START:
                out.append({"form": form, "filed": fd, "period_end": rd, "acc": acc})
    return out


def fetch_zip(cik, acc):
    path = CACHE / f"{cik}_{acc}.zip"
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    a = acc.replace("-", "")
    r = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{a}/{acc}-xbrl.zip")
    if r is None:
        # older filings: no -xbrl.zip, look for the instance in the index
        idx = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{a}/index.json")
        if idx is None:
            return None
        names = [i["name"] for i in idx.json()["directory"]["item"]]
        inst = [n for n in names if n.endswith("_htm.xml")] or \
               [n for n in names if n.endswith(".xml") and not re.search(r"_(cal|def|lab|pre)\.xml$|FilingSummary", n)]
        if not inst:
            return None
        r2 = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{a}/{inst[0]}")
        if r2 is None:
            return None
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(inst[0], r2.content)
        data = buf.getvalue()
    else:
        data = r.content
    path.write_bytes(data)
    return data


IX = "{http://www.xbrl.org/2013/inlineXBRL}"


def ix_value(el):
    """Numeric value of an ix:nonFraction (handles scale, sign and the dash used for zero)."""
    if el.get("{http://www.w3.org/2001/XMLSchema-instance}nil") == "true":
        return None
    txt = "".join(el.itertext()).strip()
    fmt = el.get("format", "")
    if "zerodash" in fmt or "fixed-zero" in fmt or txt in ("-", "—", "–"):
        v = 0.0
    else:
        if "numcommadecimal" in fmt or "num-comma-decimal" in fmt:
            txt = txt.replace(".", "").replace(" ", "").replace(",", ".")
        else:
            txt = txt.replace(",", "").replace(" ", "")
        try:
            v = float(txt)
        except ValueError:
            return None
    v *= 10 ** int(el.get("scale", "0") or 0)
    return -v if el.get("sign") == "-" else v


def iter_facts(root, inline):
    """Yield (concept localname, contextRef, value) from an XBRL instance or an inline XBRL document."""
    if inline:
        for el in root.iter(IX + "nonFraction", IX + "nonNumeric"):
            name = el.get("name", "").split(":")[-1]
            col = CONCEPTS.get(name)
            if col is None:
                continue
            val = " ".join("".join(el.itertext()).split()) if el.tag.endswith("nonNumeric") else ix_value(el)
            yield name, el.get("contextRef"), val
    else:
        for el in root:
            if not isinstance(el.tag, str) or el.text is None:
                continue
            name = etree.QName(el).localname
            if name in CONCEPTS:
                yield name, el.get("contextRef"), el.text.strip()


def parse_positions(zbytes, period_end):
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    names = z.namelist()
    inst = [n for n in names if n.endswith("_htm.xml")] or \
           [n for n in names if n.endswith(".xml") and not re.search(r"_(cal|def|lab|pre)\.xml$|FilingSummary", n)]
    inline = not inst
    if inline:
        # inline XBRL: the main document is the largest .htm in the zip (exhibits are small)
        htm = sorted((n for n in names if n.endswith(".htm")), key=lambda n: z.getinfo(n).file_size)
        if not htm:
            return []
        inst = [htm[-1]]
    root = etree.parse(z.open(inst[0]), etree.XMLParser(huge_tree=True, recover=True)).getroot()
    X = "{http://www.xbrl.org/2003/instance}"
    D = "{http://xbrl.org/2006/xbrldi}"
    ctx = {}
    for c in root.iter(X + "context"):
        inst_el = c.find(f".//{X}instant")
        if inst_el is None or inst_el.text.strip() != period_end:
            continue
        dims = {}
        for m in c.iter(D + "explicitMember", D + "typedMember"):
            ax = m.get("dimension").split(":")[-1]
            dims[ax] = " ".join("".join(m.itertext()).split()) if m.tag.endswith("typedMember") else (m.text or "").strip()
        if any(a in dims for a in POS_AXES):
            ctx[c.get("id")] = dims
    rows = {}
    for local, cref, val in iter_facts(root, inline):
        col = CONCEPTS[local]
        d = ctx.get(cref)
        if d is None or val is None:
            continue
        key = (d.get("InvestmentIdentifierAxis", ""), d.get("InvestmentIssuerNameAxis", ""))
        ndim = len(d)
        rec = rows.setdefault(key, {"identifier": key[0], "issuer_member": key[1], "_ndim": {}})
        # keep the value from the context with the fewest extra dimensions (avoids double counting sub-breakdowns)
        if col in rec and rec["_ndim"].get(col, 99) <= ndim:
            continue
        rec["_ndim"][col] = ndim
        rec["n_dims"] = ndim
        rec["other_dims"] = "; ".join(f"{k}={v}" for k, v in d.items() if k not in POS_AXES)
        rec[col] = val if col == "maturity" else pd.to_numeric(val, errors="coerce")
    out = []
    for rec in rows.values():
        rec.pop("_ndim")
        out.append(rec)
    return out


def main():
    uni = universe()
    navs, pos_frames = [], []
    done_file = DATA / "_done_positions.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    pos_parts = DATA / "positions_parts"
    pos_parts.mkdir(exist_ok=True)
    names = {}
    t0 = time.time()
    for i, row in uni.iterrows():
        cik, tk = int(row.cik), row.ticker
        nav, name = facts_nav(cik)
        names[tk] = name
        if len(nav):
            nav.insert(0, "ticker", tk)
            navs.append(nav)
        fl = filings(cik)
        print(f"[{i + 1}/{len(uni)}] {tk:6s} {str(name)[:40]:40s} nav_rows={len(nav):3d} filings={len(fl):2d} "
              f"t={time.time() - t0:5.0f}s", flush=True)
        for f in fl:
            tag = f"{tk}_{f['acc']}"
            if tag in done:
                continue
            zb = fetch_zip(cik, f["acc"])
            if zb is None:
                print(f"    missing xbrl {tk} {f['acc']}", flush=True)
                continue
            try:
                rows = parse_positions(zb, f["period_end"])
            except Exception as e:  # malformed instance: skip, keep going
                print(f"    parse fail {tk} {f['acc']}: {e}", flush=True)
                rows = []
            df = pd.DataFrame(rows)
            if len(df):
                df.insert(0, "ticker", tk)
                df.insert(1, "form", f["form"])
                df.insert(2, "period_end", f["period_end"])
                df.insert(3, "filed", f["filed"])
                df.insert(4, "acc", f["acc"])
                df.to_csv(pos_parts / f"{tag}.csv", index=False)
            with open(done_file, "a") as fh:
                fh.write(tag + "\n")
            print(f"    {f['form']} {f['period_end']} positions={len(df)}", flush=True)
    uni["name"] = uni.ticker.map(names)
    uni.to_csv(DATA / "universe.csv", index=False)
    pd.concat(navs).to_csv(DATA / "bdc_nav.csv", index=False)
    parts = [pd.read_csv(p) for p in sorted(pos_parts.glob("*.csv"))]
    allpos = pd.concat(parts, ignore_index=True)
    # a 10-K and 10-Q can both report the same period end (e.g. comparative); keep the first filed per ticker/period
    first = allpos.groupby(["ticker", "period_end"]).filed.transform("min")
    allpos = allpos[allpos.filed == first]
    allpos.to_csv(DATA / "bdc_positions.csv.gz", index=False)
    print(f"done in {time.time() - t0:.0f}s: {len(allpos):,} position rows, "
          f"{allpos.groupby(['ticker', 'period_end']).ngroups} filings", flush=True)


if __name__ == "__main__":
    main()
