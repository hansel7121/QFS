"""
MOPS AGM-convening announcements (t05st01), with the time each was posted.

suspension.csv gives the forced-covering deadline D of every AGM ban but not when it became public. The AGM
date, and with it the book closure (59 days before the meeting) and D (6 trading days before the closure),
is public from the moment the company posts its "董事會決議召開股東常會" material information on MOPS.

For every AGM event in the backtest's trade list (days-to-cover on D-17 >= THR, 20-day turnover >= NT$20m,
same construction as build_backtest.py) this pulls the company's material-information list for that ROC year
(one request), opens every own announcement about the AGM (one request each), and keeps the earliest
one whose book-closure date implies the same D (within one trading day). Announcements posted on behalf of a
subsidiary or parent (代子公司 / 代母公司 ...) are skipped. If nothing matches in that year, the previous ROC
year is tried, for meetings announced in December. Events with no match get no lead.

Raw responses are cached under data/mops_agm_cache/, so the job can be killed and resumed.

Output:
  data/mops_agm.csv  stock_id, D, announce_date, announce_time, agm_date, closure_start, D_implied,
                     first_close, lead, matched, n_candidates
     first_close = first trading day whose close comes after the post (same day if posted before 13:25)
     lead        = trading days from first_close to D; a long entered at the close of D-k is feasible
                   only if lead >= k

Usage:
    python collect_mops_agm.py [--interval 3.0] [--all]     # --all: every AGM event, not only Q5 trades
"""

import argparse
import hashlib
import os
import re
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE = os.path.join(DATA, "mops_agm_cache")
OUT = os.path.join(DATA, "mops_agm.csv")
URL = "https://mopsov.twse.com.tw/mops/web/ajax_t05st01"

SIG_LAG, THR, MIN_VAL20 = 17, 0.167, 2e7      # frozen backtest rule (build_backtest.py)
CLOSE_CUTOFF = "132500"                        # posts after the closing auction opens count from the next day


class MOPS:
    def __init__(self, interval):
        self.interval = interval
        self.last = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def post(self, data):
        key = hashlib.md5(repr(sorted(data.items())).encode()).hexdigest()
        path = os.path.join(CACHE, key + ".html")
        if os.path.exists(path):
            return open(path, encoding="utf-8").read()
        delay = 60
        for _ in range(8):
            wait = self.interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = self.session.post(URL, data=data, timeout=90)
                r.encoding = "utf-8"
                text = r.text
            except Exception as e:
                print(f"    network error: {e}; retry in {delay}s", flush=True)
                time.sleep(delay)
                continue
            if "查詢過於頻繁" in text or "SECURITY REASONS" in text or r.status_code != 200:
                print(f"    throttled (http {r.status_code}); backing off {delay}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 1800)
                self.interval = min(self.interval * 1.5, 15)
                continue
            open(path, "w", encoding="utf-8").write(text)
            return text
        raise RuntimeError(f"gave up on {data}")

    def year_list(self, sid, roc_year):
        return self.post(dict(encodeURIComponent=1, step=1, firstin=1, off=1, keyword4="", code1="", TYPEK2="",
                              checkbtn="", queryName="co_id", inpuType="co_id", TYPEK="all", co_id=sid,
                              year=roc_year, month="", b_date="", e_date=""))

    def detail(self, sid, typek, date, tm, seq, roc_year):
        return self.post(dict(step=2, off=1, firstin=1, TYPEK=typek, co_id=sid, spoke_date=date, spoke_time=tm,
                              seq_no=seq, b_date="", e_date="", year=roc_year, month=""))


ROW = re.compile(r"&nbsp;(\d{2,3}/\d{2}/\d{2})</td><td[^>]*>&nbsp;(\d{2}:\d{2}:\d{2})</td><td[^>]*>"
                 r"<pre[^>]*><font[^>]*>&nbsp;(.*?)</font></pre>.*?seq_no\.value='(\d+)';.*?"
                 r"spoke_time\.value='(\d+)';document\.t05st01_fm\.spoke_date\.value='(\d{8})';.*?"
                 r"TYPEK\.value='(\w+)'", re.S)


def roc(s):
    y, m, d = (int(x) for x in s.split("/"))
    return pd.Timestamp(y + 1911, m, d)


def candidates(html):
    """Own announcements about the AGM (股東常會, or 股東會 that is not an EGM), excluding meeting results and
    posts made for another company."""
    out = []
    for date, tm, subj, seq, stime, sdate, typek in ROW.findall(html):
        subj = re.sub(r"\s+", "", subj)
        agm = "股東常會" in subj or ("股東會" in subj and "臨時" not in subj)
        if not agm or subj.startswith("代") or any(w in subj for w in ("重要決議", "選舉結果", "競業")):
            continue
        out.append(dict(date=sdate, time=stime.zfill(6), subj=subj, seq=seq, typek=typek))
    return out


def parse_detail(html):
    text = re.sub(r"<[^>]+>", " ", html)
    cs = re.search(r"停止過戶起始日期\s*[:：]\s*(\d{2,3}/\d{2}/\d{2})", text)
    agm = re.search(r"股東會召開日期\s*[:：]\s*(\d{2,3}/\d{2}/\d{2})", text)
    return (roc(cs.group(1)) if cs else pd.NaT), (roc(agm.group(1)) if agm else pd.NaT)


def load_events(all_events):
    is_stock = lambda s: s.str.fullmatch(r"[1-9]\d{3}")
    cal = pd.DatetimeIndex(pd.read_csv(os.path.join(DATA, "index.csv"), parse_dates=["date"])
                           .drop_duplicates("date").sort_values("date").date)
    ev = pd.read_csv(os.path.join(DATA, "suspension.csv"), dtype={"stock_id": str},
                     parse_dates=["date", "end_date"]).drop_duplicates()
    if all_events:
        ev = ev[ev.reason.str.contains("股東常會") & is_stock(ev.stock_id)].copy()
        ev["D"] = cal.searchsorted(ev.date.values)
        return ev[ev.D < len(cal)][["stock_id", "D"]].reset_index(drop=True), cal

    px = pd.read_csv(os.path.join(DATA, "prices.csv"), dtype={"stock_id": str}, parse_dates=["date"],
                     usecols=["date", "stock_id", "volume", "value"])
    px = px[is_stock(px.stock_id)].drop_duplicates(["date", "stock_id"])
    mg = pd.read_csv(os.path.join(DATA, "margin.csv"), dtype={"stock_id": str}, parse_dates=["date"],
                     usecols=["date", "stock_id", "short_bal"])
    mg = mg[is_stock(mg.stock_id)].drop_duplicates(["date", "stock_id"])
    stocks = pd.Index(sorted(set(px.stock_id) & set(mg.stock_id)))
    wide = lambda df, c: df.pivot(index="date", columns="stock_id", values=c).reindex(index=cal, columns=stocks)
    SB = wide(mg, "short_bal")
    ADV = (wide(px, "volume") / 1000).rolling(20, min_periods=10).mean()
    VAL20 = wide(px, "value").rolling(20, min_periods=10).mean()
    col = {s: i for i, s in enumerate(stocks)}

    ev = ev[ev.stock_id.isin(stocks)].copy()
    ev["D"] = cal.searchsorted(ev.date.values)
    ev = ev[(ev.D >= SIG_LAG + 1) & (ev.D < len(cal))]
    ev["j"] = ev.stock_id.map(col)
    ev = ev.sort_values(["stock_id", "D"])
    ev = ev[ev.groupby("stock_id").D.diff().fillna(99) >= 10]
    s_ = ev.D.values - SIG_LAG
    sb = SB.values[s_, ev.j.values]
    dtc = sb / ADV.values[s_, ev.j.values]
    keep = (sb > 0) & (dtc >= THR) & (VAL20.values[s_, ev.j.values] >= MIN_VAL20) & ev.reason.str.contains("股東常會").values
    return ev[keep][["stock_id", "D"]].reset_index(drop=True), cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    ev, cal = load_events(args.all)
    print(f"{len(ev)} AGM events to resolve", flush=True)
    cli = MOPS(args.interval)
    rows = []
    for k, (sid, D) in enumerate(ev.itertuples(index=False)):
        Ddate = cal[D]
        best, n_cand = None, 0
        for roc_year in (Ddate.year - 1911, Ddate.year - 1912):
            for c in candidates(cli.year_list(sid, roc_year)):
                if pd.Timestamp(c["date"]) > Ddate:
                    continue
                n_cand += 1
                cs, agm = parse_detail(cli.detail(sid, c["typek"], c["date"], c["time"], c["seq"], roc_year))
                if pd.isna(cs):
                    continue
                Di = cal.searchsorted(cs) - 6
                if abs(Di - D) <= 1 and (best is None or (c["date"], c["time"]) < (best["date"], best["time"])):
                    best = dict(c, cs=cs, agm=agm, Di=Di)
            if best is not None:
                break

        row = dict(stock_id=sid, D=Ddate.date(), matched=best is not None, n_candidates=n_cand)
        if best is not None:
            ad = pd.Timestamp(best["date"])
            fc = cal.searchsorted(ad)
            if fc < len(cal) and cal[fc] == ad and best["time"] >= CLOSE_CUTOFF:
                fc += 1
            row.update(announce_date=ad.date(), announce_time=best["time"], first_close=cal[min(fc, len(cal) - 1)].date(),
                       lead=D - fc, subject=best["subj"],
                       agm_date=best["agm"].date() if pd.notna(best["agm"]) else None,
                       closure_start=best["cs"].date(), D_implied=cal[best["Di"]].date())
        rows.append(row)
        if k % 25 == 0:
            print(f"  [{k + 1}/{len(ev)}] {sid} D={Ddate.date()} -> {row.get('announce_date')} lead={row.get('lead')} "
                  f"matched={row['matched']}", flush=True)
        if k % 100 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows to {OUT} | matched {df.matched.mean():.1%}", flush=True)
    m = df[df.matched]
    print("lead (trading days) for matched events:", m.lead.describe(percentiles=[.05, .1, .25, .5]).round(1).to_dict())
    print("share with lead >= 12 (D-12 close entry feasible):", round((m.lead >= 12).mean(), 3))


if __name__ == "__main__":
    main()
