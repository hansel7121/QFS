"""
TWSE bulk collector for the forced short-covering pitch.

One pass over every weekday in [START, END], newest first, pulling two
whole-market snapshots per day straight from the exchange:

  MI_INDEX  type=ALLBUT0999    -> every listed security's OHLCV + TAIEX levels
  MI_MARGN  selectType=ALL     -> every security's margin / short balances

plus the ex-rights / ex-dividend reference prices (TWT49U), one call per year,
which we need to build dividend-adjusted returns -- half of the events in this
pitch ARE ex-dividend dates, so raw close-to-close returns would be garbage.

Output (appended, checkpointed per day so the job can be killed and resumed):
  data/prices.csv    date, stock_id, name, volume, value, open, high, low, close, chg
  data/margin.csv    date, stock_id, margin_buy, ..., short_bal, short_limit, offset, note
  data/index.csv     date, taiex, taiex_tr
  data/exdiv.csv     date, stock_id, close_before, ref_price, div_value, kind
  data/_done_twse.txt  one YYYYMMDD per finished (or holiday) day

Usage:
    python collect_twse.py [--interval 3.0]
"""

import argparse
import csv
import os
import re
from datetime import date, timedelta

from twse_client import TWSEClient

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

START = date(2018, 1, 1)
END = date(2026, 9, 23)

PRICES = os.path.join(DATA, "prices.csv")
MARGIN = os.path.join(DATA, "margin.csv")
INDEX = os.path.join(DATA, "index.csv")
EXDIV = os.path.join(DATA, "exdiv.csv")
DONE = os.path.join(DATA, "_done_twse.txt")
FINISHED = os.path.join(DATA, "_twse_finished.flag")

PRICE_COLS = ["date", "stock_id", "name", "volume", "value", "open", "high", "low", "close", "chg"]
MARGIN_COLS = ["date", "stock_id", "margin_buy", "margin_sell", "margin_repay", "margin_prev",
               "margin_bal", "margin_limit", "short_sell", "short_buy", "short_repay",
               "short_prev", "short_bal", "short_limit", "offset", "note"]
INDEX_COLS = ["date", "taiex", "taiex_tr"]
EXDIV_COLS = ["date", "stock_id", "close_before", "ref_price", "div_value", "kind"]


def num(s):
    s = str(s).replace(",", "").strip()
    if s in ("", "--", "---", "X", "除權息", "除息", "除權"):
        return ""
    try:
        return float(s)
    except ValueError:
        return ""


def roc_to_iso(s):
    """'113年01月04日' -> '2024-01-04'."""
    m = re.match(r"(\d+)年(\d+)月(\d+)日", s)
    y, mo, d = (int(x) for x in m.groups())
    return f"{y + 1911:04d}-{mo:02d}-{d:02d}"


def appender(path, cols):
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(cols)
    return f, w


def find_table(j, needle):
    for t in j.get("tables", []):
        if needle in (t.get("title") or ""):
            return t
    return None


def parse_prices(j, iso):
    t = find_table(j, "每日收盤行情")
    if t is None:
        return []
    rows = []
    for r in t["data"]:
        sign = -1 if "-" in re.sub(r"<[^>]+>", "", r[9]) else 1
        chg = num(r[10])
        rows.append([iso, r[0].strip(), r[1].strip(), num(r[2]), num(r[4]), num(r[5]),
                     num(r[6]), num(r[7]), num(r[8]), "" if chg == "" else sign * chg])
    return rows


def parse_index(j, iso):
    px = tr = ""
    for t in j.get("tables", []):
        for r in t.get("data", []):
            if r[0] == "發行量加權股價指數":
                px = num(r[1])
            elif r[0] == "發行量加權股價報酬指數":
                tr = num(r[1])
    return [iso, px, tr]


def parse_margin(j, iso):
    t = find_table(j, "融資融券彙總")
    if t is None:
        return []
    out = []
    for r in t["data"]:
        # 代號 名稱 | 融資: 買進 賣出 現金償還 前日 今日 限額 | 融券: 買進 賣出 現券償還 前日 今日 限額 | 資券互抵 註記
        out.append([iso, r[0].strip(), num(r[2]), num(r[3]), num(r[4]), num(r[5]), num(r[6]),
                    num(r[7]),
                    num(r[9]),   # 融券 賣出 = new shorts
                    num(r[8]),   # 融券 買進 = short covering
                    num(r[10]), num(r[11]), num(r[12]), num(r[13]), num(r[14]),
                    (r[15] if len(r) > 15 else "").strip()])
    return out


def pull_exdiv(cli):
    if os.path.exists(EXDIV):
        return
    f, w = appender(EXDIV, EXDIV_COLS)
    for y in range(START.year, END.year + 1):
        j = cli.get_json("https://www.twse.com.tw/rwd/zh/exRight/TWT49U",
                         params=dict(startDate=f"{y}0101", endDate=f"{y}1231", response="json"))
        n = 0
        for r in (j or {}).get("data", []) or []:
            w.writerow([roc_to_iso(r[0]), r[1].strip(), num(r[3]), num(r[4]), num(r[5]), r[6].strip()])
            n += 1
        f.flush()
        print(f"exdiv {y}: {n} rows", flush=True)
    f.close()


def run_pass(cli):
    """Pull every weekday not yet in the done file. Returns how many days are still missing."""
    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    days = []
    d = END
    while d >= START:
        if d.weekday() < 5 and d.strftime("%Y%m%d") not in done:
            days.append(d)
        d -= timedelta(days=1)
    print(f"{len(days)} days to pull", flush=True)

    fp, wp = appender(PRICES, PRICE_COLS)
    fm, wm = appender(MARGIN, MARGIN_COLS)
    fi, wi = appender(INDEX, INDEX_COLS)
    fd = open(DONE, "a")
    missing = 0

    for k, d in enumerate(days):
        ymd, iso = d.strftime("%Y%m%d"), d.isoformat()
        jp = cli.get_json("https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
                          params=dict(date=ymd, type="ALLBUT0999", response="json"))
        if jp is None:                                  # network / throttle failure, not a holiday
            print(f"{ymd}: request failed; will retry next pass", flush=True)
            missing += 1
            continue
        if jp.get("stat") != "OK":
            fd.write(ymd + "\n"); fd.flush()          # holiday: TWSE answers with a 'no data' message
            continue
        jm = cli.get_json("https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
                          params=dict(date=ymd, selectType="ALL", response="json"))
        prices = parse_prices(jp, iso)
        margin = parse_margin(jm or {}, iso)
        if not prices or not margin:
            print(f"{ymd}: incomplete (prices={len(prices)}, margin={len(margin)}); will retry next pass",
                  flush=True)
            missing += 1
            continue
        wp.writerows(prices)
        wm.writerows(margin)
        wi.writerow(parse_index(jp, iso))
        for f in (fp, fm, fi):
            f.flush()
        fd.write(ymd + "\n"); fd.flush()
        if k % 20 == 0:
            print(f"{ymd}: {len(prices)} px, {len(margin)} margin  [{k+1}/{len(days)}]", flush=True)

    for f in (fp, fm, fi, fd):
        f.close()
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    cli = TWSEClient(min_interval=args.interval)
    pull_exdiv(cli)
    for _ in range(3):
        if run_pass(cli) == 0:
            break

    open(FINISHED, "w").write("done\n")
    print("finished", flush=True)


if __name__ == "__main__":
    main()
