"""
TWSE SBL short-sale collector (借券賣出).

One whole-market snapshot per trading day from TWT93U (信用額度總量管制餘額表),
which carries the 借券賣出 flow and balance next to the 融券 one. We need it to
tell whether an SBL short was actually feasible on each event's entry date:
non-zero sbl_sell that day means someone did borrow and short the name.

Trading days come from data/margin.csv (already pulled by collect_twse.py), so
we never waste calls on holidays.

Output (appended, checkpointed per day so the job can be killed and resumed):
  data/sbl.csv        date, stock_id, short_prev, short_sell, short_buy, short_cash,
                      short_bal, short_limit, sbl_prev, sbl_sell, sbl_return,
                      sbl_adjust, sbl_bal, sbl_limit_next, note
  data/_done_sbl.txt  one YYYYMMDD per finished day

Usage:
    python collect_twse_sbl.py [--interval 3.0]
"""

import argparse
import csv
import os

from collect_twse import DATA, MARGIN, appender, num
from twse_client import TWSEClient

SBL = os.path.join(DATA, "sbl.csv")
DONE = os.path.join(DATA, "_done_sbl.txt")

SBL_COLS = ["date", "stock_id", "short_prev", "short_sell", "short_buy", "short_cash",
            "short_bal", "short_limit", "sbl_prev", "sbl_sell", "sbl_return",
            "sbl_adjust", "sbl_bal", "sbl_limit_next", "note"]


def trading_days():
    with open(MARGIN, encoding="utf-8") as f:
        next(f)
        return sorted({line.split(",", 1)[0] for line in f}, reverse=True)


def parse_sbl(j, iso):
    out = []
    for r in j.get("data", []) or []:
        # 代號 名稱 | 融券: 前日 賣出 買進 現券 今日 次一營業日限額 | 借券賣出: 前日 當日賣出 當日還券 當日調整 當日餘額 次一營業日可限額 | 備註
        out.append([iso, r[0].strip()] + [num(x) for x in r[2:14]] +
                   [(r[14] if len(r) > 14 else "").strip()])
    return out


def run_pass(cli):
    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    days = [d for d in trading_days() if d.replace("-", "") not in done]
    print(f"{len(days)} days to pull", flush=True)

    fs, ws = appender(SBL, SBL_COLS)
    fd = open(DONE, "a")
    missing = 0
    for k, iso in enumerate(days):
        ymd = iso.replace("-", "")
        j = cli.get_json("https://www.twse.com.tw/rwd/zh/marginTrading/TWT93U",
                         params=dict(date=ymd, response="json"))
        rows = parse_sbl(j, iso) if j and j.get("stat") == "OK" else []
        if not rows:
            print(f"{ymd}: no data; will retry next pass", flush=True)
            missing += 1
            continue
        ws.writerows(rows)
        fs.flush()
        fd.write(ymd + "\n"); fd.flush()
        if k % 20 == 0:
            print(f"{ymd}: {len(rows)} rows  [{k+1}/{len(days)}]", flush=True)
    fs.close(); fd.close()
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()
    cli = TWSEClient(min_interval=args.interval)
    for p in range(3):
        missing = run_pass(cli)
        print(f"pass {p+1}: {missing} days missing", flush=True)
        if missing == 0:
            break


if __name__ == "__main__":
    main()
