"""
TWSE SBL borrow-fee collector (歷史借券成交明細, t13sa710).

Every SBL trade done through the TWSE lending system -- fixed-price (定價),
competitive (競價) and negotiated (議借) -- with its annualised fee rate. One
call per calendar month for the whole market. This is the only public record
of what borrowing actually cost per name per day.

Output (appended, checkpointed per month so the job can be killed and resumed):
  data/sbl_fees.csv        date, stock_id, name, trade_type, lots, fee_rate,
                           close, return_date, days, rate_change
  data/_done_sbl_fees.txt  one YYYYMM per finished month

Usage:
    python collect_twse_sbl_fees.py [--interval 4.0]
"""

import argparse
import calendar
import os
from datetime import date

from collect_twse import DATA, appender, num, roc_to_iso
from twse_client import TWSEClient

START = date(2018, 1, 1)
END = date(2026, 9, 30)

FEES = os.path.join(DATA, "sbl_fees.csv")
DONE = os.path.join(DATA, "_done_sbl_fees.txt")

FEE_COLS = ["date", "stock_id", "name", "trade_type", "lots", "fee_rate", "close",
            "return_date", "days", "rate_change"]


def months():
    y, m = END.year, END.month
    while (y, m) >= (START.year, START.month):
        yield y, m
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)


def parse_fees(j):
    out = []
    for r in j.get("data", []) or []:
        # 成交日期 證券代號名稱 交易方式 成交數量(交易單位) 成交費率 成交日收盤價 約定還券日期 約定借券天數 費率異動
        sid, _, name = r[1].strip().partition(" ")
        out.append([roc_to_iso(r[0]), sid, name.strip(), r[2].strip(), num(r[3]), num(r[4]),
                    num(r[5]), roc_to_iso(r[6]) if r[6].strip() else "", r[7], str(r[8]).strip()])
    return out


def run_pass(cli):
    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    todo = [(y, m) for y, m in months() if f"{y}{m:02d}" not in done]
    print(f"{len(todo)} months to pull", flush=True)

    ff, wf = appender(FEES, FEE_COLS)
    fd = open(DONE, "a")
    missing = 0
    for y, m in todo:
        last = calendar.monthrange(y, m)[1]
        j = cli.get_json("https://www.twse.com.tw/SBL/t13sa710",
                         params=dict(response="json", startDate=f"{y}{m:02d}01",
                                     endDate=f"{y}{m:02d}{last:02d}"))
        if not j or j.get("stat") != "OK":
            print(f"{y}-{m:02d}: request failed; will retry next pass", flush=True)
            missing += 1
            continue
        rows = parse_fees(j)
        wf.writerows(rows)
        ff.flush()
        fd.write(f"{y}{m:02d}\n"); fd.flush()
        print(f"{y}-{m:02d}: {len(rows)} trades", flush=True)
    ff.close(); fd.close()
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=4.0)
    args = ap.parse_args()
    cli = TWSEClient(min_interval=args.interval)
    for p in range(3):
        missing = run_pass(cli)
        print(f"pass {p+1}: {missing} months missing", flush=True)
        if missing == 0:
            break


if __name__ == "__main__":
    main()
