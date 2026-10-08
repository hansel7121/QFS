"""
Step 2 of the Taiwan GIV data build: the stock x day panel.

For every stock in data/universe.csv, pull the full sample of

  TaiwanStockPrice                            -> returns, trade value
  TaiwanStockInstitutionalInvestorsBuySell    -> per-stock institutional flows
  TaiwanStockShareholding (optional)          -> shares issued, foreign ownership

FinMind's free tier needs one request per stock per dataset, so this is a long
job. It checkpoints after every stock into data/_progress.json and appends to
the CSVs, so it can be killed and restarted without losing or duplicating work.

Usage:
    python collect_panel.py                 # prices + institutional flows
    python collect_panel.py --shareholding  # also the weekly shareholding panel
"""

import argparse
import json
import os

import pandas as pd

from finmind_client import FinMindClient

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

START = "2020-03-02"      # matches the sample already pulled for the Gotobi pitch
END = "2026-09-23"

UNIVERSE = os.path.join(DATA, "universe.csv")
PROGRESS = os.path.join(DATA, "_progress.json")

JOBS = {
    "prices": ("TaiwanStockPrice", os.path.join(DATA, "prices.csv")),
    "flows": ("TaiwanStockInstitutionalInvestorsBuySell", os.path.join(DATA, "inst_flows.csv")),
    "shareholding": ("TaiwanStockShareholding", os.path.join(DATA, "shareholding.csv")),
}


def load_progress():
    if os.path.exists(PROGRESS):
        return json.load(open(PROGRESS))
    return {}


def save_progress(p):
    json.dump(p, open(PROGRESS, "w"), indent=1)


def append_rows(path, rows):
    """Append to CSV, writing the header only when the file is created."""
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_job(client, job, stock_ids, progress):
    dataset, out_csv = JOBS[job]
    done = set(progress.get(job, []))
    todo = [s for s in stock_ids if s not in done]
    print(f"\n[{job}] {len(done)} done, {len(todo)} to go -> {os.path.basename(out_csv)}")

    for n, sid in enumerate(todo, 1):
        rows = client.get(dataset, data_id=sid, start_date=START, end_date=END)
        if rows:
            append_rows(out_csv, rows)
        done.add(sid)
        progress[job] = sorted(done)
        save_progress(progress)
        if n % 10 == 0 or n == len(todo):
            print(f"  [{job}] {n}/{len(todo)}  last={sid} rows={len(rows)}  "
                  f"({client.n_calls} calls this run)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shareholding", action="store_true",
                    help="also pull the weekly shareholding panel (+1 call per stock)")
    args = ap.parse_args()

    uni = pd.read_csv(UNIVERSE, dtype={"stock_id": str})
    stock_ids = uni.stock_id.tolist()
    print(f"universe: {len(stock_ids)} stocks, sample {START} .. {END}")

    progress = load_progress()
    client = FinMindClient()

    jobs = ["prices", "flows"] + (["shareholding"] if args.shareholding else [])
    for job in jobs:
        run_job(client, job, stock_ids, progress)

    print(f"\ndone. {client.n_calls} API calls this run.")
    for job in jobs:
        path = JOBS[job][1]
        if os.path.exists(path):
            n = sum(1 for _ in open(path)) - 1
            print(f"  {os.path.basename(path):20s} {n:>10,} rows")


if __name__ == "__main__":
    main()
