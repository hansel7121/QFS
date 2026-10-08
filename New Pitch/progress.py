"""
Live progress for the TWSE pull. Safe to run any time, including while
collect_twse.py is mid-flight -- it only reads the checkpoint files.

    python progress.py          one snapshot
    python progress.py --watch  refresh until the pull finishes
"""

import argparse
import os
import time
from datetime import datetime, timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

START = "2020-03-02"
END = "2026-09-23"

DONE_T86 = os.path.join(DATA, "_done_t86.txt")
DONE_PX = os.path.join(DATA, "_done_px.txt")
FLOWS = os.path.join(DATA, "inst_flows.csv")
PRICES = os.path.join(DATA, "prices.csv")


def n_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def bar(frac, width=44):
    filled = int(round(frac * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def human(seconds):
    if seconds is None or seconds < 0:
        return "?"
    return str(timedelta(seconds=int(seconds)))


def is_running():
    return os.system("pgrep -f collect_twse.py > /dev/null 2>&1") == 0


def snapshot(total_days, started_at=None, started_done=None):
    done_t86 = n_lines(DONE_T86)
    done_px = n_lines(DONE_PX)
    done = min(done_t86, done_px)
    frac = done / total_days if total_days else 0.0

    last = "-"
    if os.path.exists(DONE_T86) and done_t86:
        with open(DONE_T86) as f:
            tail = f.read().split()
        if tail:
            d = tail[-1]
            last = f"{d[:4]}-{d[4:6]}-{d[6:]}"

    # Rate from this process's own observation window, so it reflects the
    # current pace rather than an average polluted by earlier restarts.
    eta = None
    rate = None
    if started_at is not None and started_done is not None:
        elapsed = time.time() - started_at
        progressed = done - started_done
        if progressed > 0 and elapsed > 0:
            rate = progressed / elapsed
            eta = (total_days - done) / rate

    print(f"\r{bar(frac)} {frac:6.1%}  {done:>5}/{total_days} days", end="")
    print(f"  last {last}", end="")
    if rate:
        print(f"  {rate*60:5.1f} days/min  ETA {human(eta)}", end="")
    print(f"  [{'running' if is_running() else 'STOPPED'}]   ", end="", flush=True)
    return done, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=float, default=10.0, help="refresh seconds")
    args = ap.parse_args()

    total_days = len(pd.bdate_range(START, END))
    started_at = time.time()
    started_done = min(n_lines(DONE_T86), n_lines(DONE_PX))

    if not args.watch:
        snapshot(total_days, started_at, started_done)
        print()
        print(f"  rows so far   inst_flows {max(n_lines(FLOWS)-1,0):>10,}"
              f"   prices {max(n_lines(PRICES)-1,0):>10,}")
        print(f"  checkpoints   {DONE_T86}")
        return

    try:
        while True:
            done, frac = snapshot(total_days, started_at, started_done)
            if done >= total_days:
                print("\ndone.")
                return
            if not is_running():
                print("\ncollector is not running -- restart with:")
                print("  python collect_twse.py")
                return
            time.sleep(args.every)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
