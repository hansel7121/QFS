import os, time, json, math, datetime as dt
from collections import deque
import requests
import pandas as pd

API_KEY    = open("twelvedata.key").read().strip()   # key lives in gitignored file next to notebook
SYMBOL     = "USD/TWD"
INTERVAL   = "1min"
TZ         = "Asia/Taipei"          # all datetimes below are Taipei local
OUTPUTSIZE = 5000                   # max per call on free tier
FLOOR_DATE = pd.Timestamp("2020-04-01")   # stop here even if API still has data

PER_MIN = 8
PER_DAY = 800

OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "usdtwd_1m.csv")
os.makedirs(OUT_DIR, exist_ok=True)

URL = "https://api.twelvedata.com/time_series"

# ---------- cell ----------
class RateLimiter:
    """Sliding-window limiter: never more than PER_MIN calls in any 60s window."""
    def __init__(self, per_min):
        self.per_min = per_min
        self.calls = deque()
        self.total = 0

    def wait(self):
        now = time.time()
        while self.calls and now - self.calls[0] >= 60:
            self.calls.popleft()
        if len(self.calls) >= self.per_min:
            sleep_for = 60 - (now - self.calls[0]) + 0.2
            time.sleep(max(sleep_for, 0))
        self.calls.append(time.time())
        self.total += 1

    def used_this_minute(self):
        now = time.time()
        return sum(1 for t in self.calls if now - t < 60)


def fetch_page(end_date, limiter):
    """One API call. Returns (df_newest_first, status) where status in {'ok','nodata','daily_limit'}."""
    params = dict(symbol=SYMBOL, interval=INTERVAL, outputsize=OUTPUTSIZE,
                  timezone=TZ, end_date=end_date.strftime("%Y-%m-%d %H:%M:%S"),
                  apikey=API_KEY)
    while True:
        limiter.wait()
        try:
            r = requests.get(URL, params=params, timeout=30).json()
        except Exception as e:
            print(f"    network error: {e}; retrying in 5s")
            time.sleep(5)
            continue

        if r.get("status") == "error":
            msg = r.get("message", "")
            code = r.get("code")
            # both minute and day limits come back as code 429; check the day one FIRST
            # ("You have run out of API credits for the day. ...")
            if code == 429 and "for the day" in msg:
                return None, "daily_limit"
            if code == 429 or "credits for the current minute" in msg:
                # API minute is wall-clock; sleep to next minute boundary
                sleep_for = 60 - (time.time() % 60) + 0.5
                print(f"    minute limit hit; sleeping {sleep_for:.0f}s")
                time.sleep(sleep_for)
                continue
            if code in (400, 404) or "not found" in msg.lower():
                print(f"    API {code}: {msg}")        # a bad symbol/param also lands here
                return None, "nodata"
            raise RuntimeError(f"API error {code}: {msg}")

        vals = r.get("values", [])
        if not vals:
            return None, "nodata"
        df = pd.DataFrame(vals)
        df["datetime"] = pd.to_datetime(df["datetime"])
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)
        return df[["datetime", "open", "high", "low", "close"]], "ok"


def fmt_td(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"

# ---------- cell ----------
# ---- resume point ----
if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 0:
    existing_min = pd.to_datetime(pd.read_csv(OUT_CSV, usecols=["datetime"])["datetime"]).min()
    end_date = existing_min - pd.Timedelta(minutes=1)
    print(f"resuming: CSV oldest = {existing_min}, continuing from {end_date}")
else:
    end_date = pd.Timestamp.now(tz=TZ).tz_localize(None) + pd.Timedelta(days=1)
    print(f"fresh pull, starting from {end_date}")

limiter = RateLimiter(PER_MIN)
t0 = time.time()
n_calls = 0
n_bars = 0
first_oldest = None
day_span_hist = deque(maxlen=20)      # calendar days covered per call (recent avg)

print(f"{'call':>4} {'min':>3} {'day':>3} {'bars':>5} {'oldest reached':>19} {'days/call':>9} {'bars/s':>6} {'elapsed':>8} {'ETA(pace)':>10} {'ETA(8/min)':>10}")

while True:
    if end_date < FLOOR_DATE:
        print("reached FLOOR_DATE; done")
        break
    if limiter.total >= PER_DAY:
        print("hit 800/day cap; re-run notebook tomorrow to resume")
        break

    page, status = fetch_page(end_date, limiter)
    n_calls += 1

    if status == "daily_limit":
        print("API says daily credits exhausted; re-run tomorrow to resume")
        break
    if status == "nodata":
        print(f"no data before {end_date}; API history exhausted; done")
        break

    # checkpoint
    page.to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0, index=False)

    newest, oldest = page["datetime"].max(), page["datetime"].min()
    if first_oldest is None:
        first_oldest = newest
    n_bars += len(page)
    span_days = (end_date - oldest).total_seconds() / 86400
    day_span_hist.append(span_days)
    avg_span = sum(day_span_hist) / len(day_span_hist)

    elapsed = time.time() - t0
    remaining_days = max((oldest - FLOOR_DATE).total_seconds() / 86400, 0)
    remaining_calls = remaining_days / avg_span if avg_span > 0 else float("inf")
    eta_pace = remaining_calls * (elapsed / n_calls)
    eta_cap  = remaining_calls * (60 / PER_MIN)

    print(f"{n_calls:>4} {limiter.used_this_minute():>3} {limiter.total:>3} {len(page):>5} "
          f"{str(oldest):>19} {span_days:>9.2f} {n_bars/elapsed:>6.0f} {fmt_td(elapsed):>8} "
          f"{fmt_td(eta_pace):>10} {fmt_td(eta_cap):>10}")

    end_date = oldest - pd.Timedelta(minutes=1)

print(f"\nfinished: {n_calls} calls, {n_bars:,} bars, {fmt_td(time.time()-t0)} elapsed, oldest = {end_date + pd.Timedelta(minutes=1)}")
print("NOTE: ETA(8/min) assumes API history goes all the way to FLOOR_DATE; real stop is when API returns no data (~2020), so actual finish is earlier.")