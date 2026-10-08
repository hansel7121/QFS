"""
Thin FinMind v4 client with a sliding-window rate limiter and retry/backoff.

The free tier requires data_id on every request (no bulk "all stocks on date"
queries) and caps requests per hour, so every pull in this project is one call
per stock. The limiter below keeps us under the cap without having to babysit
the job; on a 402/429 it backs off and retries rather than losing progress.

A token is optional. If data/finmind.token exists (gitignored) it raises the
hourly cap; without one the free anonymous cap applies.
"""

import os
import time
from collections import deque

import requests

URL = "https://api.finmindtrade.com/api/v4/data"
HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(HERE, "data", "finmind.token")

# Anonymous free tier is documented at 600 requests/hour. Stay under it.
DEFAULT_PER_HOUR = 500


class FinMindClient:
    def __init__(self, per_hour=DEFAULT_PER_HOUR, token=None, verbose=True):
        self.per_hour = per_hour
        self.calls = deque()
        self.n_calls = 0
        self.verbose = verbose
        if token is None and os.path.exists(TOKEN_FILE):
            token = open(TOKEN_FILE).read().strip()
        self.token = token

    def _throttle(self):
        now = time.time()
        while self.calls and now - self.calls[0] >= 3600:
            self.calls.popleft()
        if len(self.calls) >= self.per_hour:
            sleep_for = 3600 - (now - self.calls[0]) + 1
            if self.verbose:
                print(f"    [rate limit] sleeping {sleep_for/60:.1f} min")
            time.sleep(max(sleep_for, 0))
        self.calls.append(time.time())
        self.n_calls += 1

    def get(self, dataset, max_retries=6, **params):
        """Return the 'data' list, or [] if the API has nothing for this query."""
        params = dict(dataset=dataset, **params)
        if self.token:
            params["token"] = self.token

        delay = 5
        for attempt in range(max_retries):
            self._throttle()
            try:
                r = requests.get(URL, params=params, timeout=60)
            except Exception as e:
                if self.verbose:
                    print(f"    network error: {e}; retry in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue

            if r.status_code in (402, 429, 500, 502, 503, 504):
                if self.verbose:
                    print(f"    http {r.status_code}; backing off {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 900)
                continue

            try:
                j = r.json()
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue

            if j.get("msg") == "success" or "data" in j:
                return j.get("data", [])

            # A hard "upgrade your level" message will never succeed on retry.
            msg = str(j.get("msg", ""))
            if "level" in msg.lower():
                raise RuntimeError(f"{dataset}: {msg}")
            if self.verbose:
                print(f"    unexpected response: {msg[:120]}; retry in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 300)

        return []
