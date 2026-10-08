"""
Polite client for TWSE's public RWD/OpenAPI endpoints.

These are the exchange's own feeds and need no account, unlike FinMind's free
tier which caps out at a few hundred requests an hour. The trade-off is that
TWSE will throttle or drop a client that hammers it, so every call goes through
a minimum-interval gate with exponential backoff on failure.
"""

import time

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.twse.com.tw/",
}

MIN_INTERVAL = 2.0   # seconds between requests; TWSE starts dropping below ~1s


class TWSEClient:
    def __init__(self, min_interval=MIN_INTERVAL, verbose=True):
        self.min_interval = min_interval
        self.last_call = 0.0
        self.n_calls = 0
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _gate(self):
        wait = self.min_interval - (time.time() - self.last_call)
        if wait > 0:
            time.sleep(wait)
        self.last_call = time.time()
        self.n_calls += 1

    def get_json(self, url, params=None, max_retries=5):
        """Return parsed JSON, or None if the endpoint has nothing for this query.

        A None return is normal and expected -- TWSE serves an empty or non-OK
        payload for market holidays, which is how we detect them.
        """
        delay = 5
        for _ in range(max_retries):
            self._gate()
            try:
                r = self.session.get(url, params=params, timeout=60)
            except Exception as e:
                if self.verbose:
                    print(f"    network error: {e}; retry in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue

            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return None          # holidays can come back as empty body

            if r.status_code in (403, 429):
                if self.verbose:
                    print(f"    http {r.status_code} (throttled); backing off {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 600)
                self.min_interval = min(self.min_interval * 1.5, 10.0)
                continue

            if 500 <= r.status_code < 600:
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue

            return None

        return None
