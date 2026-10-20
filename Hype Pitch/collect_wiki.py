"""Daily English-Wikipedia pageviews (human traffic, agent=user) for each theme page in data/themes.csv, split by
access method (desktop, mobile-web, mobile-app), 2015-07-01 (start of the Wikimedia pageviews API) to yesterday.

Also pulls the views of every title that redirects to each page (all access methods combined). When an article is
renamed, its traffic before the move is recorded under the old title, which then becomes a redirect; adding the
redirects' views keeps the series continuous across renames (e.g. "Rare earth element" -> "Rare-earth element" in
June 2017, "Robinhood (company)" -> "Robinhood Markets" in July 2021).

Output: data/wiki_views.csv      date, page, desktop, mobile_web, mobile_app
        data/wiki_redirects.csv  date, page, redirect, views
"""

import time
import urllib.parse as up
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
H = {"User-Agent": "QFS research hanselliu206@gmail.com"}
API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/{acc}/user/{page}/daily/20150701/{end}"


def pull(page, acc, end):
    url = API.format(acc=acc, page=up.quote(page.replace(" ", "_"), safe=""), end=end)
    for k in range(4):
        r = requests.get(url, headers=H, timeout=60)
        if r.status_code == 200:
            return pd.Series({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in r.json()["items"]}, name=acc)
        if r.status_code == 404:
            return pd.Series(dtype=float, name=acc)
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"{page} {acc}: {r.status_code}")


def redirects(page):
    out, cont = [], {}
    while True:
        r = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "titles": page, "prop": "redirects", "rdlimit": "max",
                                                                     "rdnamespace": 0, "format": "json", **cont}, headers=H, timeout=60).json()
        for pg in r["query"]["pages"].values():
            out += [x["title"] for x in pg.get("redirects", [])]
        if "continue" not in r:
            return out
        cont = r["continue"]


def main_redirects():
    themes = pd.read_csv(DATA / "themes.csv")
    pages = sorted({p for s in themes.pages for p in s.split(";")})
    end = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    out = []
    for p in pages:
        rd = redirects(p)
        for t in rd:
            s = pull(t, "all-access", end)
            if len(s):
                out.append(pd.DataFrame({"date": s.index, "page": p, "redirect": t, "views": s.values}))
            time.sleep(0.1)
        print(p, len(rd), "redirects", flush=True)
    pd.concat(out).to_csv(DATA / "wiki_redirects.csv", index=False)


def main():
    themes = pd.read_csv(DATA / "themes.csv")
    pages = sorted({p for s in themes.pages for p in s.split(";")})
    end = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    out = []
    for p in pages:
        s = pd.concat([pull(p, a, end) for a in ["desktop", "mobile-web", "mobile-app"]], axis=1)
        s.columns = ["desktop", "mobile_web", "mobile_app"]
        s.index.name = "date"; s["page"] = p
        out.append(s.reset_index())
        print(p, len(s), int(s[["desktop", "mobile_web", "mobile_app"]].sum(1).median()), flush=True)
        time.sleep(0.3)
    pd.concat(out).to_csv(DATA / "wiki_views.csv", index=False)


if __name__ == "__main__":
    main()
    main_redirects()
