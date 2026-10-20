"""Scheduled FOMC meeting dates 1994-2026 from federalreserve.gov (historical pages + current calendar).

The announcement is on the meeting's last day (14:15 ET until 2011 roughly, 14:00 since). Unscheduled meetings and
conference calls are excluded: their timing was not known in advance.

Output: data/fomc.csv  date (announcement day), year
"""

import re
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
UA = {"User-Agent": "Mozilla/5.0"}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def last_day(year, text):
    """'April 30-May 1' -> May 1; 'June 22-23' -> June 23; 'March 16' -> March 16."""
    m = re.match(rf"({MONTHS})\s+(\d+)(?:-(?:({MONTHS})\s+)?(\d+))?", text)
    mon = m.group(3) or m.group(1)
    day = m.group(4) or m.group(2)
    return pd.Timestamp(f"{mon} {day} {year}")


def main():
    dates = []
    for y in range(1994, 2021):
        h = requests.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm", headers=UA, timeout=60).text
        for t in re.findall(rf"((?:{MONTHS})\s+\d+(?:-(?:(?:{MONTHS})\s+)?\d+)?)\s+(?:FOMC\s+)?Meeting(?!s)([^<]{{0,40}})", h):
            if "unscheduled" in t[1].lower():
                continue
            dates.append(last_day(y, t[0]))
    h = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=UA, timeout=60).text
    for block in re.split(r"<h4><a[^>]*>", h)[1:]:
        ym = re.match(r"(\d{4}) FOMC Meetings", block)
        if not ym:
            continue
        y = int(ym.group(1))
        for mon, d in re.findall(rf"fomc-meeting__month[^>]*><strong>((?:{MONTHS})(?:/(?:{MONTHS}))?)</strong>.*?fomc-meeting__date[^>]*>([^<]*)<", block, re.S):
            if "unscheduled" in d.lower() or "notation" in d.lower():
                continue
            m2 = mon.split("/")[-1]
            dd = re.findall(r"\d+", d)
            if dd:
                dates.append(pd.Timestamp(f"{m2} {dd[-1]} {y}"))
    out = pd.DataFrame({"date": sorted(set(dates))})
    out["year"] = out.date.dt.year
    out.to_csv(DATA / "fomc.csv", index=False)
    print(len(out), "meetings;", out.groupby("year").size().to_dict())


if __name__ == "__main__":
    main()
