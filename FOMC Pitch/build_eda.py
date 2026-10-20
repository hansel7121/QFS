"""Generate eda.ipynb: the pre-FOMC announcement drift (Lucca & Moench 2015), after publication."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# FOMC Pitch EDA: is the pre-FOMC drift still there?

**The anomaly.** Lucca & Moench, *"The Pre-FOMC Announcement Drift"* (**JF** 2015): from 1994 to 2011 the S&P 500
earned about half of its excess return in the 24 hours before scheduled FOMC announcements. Meeting dates are
published a year ahead. Later work reports it faded after publication.

**Questions.** (1) FOMC-day return (close to close, which contains the 24h pre-announcement window plus the reaction)
vs other days, by era: 1994–2011 (paper sample), 2012–15, 2016–. (2) Since 2022, with 5-minute SPY bars (from
`LETF Pitch/data/intraday/SPY.csv`): previous close → 14:00 (pre-announcement) vs 14:00 → close (reaction).

**Data.** `fomc.csv` (scheduled meetings scraped from federalreserve.gov; `collect_fomc.py`; a few meetings per year
in 2017/2018/2023/2024 were not parsed, which only shrinks the sample), SPY daily (`TOM Pitch/data/etfs.csv`).
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
px = pd.read_csv("../TOM Pitch/data/etfs.csv", parse_dates=["date"])
adj = px.pivot(index="date", columns="ticker", values="adj_close").SPY; raw = px.pivot(index="date", columns="ticker", values="close").SPY
r = adj.pct_change() * 1e4; r = r[r.index >= "1994-01-01"]
f = pd.read_csv("data/fomc.csv", parse_dates=["date"]); fd = set(f.date[f.date <= r.index[-1]])
isf = r.index.isin(fd); prev = np.r_[isf[1:], False]
rows = []
for per, m in [("1994–2011 (paper)", r.index < "2012"), ("2012–2015", (r.index >= "2012") & (r.index < "2016")), ("2016–", r.index >= "2016")]:
    a, b, c = r[m & isf], r[m & ~isf & ~prev], r[m & prev]
    t, _ = stats.ttest_ind(a, b, equal_var=False)
    rows.append({"era": per, "FOMC day bp": a.mean(), "n": len(a), "other days bp": b.mean(), "t": t, "day before bp": c.mean()})
print(pd.DataFrame(rows).round(2).to_string(index=False))

x = pd.read_csv("../LETF Pitch/data/intraday/SPY.csv", parse_dates=["datetime"])
x["d"] = x.datetime.dt.normalize(); x["hm"] = x.datetime.dt.strftime("%H:%M")
c = x.pivot_table(index="d", columns="hm", values="close")
pre = np.log(c["13:55"] / raw.shift(1).reindex(c.index)) * 1e4
post = np.log(raw.reindex(c.index) / c["13:55"]) * 1e4
isf2 = c.index.isin(fd)
for nm, v in [("prev close -> 14:00 (pre-announcement)", pre), ("14:00 -> close (reaction)", post)]:
    a, b = v[isf2].dropna(), v[~isf2].dropna(); t, _ = stats.ttest_ind(a, b, equal_var=False)
    print(f"2022–: {nm}: FOMC {a.mean():+.1f}bp (n {len(a)}) vs other days {b.mean():+.1f}bp, t {t:+.2f}")
""")
md(r"""
**Interpretation.**
- **The paper sample replicates:** FOMC days +34bp vs +2bp (t = 3.0) in 1994–2011, and still +31bp (t = 1.6) in 2012–15.
- **From 2016 it is gone:** +1bp on FOMC days vs +6bp otherwise (t = −0.4).
- **Since 2022 (intraday):** the pre-announcement window is +20bp vs +5bp, but with t = 1.5 on 32 meetings. The
  14:00 → close reaction averages −41bp (t = −2.3), a "sell the news" pattern from the hiking cycle. It is
  direction-of-news risk, not a pre-known flow, and 32 events is too few.

**Verdict: dead after publication.** Not tradable.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
