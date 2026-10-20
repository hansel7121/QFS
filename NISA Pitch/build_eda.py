"""Generate eda.ipynb: month-start USD/JPY buying from new-NISA foreign-equity fund flows (2024-), vs 2022-23."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# NISA Pitch EDA: does NISA auto-investment buy USD/JPY at the start of each month?

**The mechanism.** Japan's new NISA (January 2024) drove large monthly auto-investment into foreign-equity
investment trusts (*toshin*), commonly scheduled on the first business days of the month. The trusts buy USD to fund
purchases, often at the Tokyo 09:55 fix. If so, USD/JPY should rise on the first business days of each month from
2024, but not in 2022–23.

**Data.** USD/JPY 1-minute bars (Tokyo time) from `Gotobi/data/usdjpy_1m.csv` (Twelve Data, 2022-04 →). Japanese
holidays removed.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, holidays
from scipy import stats
x = pd.read_csv("../Gotobi/data/usdjpy_1m.csv", parse_dates=["datetime"]).sort_values("datetime")
x["d"] = x.datetime.dt.normalize(); x["hm"] = x.datetime.dt.strftime("%H:%M")
c = x.pivot_table(index="d", columns="hm", values="close")
jp = holidays.Japan(years=range(2022, 2027))
c = c[(c.index.dayofweek < 5) & np.array([d not in jp for d in c.index])]
r = lambda a, b: np.log(c[b] / c[a]) * 1e4
W = {"08:00→09:55 (into fix)": r("08:00", "09:54"), "09:55→10:30 (after fix)": r("09:54", "10:30"),
     "08:00→15:00 (Tokyo day)": r("08:00", "14:59"), "15:00→next 08:00 (overnight)": np.log(c["08:00"].shift(-1) / c["14:59"]) * 1e4}
D = pd.DataFrame(W); D["bd"] = pd.Series(c.index, index=c.index).groupby(c.index.to_period("M")).cumcount() + 1
D["era"] = np.where(D.index >= "2024-01-01", "2024– (new NISA)", "2022–23")
rows = []
for w in W:
    for era in ["2022–23", "2024– (new NISA)"]:
        d = D[D.era == era]; a, b = d[d.bd <= 3][w].dropna(), d[d.bd > 3][w].dropna(); t, _ = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"window": w, "era": era, "BD1-3 bp": a.mean(), "other bp": b.mean(), "t": t, "n BD1-3": len(a)})
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nmean by Japanese business day of the month (2024–), bp:")
print(D[D.era == "2024– (new NISA)"].groupby(D.bd.clip(upper=8))[list(W)].mean().round(1).T.to_string())
""")
md(r"""
**Interpretation.**
- **No month-start effect in any window.** From 2024, business days 1–3 differ from other days by at most a few bp
  (|t| < 1 in the Tokyo windows). The overnight window is negative (t ≈ −1.9), the opposite sign to USD buying.
- The pattern is the same in 2022–23, before new NISA. The flow, if it exists, is spread out or hedged, not
  concentrated at the month start.

**Verdict: dead on arrival.** Not tradable.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
