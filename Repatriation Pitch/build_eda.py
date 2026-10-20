"""Generate eda.ipynb: Japanese fiscal year-end (March) / half-year (September) repatriation and USD/JPY."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Repatriation Pitch EDA: does JPY strengthen into Japan's March / September book closings?

**The mechanism.** Japanese corporates and lifers repatriate foreign earnings and close FX positions before the fiscal
year-end (March 31) and half-year (September 30), buying JPY, so USD/JPY should fall in the second half of March and
September.

**Data.** Yahoo `JPY=X` daily, 1996–. Monthly and second-half-of-month USD/JPY returns, March/September vs other
months, 1997–2011 vs 2012–.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from scipy import stats
d = pd.read_csv("data/fx_daily.csv", parse_dates=["Date"], index_col="Date")
x = np.log(d["JPY=X"].dropna()); cal = x.index
rows = []
for y in range(1997, 2027):
    for m in range(1, 13):
        days = cal[(cal.year == y) & (cal.month == m)]
        if len(days) < 15: continue
        i0 = cal.get_loc(days[0]) - 1; i1 = cal.get_loc(days[-1]); imid = cal.get_loc(days[len(days) // 2])
        rows.append({"y": y, "m": m, "month": (x.iloc[i1] - x.iloc[i0]) * 1e4, "second_half": (x.iloc[i1] - x.iloc[imid]) * 1e4})
D = pd.DataFrame(rows)
for per, mk in [("1997–2011", D.y <= 2011), ("2012–", D.y >= 2012)]:
    for mm in [3, 9]:
        a = D[mk & (D.m == mm)]; b = D[mk & ~D.m.isin([3, 9])]
        print(per, "month", mm, {c: f"{a[c].mean():+.0f} vs {b[c].mean():+.0f} (t {stats.ttest_ind(a[c], b[c], equal_var=False)[0]:+.2f})" for c in ["month", "second_half"]}, "n", len(a))
""")
md(r"""
**Interpretation.** USD/JPY does not fall into either book closing. If anything it *rises* in the second half of March
(+108bp, t 2.0 in 1997–2011), the opposite sign, and nothing survives after 2012.

**Verdict: dead.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
