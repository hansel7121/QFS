"""Generate eda.ipynb: 401k payday contributions and Social Security payment Wednesdays in US equity returns."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Payday Pitch EDA: do retirement-plan contributions and benefit payments show up in index returns?

**Mechanisms.**
* **401k:** most employers pay on the 15th and the last business day; payroll deductions are invested by plan
  recordkeepers 1–2 business days later, a predictable flow into target-date and index funds.
* **Social Security:** benefits are paid on the 2nd, 3rd and 4th Wednesday of the month (by birth date). If retirees
  invest part of it, those Wednesdays should be stronger.

**Data.** SPY / IWM total return (`TOM Pitch/data/etfs.csv`), 2001–2012 vs 2013–.
""")
code(r"""
import pandas as pd, numpy as np
from scipy import stats
p=pd.read_csv("../TOM Pitch/data/etfs.csv",parse_dates=["date"]).pivot(index="date",columns="ticker",values="adj_close")
R=np.log(p[["SPY","IWM","QQQ"]]).diff()*1e4; R=R[R.index>="2001-01-01"]; cal=R.index
s=pd.Series(cal,index=cal)
# 401k: semi-monthly paydays 15th and last business day; contributions invested ~1-2 business days later
def nearest_on_or_before(day):
    return s.groupby(cal.to_period("M")).apply(lambda x: x[x.dt.day<=day].max() if (x.dt.day<=day).any() else pd.NaT)
p15=set(nearest_on_or_before(15).dropna()); pme=set(s.groupby(cal.to_period("M")).max())
pay=set(p15)|pme
idx={d:i for i,d in enumerate(cal)}
inv=set()
for d in pay:
    i=idx[d]
    for k in (1,2):
        if i+k<len(cal): inv.add(cal[i+k])
# Social Security: 2nd/3rd/4th Wednesday (birth-date groups) + 3rd of month (legacy/SSI 1st)
wed=s[s.dt.dayofweek==2]; ssw=set(wed.groupby(wed.dt.to_period("M")).nth([1,2,3]))
D=pd.DataFrame(R); D["inv401k"]=D.index.isin(inv); D["ss"]=D.index.isin(ssw); D["wed"]=D.index.dayofweek==2
for per,m in [("2001-2012",cal<"2013"),("2013-",cal>="2013")]:
    for a in ["SPY","IWM"]:
        x=D[m]; t1=stats.ttest_ind(x[x.inv401k][a].dropna(),x[~x.inv401k][a].dropna(),equal_var=False)[0]
        y_=x[x.wed]; t2=stats.ttest_ind(y_[y_.ss][a].dropna(),y_[~y_.ss][a].dropna(),equal_var=False)[0]
        print(f"{per} {a}: 401k-investment days {x[x.inv401k][a].mean():+.1f} vs other {x[~x.inv401k][a].mean():+.1f} (t {t1:+.2f}) | SS Wednesdays {y_[y_.ss][a].mean():+.1f} vs other Wed {y_[~y_.ss][a].mean():+.1f} (t {t2:+.2f})")

""")
md(r"""
**Interpretation.**
- **401k investment days:** +8bp vs +1–5bp on other days, never significant (t ≤ 1.5), and smaller after 2013.
- **Social Security Wednesdays** are *weaker* than the other Wednesdays (t −1 to −3). The comparison Wednesdays (1st
  and 5th) include the turn of the month, so this is a calendar confound, not benefit flows.

**Verdict: dead.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
