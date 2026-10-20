"""Generate eda.ipynb: the monthly option-expiration week effect in SPY, before and after 0DTE options."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Opex Pitch EDA: do equities rise in option-expiration week?

**The idea.** Stivers & Sun (2013) and others report high stock returns in the week of the monthly option expiration
(third Friday), linked to dealers' hedge rebalancing as large open interest expires. Daily 0DTE options (2022–) changed
how dealer gamma is distributed, so the question is whether anything is left, by era.

**Data.** SPY total return (`TOM Pitch/data/etfs.csv`). Opex week = the 5 trading days ending on the third Friday (or
the trading day before it if that Friday is a holiday). "Week after" = the following 5 days.
""")
code(r"""
import pandas as pd, numpy as np
from scipy import stats
p=pd.read_csv("../TOM Pitch/data/etfs.csv",parse_dates=["date"]).pivot(index="date",columns="ticker",values="adj_close")
r=(p.SPY.pct_change()*1e4).dropna(); cal=r.index
# third Friday of each month (if holiday, Thursday before -> use last trading day <= third Friday)
opex=[]
for per in pd.period_range(cal[0],cal[-1],freq="M"):
    d=pd.Timestamp(per.year,per.month,1); fr=[d+pd.Timedelta(days=i) for i in range(31) if (d+pd.Timedelta(days=i)).month==per.month and (d+pd.Timedelta(days=i)).weekday()==4][2]
    i=cal.searchsorted(fr,side="right")-1
    if i>=0 and cal[i].month==per.month: opex.append(i)
lab=pd.Series("other",index=cal)
for i in opex:
    lab.iloc[max(i-4,0):i+1]="opex week"
    lab.iloc[i+1:i+6]="week after"
eras=[("1993-2012",cal<"2013"),("2013-2019",(cal>="2013")&(cal<"2020")),("2020- (0DTE era)",cal>="2020")]
for nm,m in eras:
    g=r[m].groupby(lab[m]).agg(["mean","count"])
    a=r[m&(lab=="opex week").values]; b=r[m&(lab=="other").values]; c=r[m&(lab=="week after").values]
    t1,_=stats.ttest_ind(a,b,equal_var=False); t2,_=stats.ttest_ind(c,b,equal_var=False)
    st=r[m].where((lab[m]=="opex week"),0)
    print(f"{nm}: opex wk {a.mean():+.1f} vs other {b.mean():+.1f} (t {t1:+.2f}) | week after {c.mean():+.1f} (t {t2:+.2f}) | opex-week-only Sharpe {st.mean()/st.std()*np.sqrt(252):.2f} vs B&H {r[m].mean()/r[m].std()*np.sqrt(252):.2f}")

""")
md(r"""
**Interpretation.**
- **No opex-week premium in any era.** 1993–2012: identical to other days (t 0.03). 2013–19: +10.5 vs +4.5bp/day
  (t 1.3). 2020–: negative (−3.4 vs +9.2, t −1.7).
- Holding SPY only in opex weeks never beats holding it all the time.

**Verdict: dead.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
