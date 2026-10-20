"""Generate eda.ipynb: VIX futures - ETP roll pressure (data-limited) and the expiration-morning settlement (Griffin & Shams 2018)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# VIX Pitch EDA: mechanical flows in VIX futures

**1. ETP roll pressure (not testable with free data).** VXX/UVXY/VIXY sell the front VX contract and buy the second
every day between expiries. The size depends on the ETPs' daily shares outstanding, which is not freely downloadable,
so every day is a roll day and there is no variation to test. `collect_vx.py` still collects the Cboe VX settlement
history (2013–) for later use.

**2. Expiration-morning settlement.** VX settles on Wednesday mornings to the SOQ, built from SPX option opening
prices. Griffin & Shams (*RFS* 2018) document bid-ups of OTM SPX puts at that open, pushing the settlement up. Here
the proxy is the VIX index's first print (Yahoo daily open) on expiration Wednesdays vs other Wednesdays, before and
after the 2018 paper. The true SOQ (VRO) history is not on Yahoo.
""")
code(r"""
import pandas as pd, numpy as np, yfinance as yf
from scipy import stats
v=yf.download("^VIX",start="2004-01-01",progress=False,auto_adjust=False)
v.columns=v.columns.get_level_values(0); v=v[["Open","Close"]].dropna(); v.to_csv("data/vix_daily.csv")
cal=v.index
def expiry(y,m):
    nm=pd.Timestamp(y,m,1)+pd.offsets.MonthBegin(1); fr=pd.date_range(nm,nm+pd.offsets.MonthEnd(0),freq="W-FRI"); return fr[2]-pd.Timedelta(days=30)
ex=set()
for y in range(2004,2027):
    for m in range(1,13):
        e=expiry(y,m)
        if e in cal: ex.add(e)
        elif e-pd.Timedelta(days=1) in cal: ex.add(e-pd.Timedelta(days=1))
D=pd.DataFrame({"gap":np.log(v.Open/v.Close.shift(1))*100,"id":np.log(v.Close/v.Open)*100},index=cal)
D["ex"]=D.index.isin(ex); D["wed"]=D.index.dayofweek==2
for per,m in [("2004-2017",D.index<"2018"),("2018-",D.index>="2018")]:
    a=D[m&D.ex]; b=D[m&D.wed&~D.ex]
    print(per,"n",len(a),{c:f"{a[c].mean():+.2f}% vs {b[c].mean():+.2f}% (t {stats.ttest_ind(a[c],b[c],equal_var=False)[0]:+.2f})" for c in ["gap","id"]})

""")
md(r"""
**Interpretation.** On expiration Wednesdays the VIX opening gap and the open→close move are no different from other
Wednesdays in either era (|t| ≤ 1.6). The daily VIX open is only a proxy for the SOQ, but nothing in it suggests a
tradable settlement push.

**Verdict: nothing testable / dead with free data.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
