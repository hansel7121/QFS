"""Generate eda.ipynb: does a new single-stock leveraged ETF's seed hedge lift the underlying on launch day?"""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# LETFLaunch Pitch EDA: seed hedging at leveraged-ETF launches

**The mechanism.** When a 2× single-stock ETF launches, its swap dealers buy 2× the seed capital (plus first-day
creations) in the underlying. For the first LETF on a smaller stock that could be a visible, one-off buy.

**Design.** LETF universe and prices from `LETF Pitch/data` (first trading day = launch, 2022-07 onwards). Underlying
return minus SPY around the launch: 5 days before, launch day, 5 and 20 days after. Winsorised 2%.
""")
code(r"""
import pandas as pd, numpy as np
u=pd.read_csv("../LETF Pitch/data/universe.csv")
d=pd.read_csv("../LETF Pitch/data/daily.csv",parse_dates=["date"])
A=d.pivot(index="date",columns="ticker",values="adj_close"); V=d.pivot(index="date",columns="ticker",values="volume")
spy=np.log(A.SPY).diff()*1e4; R=np.log(A).diff()*1e4; cal=A.index
rows=[]
for r in u.itertuples():
    if r.etf not in A or r.under not in R: continue
    first=A[r.etf].first_valid_index()
    if first is None or first<pd.Timestamp("2022-07-01"): continue
    i=cal.get_loc(first)
    if i<25 or i+20>=len(cal): continue
    x=(R[r.under]-spy).values
    prior=[e for e in u[u.under==r.under].etf if e in A and A[e].first_valid_index() is not None and A[e].first_valid_index()<first]
    rows.append({"etf":r.etf,"under":r.under,"lev":r.lev,"date":first,"first_on_stock":len(prior)==0,
                 "pre5":np.nansum(x[i-5:i]),"d0":x[i],"post5":np.nansum(x[i+1:i+6]),"post20":np.nansum(x[i+1:i+21]),
                 "etf_dv20":np.nanmean((A[r.etf]*V[r.etf]).values[i:i+20])/1e6})
E=pd.DataFrame(rows)
for nm,m in [("all launches",E.index==E.index),("first LETF on that stock",E.first_on_stock),("first, long 2x",E.first_on_stock&(E.lev>0))]:
    d_=E[m]; w=lambda c: d_[c].clip(d_[c].quantile(.02),d_[c].quantile(.98))
    print(nm,"n",len(d_),{c:f"{w(c).mean():+.0f} (t {w(c).mean()/w(c).sem():+.1f}) med {d_[c].median():+.0f}" for c in ["pre5","d0","post5","post20"]})

""")
md(r"""
**Interpretation.**
- **Launch day is up** (+60bp all launches, t 2.3; +82bp for the first LETF on a stock, t 2.0, median +44bp), with
  no follow-through (post-20 insignificant).
- **Not tradable as is.** Capturing it means buying at the close before launch, which needs the launch date publicly
  fixed in advance (prospectus effective dates usually are, but that was not verified). One sample, 2022–26, no
  out-of-sample period. Issuers launch on hot stocks, so the launch-day pop may be selection, not seed flow.

**Verdict: weak.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
