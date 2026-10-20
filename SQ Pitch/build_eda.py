"""Generate eda.ipynb: Nikkei 225 around SQ (special quotation) settlement, the opening auction of the 2nd Friday."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# SQ Pitch EDA: does the Nikkei SQ settlement move the open?

**The mechanism.** Nikkei 225 options (monthly) and futures (quarterly) settle on the SQ, computed from each
constituent's *opening* price on the 2nd Friday. Arbitrage books unwind and dealers settle hedges at that open, so the
opening auction should carry a large imbalance (a gap), possibly reversed during the day.

**Design.** ^N225 daily open/close (`JapanDiv Pitch/data/prices.csv`); SQ = 2nd Friday (holiday shifts ignored).
Compared with other Fridays: 3 days before, the opening gap, open→close, the next day. 2000–12 vs 2013–.
""")
code(r"""
import pandas as pd, numpy as np
from scipy import stats
d=pd.read_csv("../JapanDiv Pitch/data/prices.csv",parse_dates=["date"])
O=d.pivot(index="date",columns="ticker",values="open")["^N225"]; C=d.pivot(index="date",columns="ticker",values="close")["^N225"]
ok=(O>0)&(C>0); O=O[ok]; C=C[ok]; cal=C.index
gap=np.log(O/C.shift(1))*1e4; idr=np.log(C/O)*1e4; nxt=np.log(C.shift(-1)/C)*1e4; pre=np.log(C.shift(1)/C.shift(4))*1e4
fr=pd.Series(cal,index=cal); fr=fr[fr.dt.dayofweek==4]
sq=fr.groupby(fr.dt.to_period("M")).nth(1)   # 2nd Friday (holiday shifts ignored: approx)
sqd=set(pd.DatetimeIndex(sq.values))
D=pd.DataFrame({"gap":gap,"id":idr,"next":nxt,"pre3":pre},index=cal)
D["sq"]=D.index.isin(sqd); D["q"]=D.sq&D.index.month.isin([3,6,9,12]); D["fri"]=D.index.dayofweek==4
for per,m in [("2000-2012",(D.index>="2000")&(D.index<"2013")),("2013-",D.index>="2013")]:
    base=D[m&D.fri&~D.sq]
    for nm,sel in [("monthly SQ",m&D.sq&~D.q),("quarterly SQ",m&D.q)]:
        x=D[sel]
        out={c:f"{x[c].mean():+.0f} vs {base[c].mean():+.0f} (t {stats.ttest_ind(x[c].dropna(),base[c].dropna(),equal_var=False)[0]:+.1f})" for c in ["pre3","gap","id","next"]}
        out["|gap|"]=f"{x.gap.abs().mean():.0f} vs {base.gap.abs().mean():.0f}"
        print(per,nm,"n",len(x),out)
    # gap reversal on SQ: id ~ gap
    x=D[m&D.sq]; b=D[m&D.fri&~D.sq]
    print("   corr(gap, open->close): SQ",round(x.gap.corr(x.id),2),"other Fri",round(b.gap.corr(b.id),2))

""")
md(r"""
**Interpretation.**
- **No SQ effect on the open.** Gaps on SQ days are not different in sign (|t| < 1) and only slightly bigger in size
  (69–74 vs 59bp since 2013). The open→close reversal is no stronger than on other Fridays (gap/intraday correlation
  0.19 vs 0.17).
- The one significant number (3 days before monthly SQ, −64bp, t −2.5, 2000–12) flips sign after 2013.

**Verdict: dead.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
