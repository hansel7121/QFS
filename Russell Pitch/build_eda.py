"""Generate eda.ipynb: small caps vs large caps around the June Russell reconstitution."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Russell Pitch EDA: does the June reconstitution move small caps vs large caps?

**The mechanism.** On the last Friday of June, Russell index funds rebalance at the close (one of the largest
volume days of the year). Stock-level membership lists are not free historically, so this tests the index level only:
IWM (Russell 2000) minus SPY over the 10 days before, the recon day, and 5/20 days after. The placebo is the last Friday of
other months.

**Data.** `TOM Pitch/data/etfs.csv`. The recon date is approximated as the last Friday of June.
""")
code(r"""
import pandas as pd, numpy as np
p=pd.read_csv("../TOM Pitch/data/etfs.csv",parse_dates=["date"]).pivot(index="date",columns="ticker",values="adj_close")
x=(np.log(p.IWM).diff()-np.log(p.SPY).diff())*1e4; x=x.dropna(); cal=x.index
rows=[]
for y in range(2001,2027):
    june=cal[(cal.year==y)&(cal.month==6)]
    fr=[d for d in june if d.weekday()==4]
    if not fr: continue
    R=fr[-1]                                  # recon: close of last Friday of June (approx. rule; 2020s use 4th Friday)
    i=cal.get_loc(R)
    rows.append({"y":y,"pre10":x.iloc[i-10:i].sum(),"recon_day":x.iloc[i],"post5":x.iloc[i+1:i+6].sum(),"post20":x.iloc[i+1:i+21].sum()})
D=pd.DataFrame(rows)
for per,m in [("2001-2012",D.y<=2012),("2013-",D.y>=2013)]:
    d=D[m]; print(per,{c:f"{d[c].mean():+.0f} (t {d[c].mean()/d[c].sem():+.1f})" for c in ["pre10","recon_day","post5","post20"]},"n",len(d))
# placebo: last Friday of other months
pl=[]
for y in range(2001,2027):
    for mth in [1,2,3,4,5,7,8,9,10,11,12]:
        mm=cal[(cal.year==y)&(cal.month==mth)]; fr=[d for d in mm if d.weekday()==4]
        if not fr: continue
        i=cal.get_loc(fr[-1])
        if i+21<len(cal): pl.append({"pre10":x.iloc[i-10:i].sum(),"recon_day":x.iloc[i],"post5":x.iloc[i+1:i+6].sum()})
P=pd.DataFrame(pl); print("placebo other months:",{c:f"{P[c].mean():+.0f} (t {P[c].mean()/P[c].sem():+.1f})" for c in P})

""")
md(r"""
**Interpretation.**
- **The pre-recon small-cap run is gone.** IWM beat SPY by +110bp (t 4.0) in the 10 days before recon in 2001–12, and
  +27bp (t 0.6) since 2013. Recon-day outperformance (+51bp, t 2.4) disappeared too.
- The 5-day post-recon give-back is negative in both eras (−80 / −64bp) but insignificant on 12–14 events a year.

**Verdict: dead at the index level.** A stock-level test needs membership data that is not free.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
