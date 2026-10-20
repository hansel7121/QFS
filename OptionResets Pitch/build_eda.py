"""Generate eda.ipynb: SPY into the close on option-overlay reset days (buffer ETFs monthly, JPM collar quarterly)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# OptionResets Pitch EDA: do option-overlay resets move the S&P close?

**Mechanisms.**
* **Buffer / defined-outcome ETFs** (about $60–70bn) reset monthly series at the close of the first trading day of
  the month, buying new FLEX option packages; dealers hedge the delta at the close.
* **JPMorgan Hedged Equity** rolls its large SPX put-spread collar at the close of the last business day of each quarter.

**Design.** SPY 5-minute bars 2020–2026 (`IndexGamma Pitch/data`), official closes from Yahoo. Last 30 / 60 minutes and
absolute last-30 move on reset days vs ordinary days; other month-ends as a comparison.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from scipy import stats
x=pd.read_csv("../IndexGamma Pitch/data/intraday/SPY.csv",parse_dates=["datetime"]); x["d"]=x.datetime.dt.normalize(); x["hm"]=x.datetime.dt.strftime("%H:%M")
c=x.pivot_table(index="d",columns="hm",values="close"); c=c[c.notna().sum(1)>=76]
cl=pd.read_csv("../TOM Pitch/data/etfs.csv",parse_dates=["date"]).pivot(index="date",columns="ticker",values="close").SPY
C=cl.reindex(c.index); P=cl.shift(1).reindex(c.index)
D=pd.DataFrame({"last30":np.log(C/c["15:25"])*1e4,"last60":np.log(C/c["14:55"])*1e4,"day":np.log(C/P)*1e4,"abs30":np.abs(np.log(C/c["15:25"])*1e4)})
s=pd.Series(D.index,index=D.index); ym=D.index.to_period("M")
first=s.groupby(ym).transform("min")==s; last=s.groupby(ym).transform("max")==s
qend=last&D.index.month.isin([3,6,9,12])
D["kind"]=np.select([first,qend,last],["first day (buffer resets)","quarter-end (JPM collar)","other month-end"],"other day")
nxt=D.shift(-1)
for k in ["first day (buffer resets)","quarter-end (JPM collar)","other month-end"]:
    a=D[D.kind==k]; b=D[D.kind=="other day"]
    print(f"{k:28s} n {len(a):3d}: " + " | ".join(f"{c} {a[c].mean():+.1f} vs {b[c].mean():+.1f} (t {stats.ttest_ind(a[c].dropna(),b[c].dropna(),equal_var=False)[0]:+.2f})" for c in ["last30","last60","abs30"]))

""")
md(r"""
**Interpretation.** Neither reset leaves a mark: first-day closes (+3bp, t 0.9) and quarter-end closes (−7bp, t −0.9)
match ordinary days, and quarter-end volatility is only mildly higher (t 1.6). Ordinary month-ends are the ones that
stand out (+11bp in the last hour, t 1.9; volatility up, t 2.6): the generic month-end closing imbalance, not the
option overlays.

**Verdict: dead.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
