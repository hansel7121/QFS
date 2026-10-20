"""Generate eda.ipynb: SPY on quarterly index-rebalance / quad-witching Fridays and the overnight after."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# QuadWitch Pitch EDA: does the quarterly rebalance close distort SPY, and does it reverse?

**The mechanism.** On the third Friday of March/June/September/December, S&P quarterly share and float updates, index
futures expiry and stock/index option expiry all hit the closing auction. If the forced trading pushes the close,
it should reverse overnight and the next day.

**Data.** SPY daily open/close (`Overnight Pitch/data/panel.csv`), 2004–2026. Quad-witching Fridays and monthly opex
Fridays vs other Fridays: the day's return, close → next open, and the next day. 2004–14 vs 2015–.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from scipy import stats
p=pd.read_csv("../Overnight Pitch/data/panel.csv",parse_dates=["date"])
s=p[p.ticker=="SPY"].set_index("date").sort_index()
f=s.adj_close/s.close
on=np.log(s.open.shift(-1)*f.shift(-1)/(s.close*f))*1e4          # close -> next open (dividend-adjusted)
day=np.log(s.adj_close/s.adj_close.shift(1))*1e4
cal=s.index
def third_fri(y,m):
    d=pd.date_range(pd.Timestamp(y,m,1),periods=31,freq="D"); fr=d[(d.weekday==4)&(d.month==m)][2]
    return cal[cal.searchsorted(fr,side="right")-1]
qw={third_fri(y,m) for y in range(2004,2027) for m in (3,6,9,12) if pd.Timestamp(y,m,28)<=cal[-1]}
mo={third_fri(y,m) for y in range(2004,2027) for m in range(1,13) if m not in (3,6,9,12) and pd.Timestamp(y,m,28)<=cal[-1]}
D=pd.DataFrame({"day":day,"next_on":on,"next_day":day.shift(-1)})
D["kind"]=np.where(D.index.isin(qw),"quad witching (index rebalance)",np.where(D.index.isin(mo),"monthly opex Friday",np.where(D.index.dayofweek==4,"other Friday","other")))
for per,m in [("2004-2014",D.index<"2015"),("2015-",D.index>="2015")]:
    b=D[m&(D.kind=="other Friday")]
    for k in ["quad witching (index rebalance)","monthly opex Friday"]:
        a=D[m&(D.kind==k)]
        print(per,f"{k:32s} n {len(a)}: "+" | ".join(f"{c} {a[c].mean():+.1f} vs {b[c].mean():+.1f} (t {stats.ttest_ind(a[c].dropna(),b[c].dropna(),equal_var=False)[0]:+.2f})" for c in ["day","next_on","next_day"]))

""")
md(r"""
**Interpretation.** No reversal in either era (next open and next day, |t| ≤ 0.6). Quad-witching days themselves
were flat in 2004–14 (+10bp, t 0.7) and weak since 2015 (−37bp, t −2.9). That is a single-era pattern with no
in-sample support, and not a pressure-reversal mechanism.

**Verdict: fails the bar.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
