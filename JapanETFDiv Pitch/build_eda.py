"""Generate eda.ipynb: Japanese ETF distribution-funding sales around July 8-10 (record dates of 1321 / 1306)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# JapanETFDiv Pitch EDA: do Japanese ETFs' July payouts force selling?

**The mechanism.** The big Japanese equity ETFs (1321 Nikkei, 1306 TOPIX and peers, swollen by BOJ purchases from
2013) pay their annual distribution with record dates on July 8 and July 10. To fund about ¥1tn in cash they sell
stocks/futures in the days around those dates (Japanese press: 分配金捻出売り), then the pressure ends.

**Design.** ^N225 daily close (`JapanDiv Pitch/data/prices.csv`). Event day = first trading day on or after the 8th of
each month; July vs the same day in other months. Windows: 4 days before, the day, 4 days after. 2001–12 (small
ETFs) vs 2013–.
""")
code(r"""
import pandas as pd, numpy as np
from scipy import stats
d=pd.read_csv("../JapanDiv Pitch/data/prices.csv",parse_dates=["date"])
C=d.pivot(index="date",columns="ticker",values="close")["^N225"]; C=C[C>0]; cal=C.index
r=np.log(C).diff()*1e4
rows=[]
for y in range(2001,2027):
    for m in range(1,13):
        tgt=pd.Timestamp(y,m,8)          # 1321 record date July 8; 1306 July 10 -> selling in the days up to ~July 7-9
        i=cal.searchsorted(tgt)
        if i<5 or i+5>=len(cal): continue
        rows.append({"y":y,"m":m,"pre4":r.iloc[i-4:i].sum(),"day":r.iloc[i],"post4":r.iloc[i+1:i+5].sum()})
D=pd.DataFrame(rows)
for per,m in [("2001-2012",D.y<=2012),("2013-",D.y>=2013)]:
    j=D[m&(D.m==7)]; o=D[m&(D.m!=7)]
    print(per,"July n",len(j),{c:f"{j[c].mean():+.0f} vs {o[c].mean():+.0f} (t {stats.ttest_ind(j[c],o[c],equal_var=False)[0]:+.2f})" for c in ["pre4","day","post4"]})
print(D[(D.m==7)&(D.y>=2013)].set_index("y")[["pre4","post4"]].round(0).T.to_string())
j=D[(D.m==7)&(D.y>=2013)].post4
print(f"post4 2013-: mean {j.mean():+.0f} median {j.median():+.0f} t {j.mean()/j.std()*np.sqrt(len(j)):+.2f}; ex-2016 mean {j[D.y!=2016].mean():+.0f} t {j[D.y!=2016].mean()/j[D.y!=2016].std()*np.sqrt(len(j)-1):+.2f}")

""")
md(r"""
**Interpretation.**
- **2013–: a sell-then-rebound shape around July 8.** The day itself −68bp vs +25bp in other months (t −2.8). The
  next 4 days +162bp vs +38bp (t 2.0), positive in 12 of 14 years. Without the 2016 Brexit rebound the mean is
  still about +110bp.
- **Absent in 2001–12** (small ETFs), which fits the mechanism but means there is no out-of-sample period.
- **One event a year, three windows looked at:** too few observations and some selection.

**Verdict: a lead, not a pass.** Check exact record dates per year (the 8th and 10th shift with weekends) and the
2027 event out of sample.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
