"""Generate eda.ipynb: share of S&P 500 buyback dollars in earnings blackout vs next-week index returns."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# BlackoutAgg Pitch EDA: is the market weaker when corporate buybacks are in blackout?

**The claim** (sell-side commentary every earnings season): when most S&P 500 buyback dollars are in pre-earnings
blackout, the biggest net buyer of US stocks steps away and the index is weaker.

**Design.** Earnings dates (8-K Item 2.02) and annual buybacks from `Buyback Pitch/data`. Each week, the share of
buyback dollars whose firm is within 35 calendar days before earnings (not yet reported), using only buybacks
filed by then. Outcome: SPY over the next 5 trading days. 2010–17 vs 2018–.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, statsmodels.api as sm
e=pd.read_csv("../Buyback Pitch/data/earnings.csv",parse_dates=["date"]).sort_values(["ticker","date"])
e=e[e.groupby("ticker").date.diff().dt.days.fillna(999)>45]
b=pd.read_csv("../Buyback Pitch/data/buybacks.csv",parse_dates=["fy_end","filed"]).dropna(subset=["repurchases"])
spy=pd.read_csv("../TOM Pitch/data/etfs.csv",parse_dates=["date"]).pivot(index="date",columns="ticker",values="adj_close").SPY
cal=spy.index[spy.index>="2010-01-01"]
# each firm's annual buyback $ (last filed before date); blackout = from 35 calendar days before earnings to 1 trading day after
bb=b.sort_values("filed")
open_share=[]
for d in cal[::5]:                                         # weekly sample
    last=bb[bb.filed<=d].groupby("ticker").repurchases.last()
    if last.sum()<=0: continue
    # next and previous earnings per firm relative to d
    ee=e[e.ticker.isin(last.index)]
    nxt=ee[ee.date>=d].groupby("ticker").date.min(); prv=ee[ee.date<d].groupby("ticker").date.max()
    inb=((nxt-d).dt.days<=35).reindex(last.index).fillna(False) & ~(((d-prv).dt.days<=1).reindex(last.index).fillna(False))
    open_share.append({"d":d,"blackout_share":(last[inb].sum()/last.sum())})
S=pd.DataFrame(open_share).set_index("d")
fw=np.log(spy.shift(-5)/spy)*1e4
S["fwd1w"]=fw.reindex(S.index)
S=S.dropna()
print(S.blackout_share.describe().round(2).to_dict())
for per,m in [("2010-2017",S.index<"2018"),("2018-",S.index>="2018")]:
    d=S[m]; f=sm.OLS(d.fwd1w,sm.add_constant(d.blackout_share)).fit(cov_type="HAC",cov_kwds={"maxlags":2})
    hi=d[d.blackout_share>0.7].fwd1w.mean(); lo=d[d.blackout_share<0.3].fwd1w.mean()
    print(f"{per}: slope {f.params.blackout_share:+.0f}bp per 100% in blackout (t {f.tvalues.blackout_share:+.2f}); weeks >70% in blackout {hi:+.0f}bp vs <30% {lo:+.0f}bp, n {len(d)}")

""")
md(r"""
**Interpretation.** The relation has the *wrong* sign and is insignificant in both eras: weeks with more than 70% of
buyback dollars in blackout return +49 / +58bp vs +32 / +12bp when less than 30% are (t 0.7 / 1.5). Corporate
buybacks are spread out (10b5-1 plans keep running) and the blackout calendar coincides with earnings season, which
has its own positive drift.

**Verdict: dead.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
