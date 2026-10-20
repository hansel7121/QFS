"""Generate eda.ipynb: the dividend-month premium (Hartzmark & Solomon 2013) after publication."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# DivMonth Pitch EDA: do stocks earn more in months they are expected to go ex-dividend?

**The anomaly.** Hartzmark & Solomon (*JFE* 2013): stocks earn about 40bp/month abnormal returns in months when a
dividend is predictably expected (same month a quarter or a year earlier), driven by dividend-seeking demand before
the ex-date and reversed afterwards.

**Design.** About 2,800 fund-held US stocks, 2019–2026 (`FundFlow Pitch/data/prices.csv`). Ex-dividend dates are read
from Yahoo's adjustment factor (adjusted / raw close steps up by dividend/price on each ex-date; checked: four per
year for quarterly payers, dates match published ones). Predicted dividend month = went ex 3 or 12 months earlier.
Outcome: total-return monthly return minus the equal-weighted universe, predicted-month payers vs other payers.
2020–22 vs 2023–.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
P=pd.read_csv("../FundFlow Pitch/data/prices.csv",parse_dates=["date"]).drop_duplicates(["date","ticker"])
A=P.pivot(index="date",columns="ticker",values="adj_close"); C=P.pivot(index="date",columns="ticker",values="close")
A=A.where(A>0); C=C.where(C>0)
f=(A/C)                                            # adjustment factor; steps up on ex-dividend dates (looking backward)
step=(f/f.shift(1)-1)                              # positive jump at ex-date = dividend/price
ex=(step>0.0005)&(step<0.15)                       # cash dividend between 5bp and 15% of price
exm=ex.groupby(ex.index.to_period("M")).any()      # stock x month: went ex-div that month
LA=np.log(A); RD=LA.diff(); RD=RD.where(RD.abs()<0.5)
mret=RD.groupby(RD.index.to_period("M")).sum(min_count=10)*1e4
mret=mret.sub(mret.mean(1),axis=0)                 # vs equal-weighted universe
# predicted dividend month: ex-div in month m-3 or m-12 (quarterly / annual payers), known at the start of m
pred=(exm.shift(3)|exm.shift(12)).fillna(False)
payer=exm.rolling(12,min_periods=12).sum().shift(1)>0
rows=[]
for m in mret.index[13:]:
    r=mret.loc[m]; pdm=pred.loc[m]; pay=payer.loc[m]
    ok=r.notna()&pay.fillna(False)
    a=r[ok&pdm]; b=r[ok&~pdm]
    if len(a)>20 and len(b)>20: rows.append({"m":m,"pred":a.mean(),"other_payers":b.mean(),"n":len(a)})
D=pd.DataFrame(rows).set_index("m"); D["spread"]=D.pred-D.other_payers
for per,mk in [("2020-2022",D.index.year<=2022),("2023-",D.index.year>=2023)]:
    x=D[mk].spread; print(f"{per}: predicted-dividend-month payers minus other payers {x.mean():+.1f}bp/month (t {x.mean()/x.std()*np.sqrt(len(x)):+.2f}, months {len(x)}), hit {(x>0).mean():.0%}, ann Sharpe {x.mean()/x.std()*np.sqrt(12):+.2f}")
print("by calendar month of year:", D.groupby(D.index.month).spread.mean().round(0).to_dict())
cnt=ex.groupby(ex.index.year).sum()
print("ex-div events per stock-year (payers only), median:", cnt[cnt>0].stack().groupby(level=0).median().to_dict())
for t in ["KO","JNJ","AAPL","MSFT"]:
    if t in ex: print(t, list(ex.index[ex[t]].strftime("%Y-%m-%d")[-4:]))

""")
md(r"""
**Interpretation.** The premium is gone, even negative: predicted dividend months return −13bp/month (t −0.5) in
2020–22 and −23bp (t −1.2) in 2023– relative to other payers' months.

**Verdict: dead after publication.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
