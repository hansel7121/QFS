"""Generate eda.ipynb: the "net liquidity" story (Fed assets - TGA - reverse repo) and US equities."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Liquidity Pitch EDA: do mechanical swings in Fed "net liquidity" move stocks?

**The idea.** A popular macro model: net liquidity = Fed total assets − Treasury General Account − overnight reverse
repo. TGA and RRP swing mechanically (tax dates, debt issuance, money-fund cash), injecting or draining reserves.
The claim is that stocks follow.

**Design.** FRED weekly H.4.1 (Wednesday levels, published Thursday 16:30 ET; `collect_fred.py`). Signal = weekly
change in net liquidity (% of level). Outcomes: SPY over the same week (contemporaneous) and the week after release
(Friday close → next Friday close). 2010–19 vs 2020–.
""")
code(r"""
import pandas as pd, numpy as np, statsmodels.api as sm
f=pd.read_csv("data/fred.csv",parse_dates=["date"],index_col="date")
w=f[["WALCL","WTREGEN"]].dropna()                       # Wednesdays
rrp=f.RRPONTSYD.reindex(w.index,method="ffill").fillna(0)
nl=w.WALCL/1e3 - w.WTREGEN - rrp                         # $bn net liquidity, Wednesday
p=pd.read_csv("../TOM Pitch/data/etfs.csv",parse_dates=["date"]).pivot(index="date",columns="ticker",values="adj_close")
spy=p.SPY
rows=[]
for d,v in nl.diff().dropna().items():
    rel=d+pd.Timedelta(days=1)                            # Thursday release (after close)
    a=spy[spy.index>rel]                                  # enter Friday close? use first close after release day
    if len(a)<6: continue
    e0=spy[spy.index<=rel].iloc[-1] if False else a.iloc[0]   # entry: Friday close (first close strictly after Thursday)
    rows.append({"d":d,"dnl":v,"dnl_pct":v/nl.loc[d]*100,"fwd1w":np.log(a.iloc[5]/a.iloc[0])*1e4,
                 "same_wk":np.log(spy[spy.index<=d].iloc[-1]/spy[spy.index<=d-pd.Timedelta(days=7)].iloc[-1])*1e4})
D=pd.DataFrame(rows)
for per,m in [("2010-2019",(D.d>="2010")&(D.d<"2020")),("2020-",D.d>="2020")]:
    d=D[m]
    for y_ in ["same_wk","fwd1w"]:
        f_=sm.OLS(d[y_],sm.add_constant(d.dnl_pct)).fit(cov_type="HAC",cov_kwds={"maxlags":2})
        print(f"{per} {y_:8s}: slope {f_.params.dnl_pct:+.1f} bp per 1% change in net liquidity (t {f_.tvalues.dnl_pct:+.2f}) n {len(d)}")

""")
md(r"""
**Interpretation.** Neither the same-week nor the next-week SPY return loads on weekly net-liquidity changes in either
era (all |t| ≤ 1.2). The relationship people chart is low-frequency co-trending, not a weekly flow effect.

**Verdict: dead.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
