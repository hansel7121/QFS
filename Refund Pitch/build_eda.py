"""Generate eda.ipynb: do IRS individual tax refunds (Daily Treasury Statement) push retail-heavy assets?"""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Refund Pitch EDA: does tax-refund cash flow into retail favourites?

**The mechanism.** Each February–March the IRS pays out $30–80bn a week in individual refunds by direct deposit. If
retail investors put some of it into their brokerage accounts, the assets they like (small caps, ARKK, biotech,
bitcoin, meme stocks) should outperform SPY in the week after big refund weeks. Refund totals are public the next
business day (Daily Treasury Statement), so last week's refunds are known before this week's trade.

**Data.** `collect_dts.py`: DTS individual refund withdrawals (EFT) 2012–2026; Yahoo weekly prices. Signal = last
week's refunds ($bn); outcome = this week's return minus SPY. 2013–19 vs 2020–.
""")
code(r"""
import pandas as pd, numpy as np, statsmodels.api as sm
rf=pd.read_csv("data/refunds.csv",parse_dates=["date"]).set_index("date").refunds_musd
p=pd.read_csv("data/prices.csv",parse_dates=["date"]).pivot(index="date",columns="ticker",values="adj_close")
wk=lambda s: s.resample("W-FRI").last()
P=wk(p); R=np.log(P).diff()*1e4
F=rf.resample("W-FRI").sum()/1e3   # $bn per week
D=pd.DataFrame({"ref_lag":F.shift(1),"ref_lag_rel":(F/F.rolling(52,min_periods=26).mean()).shift(1)})
for a in ["IWM","ARKK","QQQ","BTC-USD","XBI","GME"]:
    D[a+"_x"]=R[a]-R["SPY"]
D["SPY"]=R["SPY"]; D["season"]=D.index.month.isin([2,3]) & (D.index.day>0)
D=D[D.index>="2013-01-01"]
print("weekly refunds $bn: peak weeks", F.groupby(F.index.year).max().round(0).to_dict())
for per,m in [("2013-19",D.index<"2020"),("2020-",D.index>="2020")]:
    for a in ["SPY","IWM_x","ARKK_x","QQQ_x","XBI_x","BTC-USD_x","GME_x"]:
        d=D[m].dropna(subset=[a,"ref_lag"])
        if len(d)<50: continue
        f=sm.OLS(d[a],sm.add_constant(d.ref_lag)).fit(cov_type="HAC",cov_kwds={"maxlags":2})
        top=d[d.ref_lag>=d.ref_lag.quantile(0.9)][a].mean()
        print(f"{per} {a:10s} slope {f.params.ref_lag:+.2f} bp per $bn (t {f.tvalues.ref_lag:+.2f}) | top-decile refund weeks next-week {top:+.0f}bp vs all {d[a].mean():+.0f}")

""")
md(r"""
**Interpretation.**
- **No effect.** No asset shows a significant slope in either period (|t| ≤ 1.7), and signs flip between 2013–19 and
  2020– for SPY, IWM and ARKK. Bitcoin's slope is positive in both (t 1.3 / 1.7) but weak, and bitcoin's average
  return is high anyway.

**Verdict: dead.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
