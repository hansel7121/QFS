"""Generate eda.ipynb: mutual fund / ETF flow-induced trading (Lou 2012; Coval & Stafford 2007) from N-PORT, 2019-2026."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# FundFlow Pitch EDA: do stocks that funds must buy or sell because of flows overshoot and reverse?

**The mechanism.** Open-end funds that receive inflows buy more of what they hold; funds hit by redemptions must
sell, regardless of the stocks' merits. Lou (*RFS* 2012) and Coval & Stafford (*JFE* 2007) show this flow-induced
trading (FIT) pushes prices in the quarter and partly reverses afterwards, a forced, non-informational flow.

**Design.** SEC N-PORT bulk data sets 2019Q4–2026Q3 (`collect_nport.py`): for each fund report, net flow over its
three months as a fraction of start-of-period assets × its dollar holding in each US common stock, summed by
ticker and quarter. Scaled by the stock's dollar volume over the quarter. Yahoo prices (`collect_prices.py`). Entry
65 days after quarter-end (all N-PORT filings public by then). Returns relative to the equal-weighted universe;
quintiles within quarter. 2019Q4–2022 vs 2023–.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
F=pd.read_csv("data/fit.csv",parse_dates=["filed_by"])
F=F.groupby(["quarter","ticker"]).agg(fit_usd=("fit_usd","sum"),fund_value_usd=("fund_value_usd","sum"),n_funds=("n_funds","sum"),filed_by=("filed_by","max")).reset_index()
F["ticker"]=F.ticker.str.replace(".","-")
P=pd.read_csv("data/prices.csv",parse_dates=["date"]).drop_duplicates(["date","ticker"])
A=P.pivot(index="date",columns="ticker",values="adj_close"); C=P.pivot(index="date",columns="ticker",values="close"); V=P.pivot(index="date",columns="ticker",values="volume")
A=A.where(A>0); C=C.where(C>0)
cal=A.index; LA=np.log(A)
RD=LA.diff(); RD=RD.where(RD.abs()<0.5)              # drop bad prints / unadjusted splits
LA=RD.fillna(0).cumsum().where(A.notna())
DV=(C*V).rolling(63,min_periods=40).mean()
EW=RD.mean(1).fillna(0).cumsum()                                  # equal-weighted universe log index
rows=[]
for q,g in F.groupby("quarter"):
    qe=pd.Period(q,"Q").end_time.normalize()
    if qe>cal[-1]: continue
    i_qe=cal.searchsorted(qe,side="right")-1; i_qs=cal.searchsorted(pd.Period(q,"Q").start_time)-1
    i_in=cal.searchsorted(qe+pd.Timedelta(days=65))           # entry: ~65 days after quarter end (all filings public by then)
    for r in g.itertuples():
        t=r.ticker
        if t not in A or r.fund_value_usd<5e7: continue
        a=LA[t].values; adv=DV[t].iloc[i_qe]
        if not adv>0 or np.isnan(a[i_qs]) or np.isnan(a[i_qe]): continue
        def fwd(k):
            j=i_in+k
            return (a[j]-a[i_in])-(EW.iloc[j]-EW.iloc[i_in]) if j<len(cal) and not np.isnan(a[j]) and not np.isnan(a[i_in]) else np.nan
        rows.append({"q":q,"t":t,"fit_adv":r.fit_usd/(adv*63),"fit_hold":r.fit_usd/r.fund_value_usd,
                     "ret_q":((a[i_qe]-a[i_qs])-(EW.iloc[i_qe]-EW.iloc[i_qs]))*1e4 if i_qs>=0 else np.nan,
                     "gap":((a[i_in]-a[i_qe])-(EW.iloc[i_in]-EW.iloc[i_qe]))*1e4 if i_in<len(cal) else np.nan,
                     "f3m":fwd(63)*1e4,"f6m":fwd(126)*1e4,"f12m":fwd(252)*1e4})
E=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan); E["era"]=np.where(E.q<"2023Q1","2019Q4-2022","2023-")
E["x"]=E.fit_adv.clip(E.fit_adv.quantile(.01),E.fit_adv.quantile(.99))
E["qq"]=E.groupby("q").x.transform(lambda s: pd.qcut(s.rank(method="first"),5,labels=False)+1)
print(len(E),"stock-quarters;", E.q.nunique(),"quarters; median stocks/quarter", int(E.groupby("q").size().median()))
for era in ["2019Q4-2022","2023-"]:
    d=E[E.era==era]
    g=d.groupby(["q","qq"])[["ret_q","gap","f3m","f6m","f12m"]].mean().unstack()
    print(era)
    print("   quintile means (1=outflow-driven selling, 5=inflow buying):"); print(d.groupby("qq")[["ret_q","gap","f3m","f6m","f12m"]].mean().round(0).T.to_string())
    for c in ["ret_q","gap","f3m","f6m","f12m"]:
        x=(g[c][5]-g[c][1]).dropna(); print(f"   Q5-Q1 {c}: {x.mean():+.0f}bp (t {x.mean()/x.std()*np.sqrt(len(x)):+.2f}, nq {len(x)})")

""")
md(r"""
**Interpretation.**
- **Contemporaneous pressure is visible:** stocks with the most inflow-driven buying beat those with the most
  outflow-driven selling within the same quarter (+139bp, t 1.1; then +207bp, t 2.7), with roughly monotone quintiles.
  Part of this is reverse causality (funds holding winners attract flows).
- **Nothing is left to trade once the data is public.** From 65 days after quarter-end, Q5−Q1 returns over 3, 6 and
  12 months are flat in both periods (|t| ≤ 0.8). The reversal documented in pre-2010 data is gone, or too slow and
  noisy to see in 29 quarters.

**Caveat found later:** iShares (and possibly other) funds report CUSIPs without tickers, so their holdings were
left out of FIT here (see `RussellStock Pitch` for the CUSIP→ticker fix). The test covers the non-iShares fund universe.

**Verdict: not tradable with public N-PORT timing.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
