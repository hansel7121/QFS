"""Generate eda.ipynb: month-end FX hedge rebalancing at the London 4pm fix (Melvin & Prins 2015)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# FXHedge Pitch EDA: do equity hedgers move EUR/USD into the month-end fix?

**The mechanism.** Melvin & Prins (*J. Int. Money & Finance* 2015): foreign investors who hold US equities and hedge
the dollar must resize their hedges at month-end, at the WM/R London 16:00 fix. After a month in which US equities rose
(more USD exposure), they sell more USD forward, so the USD should weaken (EUR/USD rise) into 16:00 on the last
business day. Then it partly reverses.

**Design.** EUR/USD 5-minute bars, London time, 2020-03 → 2026-10 (Twelve Data, `collect_fx.py`). Signal = SPY
month-to-date return (and SPY minus EFA), known at the previous close. Outcome = EUR/USD 13:00 → 16:00 and 16:00 →
17:00 on the last business day. Placebo: the 15th trading day. The USD/JPY version (Tokyo data in `Gotobi/`, 52
month-ends) gave a wrong-signed, insignificant slope (t +0.8).
""")
code(r"""
import pandas as pd, numpy as np, statsmodels.api as sm
x=pd.read_csv("data/EURUSD_5m.csv",parse_dates=["datetime"]).sort_values("datetime")
x["d"]=x.datetime.dt.normalize(); x["hm"]=x.datetime.dt.strftime("%H:%M")
L=np.log(x.pivot_table(index="d",columns="hm",values="close")); L=L[L.index.dayofweek<5]
px=pd.read_csv("../TOM Pitch/data/etfs.csv",parse_dates=["date"]).pivot(index="date",columns="ticker",values="adj_close")
spy, efa = px.SPY, px.EFA
ym=L.index.to_period("M")
last=pd.Series(L.index,index=L.index).groupby(ym).transform("max")==pd.Series(L.index,index=L.index)
def mtd(s,d):
    m=d.to_period("M"); a=s[s.index<m.start_time].iloc[-1]; b=s[s.index<d].iloc[-1]; return np.log(b/a)*100
rows=[]
for kind,mask in [("month-end",last.values),("placebo mid-month (15th trading day)",(pd.Series(np.arange(len(L)),index=L.index).groupby(ym).transform(lambda s:s-s.min()+1)==15).values)]:
    for d in L.index[mask]:
        try:
            rows.append({"kind":kind,"d":d,"us":mtd(spy,d),"us_rel":mtd(spy,d)-mtd(efa,d),
                         "into":(L.loc[d,"15:55"]-L.loc[d,"13:00"])*1e4,"after":(L.loc[d,"17:00"]-L.loc[d,"16:00"])*1e4})
        except Exception: pass
D=pd.DataFrame(rows).dropna()
for kind in D.kind.unique():
    d=D[D.kind==kind]
    for sig in ["us","us_rel"]:
        f=sm.OLS(d.into,sm.add_constant(d[sig])).fit(cov_type="HC1"); g=sm.OLS(d.after,sm.add_constant(d[sig])).fit(cov_type="HC1")
        print(f"{kind:36s} n {len(d)} EUR/USD 13:00->16:00 ~ {sig:6s}: {f.params[sig]:+.2f} bp per 1% (t {f.tvalues[sig]:+.2f}) | 16:00->17:00: {g.params[sig]:+.2f} (t {g.tvalues[sig]:+.2f})")
# trade: on month-end, at 13:00 London go long EUR/USD if us_rel>0 (sell USD with hedgers), short if <0; exit 16:00
d=D[D.kind=="month-end"]; pnl=np.sign(d.us_rel)*d.into-1.0
print(f"trade (net 1bp): mean {pnl.mean():+.1f}bp t {pnl.mean()/pnl.std()*np.sqrt(len(pnl)):+.2f} hit {(pnl>0).mean():.0%} n {len(pnl)}")

""")
md(r"""
**Interpretation.**
- **No month-end hedge flow at the fix.** The slope of EUR/USD into 16:00 on month-to-date US equity returns is zero
  (t −0.6; +0.3 on the US-minus-EAFE return). The mid-month placebo is actually stronger (t ≈ 1.9). The fade-the-hedger
  trade loses (−3bp, t −0.9).
- Hedgers spread their rebalancing (rolling forwards, other fixes), or the effect documented for 2000s data has been
  arbitraged.

**Verdict: dead.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
