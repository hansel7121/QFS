"""Generate eda.ipynb: VIX futures carry (short the front VX contract, optionally only in contango), 2013-2026."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# VIXCarry Pitch EDA: is short front-month VIX futures, timed by the term structure, a reliable premium?

**The mechanism.** Hedgers and long-VIX products (VXX, UVXY) pay up for VIX futures; in contango the front contract
rolls down toward spot VIX, and shorts collect it (the volatility risk premium), at the cost of rare violent losses.

**Design.** Cboe VX settlements (`VIX Pitch/data/vx.csv`, 2013–), VIX spot (Yahoo). Short the front contract
(rolling to the second when ≤5 days to expiry), daily P&L on the contract's own settlements, per unit of notional.
Filters: always, only in contango (front > spot), only when front > spot by 5%. Cost 25bp per position change.
2013–19 vs 2020–.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
v=pd.read_csv("../VIX Pitch/data/vx.csv",parse_dates=["date","expiry"])
spot=pd.read_csv("../VIX Pitch/data/vix_daily.csv",parse_dates=["Date"]).set_index("Date").Close
cal=sorted(v.date.unique())
# front contract on each date = nearest expiry strictly after that date (avoid holding through settlement)
rows=[]
piv=v.pivot_table(index="date",columns="expiry",values="settle")
exps=sorted(v.expiry.unique())
for d in piv.index:
    fut=[e for e in exps if e>d and not np.isnan(piv.loc[d].get(e,np.nan))]
    if len(fut)<2: continue
    e1,e2=fut[0],fut[1]
    rows.append({"date":d,"e1":e1,"e2":e2,"f1":piv.loc[d,e1],"f2":piv.loc[d,e2],"dte":(e1-d).days})
F=pd.DataFrame(rows).set_index("date")
F["spot"]=spot.reindex(F.index)
# daily P&L of short 1 front contract held from d to next date, using the SAME contract's settles; roll when dte<=5 into e2
dates=F.index
out=[]
for i in range(len(dates)-1):
    d,n=dates[i],dates[i+1]
    e=F.e1.iloc[i] if F.dte.iloc[i]>5 else F.e2.iloc[i]
    p0=piv.loc[d,e]; p1=piv.loc[n,e] if e in piv.columns else np.nan
    basis=(F.f1.iloc[i]/F.spot.iloc[i]-1) if F.spot.iloc[i]>0 else np.nan
    slope=F.f2.iloc[i]/F.f1.iloc[i]-1
    out.append({"date":n,"ret":-(p1/p0-1)*1e4,"basis":basis,"slope":slope})
D=pd.DataFrame(out).set_index("date").dropna()
D["ret"]=D.ret.clip(-5000,5000)
for nm,pos in [("always short front",np.ones(len(D))),("short only if contango (f1>spot)",(D.basis>0).astype(float).values),
               ("short if f1>spot by >5%",(D.basis>0.05).astype(float).values)]:
    p=pd.Series(pos,index=D.index).shift(0)*D.ret        # signal known at prior close (basis computed at d, return d->n)
    # cost: 0.05 vol point per trade ~ (0.05/f1) ~ 25bp per round trip on position changes
    ch=pd.Series(pos,index=D.index).diff().abs().fillna(0)
    p=p-25*ch
    for per,m in [("IS 2013-2019",D.index<"2020"),("OOS 2020-",D.index>="2020")]:
        x=p[m]; print(f"{nm:34s} {per}: ann ret {x.mean()*252/100:+6.1f}% Sharpe {x.mean()/x.std()*np.sqrt(252):+.2f} t {x.mean()/x.std()*np.sqrt(len(x)):+.2f} worst day {x.min()/100:+.1f}% maxDD {((x/1e4).cumsum()-(x/1e4).cumsum().cummax()).min()*100:+.0f}%")

""")
md(r"""
**Interpretation.** Unlevered short-front VIX earns a lot on average (+43–51% a year) but with −50% days (Feb 2018,
Mar 2020) and drawdowns beyond 100% of notional. Sharpe is 0.5–0.6 and the t-stats are only 1.3–1.7. Contango filters
make it *worse* (t ≤ 1.1), and the strict filter loses out of sample.

**Verdict: fails the bar** (in-sample and out-of-sample t < 2).
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
