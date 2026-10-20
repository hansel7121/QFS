"""Generate eda.ipynb: commodity index roll pressure (GSCI roll, business days 5-9) in WTI and natural gas calendar spreads."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Roll Pitch EDA: does the commodity index roll still push the front month?

**The mechanism.** Commodity index funds (S&P GSCI, BCOM) roll their futures on a published schedule: the GSCI rolls
20% a day on the 5th–9th business days of the month, selling the front contract and buying the next. USO/UNG used to
roll over 4 days about two weeks before expiry. A CFTC study put USO's rolls at times above 10% of NYMEX volume, all
predictable. Mou (2011) shows that front-running the GSCI roll was very profitable before 2010. The prediction is that
the second-minus-front spread (contango) widens during business days 5–9.

**Data.** EIA daily settlements of WTI and Henry Hub contracts 1–2 (`collect.py`; the EIA stopped the series in
2024-04). The days the front contract rolls over are dropped (series switch), using exchange expiry rules (CL: 3
business days before the 25th of the prior month; NG: 3 business days before the delivery month). Eras: 1994–2004,
2005–11, 2012–24 (after Mou).
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from scipy import stats
from pandas.tseries.holiday import USFederalHolidayCalendar
BD=pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
def load(s):
    x=pd.read_excel(f"data/{s}.xls",sheet_name=1,skiprows=2); x.columns=["date",s]; return x.set_index("date")[s]
def last_trade(com,y,m):                      # m = delivery month
    if com=="CL":
        ref=pd.Timestamp(y,m,1)-pd.offsets.MonthBegin(1)+pd.Timedelta(days=24)   # 25th of prior month
        k=3 if ref==ref+0*BD and BD.is_on_offset(ref) else 4
        return ref-k*BD
    first=pd.Timestamp(y,m,1); return first-3*BD                                # NG
res={}
for com in ["CL","NG"]:
    p="RCLC" if com=="CL" else "RNGC"
    D=pd.concat([load(f"{p}{k}") for k in (1,2)],axis=1).dropna(); D.columns=["c1","c2"]; D=D[(D>0).all(1)]
    D["s"]=np.log(D.c2/D.c1)*1e4; cal=D.index
    lt=sorted({last_trade(com,y,m) for y in range(1990,2025) for m in range(1,13)})
    after=set()
    for d in lt:
        i=cal.searchsorted(d,side="right")
        if i<len(cal): after.add(cal[i])                    # first day the series refers to the next contract
    ds=D.s.diff(); ds[cal.isin(after)]=np.nan
    bd=pd.Series(np.arange(len(cal)),index=cal).groupby(cal.to_period("M")).transform(lambda x:x-x.min()+1)
    # days to front expiry
    nxt=pd.Series([lt[np.searchsorted(lt,d)] if np.searchsorted(lt,d)<len(lt) else pd.NaT for d in cal],index=cal)
    dte=pd.Series([len(pd.bdate_range(d,e))-1 if pd.notna(e) else np.nan for d,e in zip(cal,nxt)],index=cal)
    out=pd.DataFrame({"ds":ds,"bd":bd,"dte":dte})
    res[com]=out
    print(f"\n{com}: switches excluded {len(after)}")
    for per,m in [("1994-2004",(cal>="1994")&(cal<"2005")),("2005-2011",(cal>="2005")&(cal<"2012")),("2012-2024",cal>="2012")]:
        o=out[m].dropna()
        win=o[o.bd.between(5,9)].ds; oth=o[~o.bd.between(5,9)].ds
        t=stats.ttest_ind(win,oth,equal_var=False)[0]
        g=o.groupby(o.dte.clip(upper=25)).ds.mean()
        print(f"  {per}: BD5-9 {win.mean():+.2f} bp/day vs other {oth.mean():+.2f} (t {t:+.2f}) | by days-to-expiry: "+" ".join(f"{int(k)}:{v:+.1f}" for k,v in g.loc[::-1].items() if k<=22))
print("\n==== CL calendar spread trade: long C2 / short C1 from close BD4 to close BD9 (returns of BD5..BD9), cost 2bp RT")
o=res["CL"].copy(); cal=o.index
def monthly(lo,hi,cost=2.0):
    m=o.bd.between(lo,hi)
    x=o.ds.where(m,0).fillna(0)
    mon=x.groupby(cal.to_period("M")).sum()-cost*(o.bd.groupby(cal.to_period("M")).max()>=hi).astype(float)
    return mon
for nm,(lo,hi) in {"GSCI window BD5-9":(5,9),"placebo BD1-4":(1,4),"placebo BD10-14":(10,14),"placebo BD15-19":(15,19)}.items():
    mon=monthly(lo,hi)
    line=f"{nm:18s}: "
    for per,m in [("1994-2004",(mon.index.year>=1994)&(mon.index.year<2005)),("2005-11",(mon.index.year>=2005)&(mon.index.year<2012)),("2012-24",mon.index.year>=2012)]:
        x=mon[m]; line+=f"{per} {x.mean():+5.1f}bp/mo Sharpe {x.mean()/x.std()*np.sqrt(12):+.2f} t {x.mean()/x.std()*np.sqrt(len(x)):+.2f} | "
    print(line)
mon=monthly(5,9)
print("GSCI window by year (bp):", mon.groupby(mon.index.year).sum().round(0).loc[2005:].astype(int).to_dict())
print("hit rate 2012-24:", f"{(mon[mon.index.year>=2012]>0).mean():.0%}", "worst month", round(mon[mon.index.year>=2012].min(),0))
# contango vs backwardation regimes (known at BD4: spread level)
s_level=None
print("\n==== conditional on contango at BD4 (known): spread level log(C2/C1) > 0")
D=pd.concat([load(f"RCLC{k}") for k in (1,2)],axis=1).dropna(); D.columns=["c1","c2"]; D=D[(D>0).all(1)]
lev=(np.log(D.c2/D.c1)*1e4).reindex(cal)
bd4=lev[o.bd==4]; cont=pd.Series((bd4>0).values,index=bd4.index.to_period("M"))
mon=monthly(5,9)
for per,m in [("1994-2004",(mon.index.year>=1994)&(mon.index.year<2005)),("2005-11",(mon.index.year>=2005)&(mon.index.year<2012)),("2012-24",mon.index.year>=2012),("2012-24 ex-2020",(mon.index.year>=2012)&(mon.index.year!=2020))]:
    for c in [True,False]:
        x=mon[m & cont.reindex(mon.index).fillna(False).eq(c).values]
        print(f"{per:16s} {'contango' if c else 'backwardation':13s}: {x.mean():+6.1f}bp/mo t {x.mean()/x.std()*np.sqrt(len(x)):+.2f} n {len(x)} hit {(x>0).mean():.0%}")

""")
md(r"""
**Interpretation.**
- **Daily data says yes for WTI:** the spread widens +3.7 to +5.9bp/day in the roll window against −2.1 to −2.4bp
  on other days (t 2.7–2.8 in 2005–11 and 2012–24). Natural gas shows no roll effect after 2004.
- **Monthly trading data says it is too lumpy.** Holding the calendar spread over business days 5–9 every month nets
  +16bp/month in 2012–24 (Sharpe 0.37, t 1.3). The total is dominated by two contango crises (2009: +1,660bp;
  2020: +1,350bp), and ex-2020 contango months earn about +6bp. The placebo windows are not clean either
  (BD1–4 positive pre-2012, BD10–14 strongly negative, reflecting the expiry cycle).

**Verdict: not tradable outside storage crises.** The roll still leaves a footprint, but at ordinary times it is
about 10bp a month, within the noise.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
