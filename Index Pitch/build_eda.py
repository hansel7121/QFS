"""Generate eda.ipynb: S&P 500 additions and deletions around the effective date (index-fund trade at the close before)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Index Pitch EDA: do S&P 500 additions give back and deletions rebound after the effective date?

**The mechanism.** Index funds must buy additions and sell deletions at the close before the effective date D,
whatever the price. If the close is pushed, it should reverse after D: additions fall, deletions rebound
(Chen, Noronha & Singal 2004; the effect has reportedly shrunk as it became known).

**Design.** Changes from the Wikipedia change log (Feb-2026 revision, `collect_changes.py`), Yahoo prices.
Returns relative to SPY (additions) or MDY (deletions, which usually move to the S&P 400). Windows: pre = D−5…D−1
(announcement run), post = D…D+k. "Verified" events require volume on D−1 ≥ 3× normal (the index trade), which removes
recycled tickers whose Yahoo history belongs to another company. **Survivorship:** deleted stocks later acquired have
no Yahoo data.
""")
code(r"""
import pandas as pd, numpy as np
ch=pd.read_csv("data/changes.csv",parse_dates=["date"])
p=pd.read_csv("data/prices.csv",parse_dates=["date"])
A=p.pivot(index="date",columns="ticker",values="adj_close"); V=p.pivot(index="date",columns="ticker",values="volume")
R=np.log(A).diff()*1e4; cal=A.index
print("changes by 5y:",ch.groupby(ch.date.dt.year//5*5).size().to_dict())
rows=[]
for r in ch.itertuples():
    i=cal.searchsorted(r.date)          # effective day D (first day in index / out)
    if i<30 or i+25>=len(cal): continue
    for side,tk,bm in [("del",r.removed,"MDY"),("add",r.added,"SPY")]:
        if not isinstance(tk,str): continue
        tk=tk.replace(".","-")
        if tk not in R: continue
        x=(R[tk]-R[bm]).values
        if np.isnan(x[i-6:i+21]).sum()>2: continue
        vol=V[tk].values; vr=vol[i-1]/np.nanmean(vol[i-25:i-5]) if np.nanmean(vol[i-25:i-5])>0 else np.nan
        rows.append({"date":cal[i],"side":side,"tk":tk,"pre":np.nansum(x[i-5:i]),"post1":np.nansum(x[i:i+1]),"post5":np.nansum(x[i:i+5]),"post10":np.nansum(x[i:i+10]),"post20":np.nansum(x[i:i+20]),"vol_ratio_Dm1":vr})
E=pd.DataFrame(rows); E["era"]=np.where(E.date<"2013-01-01","2000-12","2013-")
print(E.groupby(["side","era"]).agg(n=("pre","size"),vol_ratio=("vol_ratio_Dm1","median")).to_string())
for side in ["del","add"]:
    for era in ["2000-12","2013-"]:
        d=E[(E.side==side)&(E.era==era)]
        print(side,era,{c:f"{d[c].mean():+.0f} (t {d[c].mean()/d[c].sem():+.1f})" for c in ["pre","post1","post5","post10","post20"]})
print("\n==== verified events (index-day volume >= 3x normal)")
V2=E[E.vol_ratio_Dm1>=3].copy(); V2["era"]=pd.cut(V2.date.dt.year,[2004,2016,2020,2026],labels=["2005-16","2017-20","2021-"])
for side in ["add","del"]:
    for era in V2.era.cat.categories:
        d=V2[(V2.side==side)&(V2.era==era)]
        if len(d)<5: continue
        print(side,era,"n",len(d),{c:f"{d[c].mean():+.0f} (t {d[c].mean()/d[c].sem():+.1f})" for c in ["pre","post5","post10","post20"]})
# trade: short additions at close D-1 (with index funds' buy), cover close D+19, SPY hedge, cost 20bp RT
a=V2[V2.side=="add"].copy(); a["pnl"]=-a.post20-20
for per,m in [("2005-16",a.date<"2017"),("2017-",a.date>="2017"),("2021-",a.date>="2021")]:
    x=a[m].pnl; print(f"short adds 20d net 20bp {per}: {x.mean():+.0f}bp/trade t {x.mean()/x.sem():+.2f} hit {(x>0).mean():.0%} n {len(x)}")
print("by year:",a.groupby(a.date.dt.year).pnl.agg(['mean','count']).round(0).to_dict('index'))

""")
md(r"""
**Interpretation.**
- **Data check matters.** Unverified 2000–12 events have a median event-day volume ratio of about 1.9× (vs 12–19× for
  2013–), so many early tickers are recycled. Only verified events are used for conclusions.
- **Additions: no stable post-inclusion reversal.** 20-day give-back of −54bp (t −0.6) in 2005–16, +41bp in 2017–20,
  and −336bp (t −2.4) in 2021–, driven by a few volatile names (2021, 2023, 2024 averages ±5%). A "short additions
  for 20 days" rule earns +28bp (t 0.3) before 2017 and +316bp (t 2.2) since 2021.
- **Deletions flip sign by era** (+367bp in 2017–20, −374bp in 2021–). No rebound to trade.

**Verdict: not tradable.** The only significant window is the most recent era, and it was not present before.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
