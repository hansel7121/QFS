"""Generate eda.ipynb: stock-level Russell reconstitution flows (IWM/IWB membership changes from N-PORT holdings)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# RussellStock Pitch EDA: do Russell recon adds and deletes move as index funds trade them?

**The mechanism.** At the June reconstitution (close of the last Friday of June), Russell 2000 funds must buy
names entering the index (downgrades from the Russell 1000 and new names) and sell those leaving (upgrades to the
Russell 1000, drops). The preliminary lists are public from late May. Because R2000 funds hold a much bigger share of
a small stock than R1000 funds, downgrades are net bought and upgrades net sold.

**Design.** IWM and IWB holdings on March 31 and June 30 from SEC N-PORT bulk files (`collect_holdings.py`; iShares
reports CUSIPs only, mapped to tickers via other funds in the same file). Groups by membership change. Returns vs IWM
(prices from `FundFlow Pitch/data/prices.csv`): rank day (Apr 30) → June 1, June 1 → recon close, the recon day, 20
days after. 2020–22 vs 2023–.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
H=pd.read_csv("data/holdings.csv",parse_dates=["report_date"])
P=pd.read_csv("../FundFlow Pitch/data/prices.csv",parse_dates=["date"]).drop_duplicates(["date","ticker"])
A=P.pivot(index="date",columns="ticker",values="adj_close"); A=A.where(A>0); cal=A.index
LA=np.log(A); RD=LA.diff(); RD=RD.where(RD.abs()<0.5); LA=RD.fillna(0).cumsum().where(A.notna())
iwm=LA["IWM"]
rows=[]
for y in range(2020,2027):
    def snap(series,month):
        d=H[(H.series==series)&(H.report_date.dt.year==y)&(H.report_date.dt.month==month)]
        return set(d.ticker), d.set_index("ticker").value_usd
    m3,_=snap("IWM",3); j6,v6=snap("IWM",6); b3,_=snap("IWB",3); b6,_=snap("IWB",6)
    if not m3 or not j6: print(y,"missing reports"); continue
    jun=cal[(cal.year==y)&(cal.month==6)]; R=[d for d in jun if d.weekday()==4][-1]
    iR=cal.get_loc(R); i0=cal.searchsorted(pd.Timestamp(y,6,1)); ir=cal.searchsorted(pd.Timestamp(y,4,30))
    groups={"down: R1000->R2000 (IWM buys)":(j6-m3)&b3,"new into R2000 (IWM buys)":(j6-m3)-b3,
            "up: R2000->R1000 (IWM sells)":(m3-j6)&b6,"dropped from R2000 (IWM sells)":(m3-j6)-b6,"stay in R2000":m3&j6}
    for g,ts in groups.items():
        for t in ts:
            if t not in LA: continue
            a=LA[t]
            def w(i,j): 
                return ((a.iloc[j]-a.iloc[i])-(iwm.iloc[j]-iwm.iloc[i]))*1e4 if j<len(cal) and not np.isnan(a.iloc[i]) and not np.isnan(a.iloc[j]) else np.nan
            rows.append({"y":y,"group":g,"t":t,"rank_to_jun":w(ir,i0),"jun_to_recon":w(i0,iR),"recon_day":w(iR-1,iR),"post20":w(iR,iR+20)})
E=pd.DataFrame(rows)
print(E.groupby(["group"]).size())
for era,m in [("2020-2022",E.y<=2022),("2023-",E.y>=2023)]:
    print("\n",era)
    for g,d in E[m].groupby("group"):
        if g=="stay in R2000": d=d.sample(min(len(d),2000),random_state=0)
        print(f"  {g:32s} n {len(d):4d} | "+" | ".join(f"{c} {d[c].mean():+5.0f} (t {d[c].mean()/d[c].sem():+.1f})" for c in ["rank_to_jun","jun_to_recon","recon_day","post20"]))

""")
md(r"""
**Interpretation.**
- **Signs flip between eras.** On the recon day, names IWM must buy fell in 2020–22 (downgrades −350bp, new adds
  −276bp) but rose in 2023– (+200 / +120bp). Names IWM must sell did the opposite. The June 1 → recon window shows
  the same instability (drops +695bp, then −1,290bp).
- **The t-stats overstate the evidence.** Every event in a year shares one recon date, so each era has only 3–4
  independent observations. Market-wide small-cap swings on those days dominate.

**Verdict: not tradable / not identifiable** with one event a year and free data.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
