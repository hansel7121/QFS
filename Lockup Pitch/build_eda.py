"""Generate eda.ipynb: returns around IPO lockup expiry (180 days after pricing), with an IPO-age placebo."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Lockup Pitch EDA: do IPO stocks fall into the lockup expiry?

**The mechanism.** Insiders and pre-IPO holders are barred from selling for (usually) 180 days. At expiry, supply
that was locked becomes available on a date known from the IPO day. Field & Hanka (*JF* 2001) find about −1 to −3%
abnormal returns around expiry.

**Design.** US IPOs 2012–2026 from the Nasdaq IPO calendar (SPACs dropped), Yahoo prices (`collect_ipos.py`;
survivorship bias: delisted IPOs are missing). Expiry = first trading day 180 calendar days after pricing. Returns
relative to IWM. The placebo is the same 10-day window ending at other ages since the IPO, because recent IPOs
underperform in general.
""")
code(r"""
import pandas as pd, numpy as np
ip=pd.read_csv("data/ipos.csv",parse_dates=["priced"])
p=pd.read_csv("data/prices.csv",parse_dates=["date"])
A=p.pivot(index="date",columns="ticker",values="adj_close"); V=p.pivot(index="date",columns="ticker",values="volume")
R=np.log(A).diff()*1e4; cal=R.index
b=pd.read_csv("../TOM Pitch/data/etfs.csv",parse_dates=["date"]).pivot(index="date",columns="ticker",values="adj_close").IWM
iwm=(np.log(b).diff()*1e4).reindex(cal)
if "IWM" in R: R=R.drop(columns="IWM")
X=R.sub(iwm,axis=0)
print(len(ip),"IPOs;", ip.ticker.str.replace(".","-").isin(R.columns).sum(),"with prices")
rows=[]
for r in ip.itertuples():
    t=str(r.ticker).replace(".","-")
    if t not in X: continue
    first=A[t].first_valid_index()
    if first is None or abs((first-r.priced).days)>10: continue        # ticker recycled / wrong security
    L=cal.searchsorted(r.priced+pd.Timedelta(days=180))               # lockup expiry (standard 180 days)
    if L<10 or L+21>=len(cal): continue
    x=X[t].values; v=V[t].values
    if np.isnan(x[L-10:L+21]).sum()>3: continue
    vr=np.nanmean(v[L:L+3])/np.nanmean(v[L-30:L-10]) if np.nanmean(v[L-30:L-10])>0 else np.nan
    rows.append({"t":t,"date":cal[L],"value":r.value_usd,"pre10":np.nansum(x[L-10:L]),"pre5":np.nansum(x[L-5:L]),"d0_2":np.nansum(x[L:L+3]),"post10":np.nansum(x[L:L+10]),"post20":np.nansum(x[L:L+20]),"vol_ratio":vr,
                 "ret_since_ipo":np.log(A[t].iloc[L-11]/A[t].loc[first])})
E=pd.DataFrame(rows); E["era"]=np.where(E.date<"2020-01-01","2012-19","2020-")
print(E.groupby("era").agg(n=("t","size"),vol_ratio=("vol_ratio","median")))
for era in ["2012-19","2020-"]:
    d=E[E.era==era]
    w=lambda c: d[c].clip(d[c].quantile(.01),d[c].quantile(.99))
    print(era,{c:f"{w(c).mean():+.0f} (t {w(c).mean()/w(c).sem():+.1f}) med {d[c].median():+.0f}" for c in ["pre10","pre5","d0_2","post10","post20"]})
    up=d[d.ret_since_ipo>0]; dn=d[d.ret_since_ipo<=0]
    print("   winners since IPO pre5", f"{up.pre5.mean():+.0f} (n {len(up)})", "| losers", f"{dn.pre5.mean():+.0f} (n {len(dn)})")
print("\n==== placebo: 10-day windows ending at other ages (calendar days since IPO pricing)")
ages=[90,120,150,180,210,240,300]
res={a:{"2012-19":[],"2020-":[]} for a in ages}
for r in ip.itertuples():
    t=str(r.ticker).replace(".","-")
    if t not in X: continue
    first=A[t].first_valid_index()
    if first is None or abs((first-r.priced).days)>10: continue
    x=X[t].values
    for a in ages:
        L=cal.searchsorted(r.priced+pd.Timedelta(days=a))
        if L<10 or L+1>=len(cal) or np.isnan(x[L-10:L]).sum()>2: continue
        res[a]["2012-19" if cal[L]<pd.Timestamp("2020-01-01") else "2020-"].append(np.nansum(x[L-10:L]))
for a in ages:
    print(a, {e:f"{np.mean(v):+.0f} (t {np.mean(v)/np.std(v)*np.sqrt(len(v)):+.1f}, n {len(v)})" for e,v in res[a].items()})

""")
md(r"""
**Interpretation.**
- **The 180-day window looks bad** (−201bp pre-10 in 2012–19, t −4.3; −320bp in 2020–, t −5.9), **but so does every
  other age**: day 90 (−178 / −556), day 150 (−194 / −412), day 210 (−165 / −434), and through day 300 in 2020–.
  It is the well-known long-run IPO underperformance (amplified in the 2020–21 IPO wave), not lockup supply.
- The event-day volume ratio is only about 1.0–1.2×, so many lockups are released early or staggered; there is no
  sharp supply date.

**Verdict: no lockup-specific effect.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
