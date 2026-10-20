"""Generate eda.ipynb: buyback blackout windows, do heavy repurchasers outperform when the window reopens after earnings?"""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Buyback Pitch EDA: does buyback demand return when the blackout lifts?

**The mechanism.** Most US companies stop repurchasing shares in a blackout before earnings and restart one to two
days after the release (10b5-1 plans soften this). For a heavy repurchaser, corporate buying is a large, mechanical,
price-insensitive bid that switches **off** before earnings and **on** after. The prediction: heavy repurchasers
outperform light ones in the days after earnings (window reopens) and, more weakly, underperform before.

**Design (no look-ahead).**
* Earnings date E = filing date of the 8-K with Item 2.02 (SEC submissions API), one per quarter. Entry at the close
  of E+1, after the release is public, so the date does not need to be known in advance.
* Buyback intensity = last 10-K's repurchases / market cap at the 10-K filing date, read 20 trading days before E.
* Terciles within each calendar quarter. Returns vs the equal-weighted universe.
* Current S&P 500 members (**survivorship bias**), 2009–2026. In-sample 2009–17, out-of-sample 2018–.

**Data.** `collect_sec.py` (8-K Item 2.02 dates, XBRL `PaymentsForRepurchaseOfCommonStock`, cover-page shares),
prices from `Overnight Pitch/data/panel.csv`.
""")
code(r"""
import pandas as pd, numpy as np, statsmodels.api as sm
p=pd.read_csv("../Overnight Pitch/data/panel.csv",parse_dates=["date"])
A=p.pivot(index="date",columns="ticker",values="adj_close"); C=p.pivot(index="date",columns="ticker",values="close")
R=A.pct_change()*1e4; spy=R.pop("SPY"); C=C.drop(columns="SPY")
cal=R.index; ew=R.mean(1)
AR=R.sub(ew,axis=0)                                  # vs equal-weight universe
e=pd.read_csv("data/earnings.csv",parse_dates=["date"]).sort_values(["ticker","date"])
e=e[e.date>="2008-01-01"]
e=e[e.groupby("ticker").date.diff().dt.days.fillna(999)>45]       # one per quarter
b=pd.read_csv("data/buybacks.csv",parse_dates=["fy_end","filed"]).dropna(subset=["repurchases"])
# intensity known from filed date: repurchases / market cap at filing
rows=[]
for r in b.itertuples():
    if r.ticker not in C or pd.isna(r.shares) or r.shares<=0: continue
    i=cal.searchsorted(r.filed)
    if i>=len(cal): continue
    mc=C[r.ticker].iloc[i]*r.shares
    if mc>0: rows.append((r.ticker,cal[i],r.repurchases/mc))
I=pd.DataFrame(rows,columns=["ticker","date","bbi"]).pivot_table(index="date",columns="ticker",values="bbi").reindex(cal).ffill(limit=300)
# event study: for each earnings date E, AR windows
ev=[]
Arr=AR.values; col={t:j for j,t in enumerate(AR.columns)}
for r in e.itertuples():
    if r.ticker not in col: continue
    i=cal.searchsorted(r.date)          # 8-K filing day = announcement day (often after close -> reaction day i or i+1)
    if i<25 or i+25>=len(cal): continue
    bbi=I[r.ticker].iloc[i-20] if r.ticker in I else np.nan
    if pd.isna(bbi): continue
    j=col[r.ticker]; x=Arr[:,j]
    ev.append({"ticker":r.ticker,"date":cal[i],"bbi":bbi,"pre":np.nansum(x[i-10:i]),"post_2_10":np.nansum(x[i+2:i+11]),"post_2_20":np.nansum(x[i+2:i+21]),"day01":np.nansum(x[i:i+2])})
E=pd.DataFrame(ev); E["q"]=E.groupby(E.date.dt.to_period("Q")).bbi.transform(lambda s: pd.qcut(s.rank(method="first"),3,labels=False)+1)
E["is"]=E.date<"2018-01-01"
print(len(E),"events")
for per,m in [("IS 2009-17",E["is"]),("OOS 2018-",~E["is"])]:
    d=E[m]; g=d.groupby("q")[["pre","day01","post_2_10","post_2_20"]].mean()
    hl={c:(d[d.q==3][c].mean()-d[d.q==1][c].mean()) for c in g.columns}
    # t via quarter clustering: per-quarter H-L
    qq=d.groupby([d.date.dt.to_period("Q"),"q"])[["pre","post_2_10","post_2_20"]].mean().unstack()
    ts={c:(qq[c][3]-qq[c][1]) for c in ["pre","post_2_10","post_2_20"]}
    print(per); print(g.round(1).to_string()); print("  H-L:",{c:f"{v.mean():+.1f} (t {v.mean()/v.std()*np.sqrt(len(v)):+.2f}, nq {len(v)})" for c,v in ts.items()})
print("\n==== placebo windows & controls")
ev2=[]
mom=(A.shift(21)/A.shift(252)-1)
for r in e.itertuples():
    if r.ticker not in col: continue
    i=cal.searchsorted(r.date)
    if i<260 or i+40>=len(cal): continue
    bbi=I[r.ticker].iloc[i-20] if r.ticker in I else np.nan
    if pd.isna(bbi): continue
    j=col[r.ticker]; x=Arr[:,j]
    ev2.append({"date":cal[i],"ticker":r.ticker,"bbi":bbi,"post_2_10":np.nansum(x[i+2:i+11]),"plac_25_33":np.nansum(x[i+25:i+34]),"plac_m20_m12":np.nansum(x[i-20:i-11]),
                "day01":np.nansum(x[i:i+2]),"mom":mom[r.ticker].iloc[i],"lmc":np.log(C[r.ticker].iloc[i]+1e-9)})
E2=pd.DataFrame(ev2); E2["is"]=E2.date<"2018-01-01"
E2["q"]=E2.groupby(E2.date.dt.to_period("Q")).bbi.transform(lambda s: pd.qcut(s.rank(method="first"),3,labels=False)+1)
for per,m in [("IS",E2["is"]),("OOS",~E2["is"])]:
    d=E2[m]
    qq=d.groupby([d.date.dt.to_period("Q"),"q"])[["post_2_10","plac_25_33","plac_m20_m12"]].mean().unstack()
    print(per,{c:f"{(qq[c][3]-qq[c][1]).mean():+.1f} (t {(qq[c][3]-qq[c][1]).mean()/(qq[c][3]-qq[c][1]).std()*np.sqrt(len(qq)):+.2f})" for c in ["post_2_10","plac_25_33","plac_m20_m12"]})
    # regression with controls, quarter FE via demeaning
    d=d.dropna(subset=["mom"]).copy()
    d["hi"]=(d.q==3).astype(float); d["lo"]=(d.q==1).astype(float)
    for c in ["post_2_10","day01","mom","hi","lo"]:
        d[c+"_dm"]=d[c]-d.groupby(d.date.dt.to_period("Q"))[c].transform("mean")
    f=sm.OLS(d.post_2_10_dm, d[["hi_dm","lo_dm","day01_dm","mom_dm"]]).fit(cov_type="cluster",cov_kwds={"groups":d.date.dt.to_period("Q").astype(str).factorize()[0]})
    print("   with controls (day01 surprise proxy, momentum, quarter FE): hi-lo =",f"{f.params.hi_dm-f.params.lo_dm:+.1f}", "t(hi)",f"{f.tvalues.hi_dm:+.2f}","t(lo)",f"{f.tvalues.lo_dm:+.2f}", "| day01 coef",f"{f.params.day01_dm:+.3f} (t {f.tvalues.day01_dm:+.1f})")
# daily portfolio: long heavy in E+2..E+10 window, short light in window
n=len(cal); L=np.zeros((n,len(AR.columns))); S=np.zeros_like(L)
for r in E2.itertuples():
    i=cal.searchsorted(r.date); j=col[r.ticker]
    if r.q==3: L[i+2:i+11,j]=1
    if r.q==1: S[i+2:i+11,j]=1
Lw=L/np.maximum(L.sum(1,keepdims=True),1); Sw=S/np.maximum(S.sum(1,keepdims=True),1)
Rv=np.nan_to_num(AR.values)
for c in [0,10]:   # c bp round trip per name; turnover: each name enters/exits once per event (9 days)
    pnl=(Lw*Rv).sum(1)-(Sw*Rv).sum(1)
    turn=(np.abs(np.diff(Lw,axis=0,prepend=0)).sum(1)+np.abs(np.diff(Sw,axis=0,prepend=0)).sum(1))/2
    pnl=pd.Series(pnl-c*turn,index=cal); act=pd.Series((L.sum(1)>0)&(S.sum(1)>0),index=cal)
    for per,m in [("IS 2009-17",(cal>="2009")&(cal<"2018")),("OOS 2018-",cal>="2018")]:
        x=pnl[m&act.values]
        print(f"daily L/S cost {c}bp RT: {per} {x.mean()*252/100:+.1f}%/yr(active) Sharpe {x.mean()/x.std()*np.sqrt(252):+.2f} t {x.mean()/x.std()*np.sqrt(len(x)):+.2f} active days {len(x)}")
print("\n==== diagnostics")
for per,m in [("IS",E2["is"]),("OOS",~E2["is"])]:
    d=E2[m]; h=d[d.q==3].post_2_10; l=d[d.q==1].post_2_10
    print(per,"pooled H-L",f"{h.mean()-l.mean():+.1f}","median H-L",f"{h.median()-l.median():+.1f}", "n",len(h),len(l))
    # winsorized
    w=d.post_2_10.clip(d.post_2_10.quantile(.01),d.post_2_10.quantile(.99)); d2=d.assign(w=w)
    print("   winsor 1% H-L",f"{d2[d2.q==3].w.mean()-d2[d2.q==1].w.mean():+.1f}")
nL=pd.Series(L.sum(1),index=cal); nS=pd.Series(S.sum(1),index=cal)
pnl=pd.Series((Lw*Rv).sum(1)-(Sw*Rv).sum(1),index=cal)
for k in [1,10,30]:
    for per,m in [("IS",(cal>="2009")&(cal<"2018")),("OOS",cal>="2018")]:
        mm=m&(nL>=k).values&(nS>=k).values; x=pnl[mm]
        # event-weighted: weight day by number of names
        wts=(nL+nS)[mm]; xw=(x*wts).sum()/wts.sum()
        print(f"min {k} names/side {per}: days {mm.sum()} mean {x.mean():+.2f}bp/day, names-weighted {xw:+.2f}bp/day, Sharpe {x.mean()/x.std()*np.sqrt(252):+.2f}")
print("\n==== fixed notional per trade (like the Taiwan backtests): P&L = sum over open positions, 1 unit each")
for c in [0,10,20]:
    entL=np.clip(np.diff(L,axis=0,prepend=0),0,None).sum(1); entS=np.clip(np.diff(S,axis=0,prepend=0),0,None).sum(1)
    raw=(L*Rv).sum(1)-(S*Rv).sum(1)-c*(entL+entS)     # c bp per round trip per name, charged at entry
    pn=pd.Series(raw,index=cal)
    for per,m in [("IS 2009-17",(cal>="2009")&(cal<"2018")),("OOS 2018-",cal>="2018"),("2022-",cal>="2022")]:
        x=pn[m]
        print(f"cost {c:2d}bp RT/name: {per}: Sharpe {x.mean()/x.std()*np.sqrt(252):+.2f} t {x.mean()/x.std()*np.sqrt(len(x)):+.2f}  bp per position-trade {x.sum()/((L+S)[m].sum()/9):+.1f}")
print("by year (cost 10):", pd.Series((L*Rv).sum(1)-(S*Rv).sum(1)-10*(np.clip(np.diff(L,axis=0,prepend=0),0,None).sum(1)+np.clip(np.diff(S,axis=0,prepend=0),0,None).sum(1)),index=cal).groupby(cal.year).sum().div(100).round(1).loc[2009:].to_dict())

""")
md(r"""
**Interpretation.**
- **The window-reopening effect is there in both samples.** Heavy minus light repurchasers over E+2…E+10: +35bp
  (t 2.0) in-sample, +32bp (t 2.4) out of sample. The terciles are roughly monotone.
- **It is specific to that window.** Placebo windows (E+25…E+33, E−20…E−12) show H−L ≈ 0 (|t| ≤ 0.6). The blackout
  itself (E−10…E−1) shows no underperformance.
- **Not post-earnings drift.** Controlling for the earnings-day reaction, momentum and quarter fixed effects leaves
  +34bp out of sample (t on the heavy tercile 3.1). In-sample the controlled estimate is weaker (+24bp, t 1.1).
- **Too small to trade.** Per position it is about +12–16bp gross (half the H−L spread on each leg). With
  fixed notional per trade, gross Sharpe is 0.64 in-sample and 0.71 out of sample (t 2.1). After a realistic 10bp
  round trip per name it drops to 0.11 / 0.27. At 20bp it is negative. Equal-weighting by day is worse, because
  off-season days with few names are pure noise.

**Verdict: real, mechanism-consistent, not tradable after costs.** A good slide for "buyback demand moves prices", but
the edge per trade is about the size of large-cap trading costs.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
