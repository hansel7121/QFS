"""Generate eda.ipynb: quarterly equal-weight resets in SPDR industry ETFs (XBI, XME, XOP, XRT, KRE)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# EqualWeight Pitch EDA: do equal-weight ETFs push their laggards up at the quarterly reset?

**The mechanism.** S&P Select Industry indices (XBI biotech, XME metals & mining, XOP E&P, XRT retail, KRE
regional banks) reset to equal weight after the close of the third Friday of each quarter-end month. The ETFs must buy
names that lagged since the last reset and sell the leaders, on a known date, in small and illiquid names.

**Design.** Today's holdings (SSGA file; survivorship bias), Yahoo prices (`collect.py`). Signal at R−5 = minus the
stock's return relative to its ETF since the last reset (the fraction the fund must buy back). Outcomes, relative to
the ETF: R−5 → R (into the reset close), the reset day, R → R+5 (after). Fama–MacBeth slopes per ETF-quarter and
quintile sorts. 2012–18 vs 2019–. (A dollar-trade/ADV version was dropped: reverse splits in small biotechs make
split-adjusted historical prices useless for holding sizes.)
""")
code(r"""
import pandas as pd, numpy as np
H=pd.read_csv("data/holdings.csv"); P=pd.read_csv("data/prices.csv",parse_dates=["date"])
A=P.pivot(index="date",columns="ticker",values="adj_close"); C=P.pivot(index="date",columns="ticker",values="close"); V=P.pivot(index="date",columns="ticker",values="volume")
LA=np.log(A); cal=A.index; DV=(C*V).rolling(20,min_periods=10).mean()
def third_fri(y,m):
    d=pd.date_range(pd.Timestamp(y,m,1),periods=31,freq="D"); f=d[(d.weekday==4)&(d.month==m)][2]
    return cal[cal.searchsorted(f,side="right")-1]
R=[third_fri(y,m) for y in range(2012,2027) for m in (3,6,9,12) if pd.Timestamp(y,m,1)<=cal[-1]]
R=[r for r in R if r<=cal[-1]]
rows=[]
for e,h in H.groupby("etf"):
    tk=[t for t in h.ticker if t in A]; sh=h.set_index("ticker").shares_held
    for q in range(1,len(R)):
        r0,r1=R[q-1],R[q]; i1=cal.get_loc(r1)
        if i1+6>=len(cal): continue
        sig_d=cal[i1-5]                                         # signal known at close of R-5
        etf_drift=LA[e].loc[sig_d]-LA[e].loc[r0]
        for t in tk:
            a=LA[t]
            if np.isnan(a.loc[r0]) or np.isnan(a.loc[sig_d]) or np.isnan(a.iloc[i1+5]): continue
            drift=(a.loc[sig_d]-a.loc[r0])-etf_drift
            hold=sh[t]*C[t].loc[sig_d]; adv=DV[t].loc[sig_d]
            if not adv>0: continue
            trade=-hold*(np.exp(drift)-1)/np.exp(drift)      # $ to buy (+) / sell (-) to get back to equal weight (approx.)
            rows.append({"etf":e,"t":t,"R":r1,"drift":drift,"trade_adv":trade/adv,
                         "into":((a.iloc[i1]-a.loc[sig_d])-(LA[e].iloc[i1]-LA[e].loc[sig_d]))*1e4,
                         "rday":((a.iloc[i1]-a.iloc[i1-1])-(LA[e].iloc[i1]-LA[e].iloc[i1-1]))*1e4,
                         "after":((a.iloc[i1+5]-a.iloc[i1])-(LA[e].iloc[i1+5]-LA[e].iloc[i1]))*1e4})
E=pd.DataFrame(rows); E["era"]=np.where(E.R<"2019-01-01","IS 2012-18","OOS 2019-")
E["sig"]=-E.drift.clip(-1,1)                      # fraction the fund must buy back (+) or sell (-)
E["adv_r"]=E.groupby(["etf","R"]).trade_adv.transform(lambda x: x)  # placeholder
def fm(d,y,x="sig"):
    s=[]
    for _,g in d.groupby(["etf","R"]):
        g=g[[x,y]].dropna()
        if len(g)<15 or g[x].std()==0: continue
        s.append(np.polyfit(g[x],g[y],1)[0])
    s=np.array(s); return s.mean(), s.mean()/s.std()*np.sqrt(len(s)), len(s)
for era in ["IS 2012-18","OOS 2019-"]:
    d=E[E.era==era]
    print(era, {y:"slope %+.1f bp per unit (t %+.2f, n %d)"%fm(d,y) for y in ["into","rday","after"]})
    d=d.copy(); d["q"]=d.groupby(["etf","R"]).sig.transform(lambda s: pd.qcut(s.rank(method="first"),5,labels=False)+1)
    g=d.groupby("q")[["into","rday","after"]].mean()
    print("   quintile of predicted trade (1=sell winners, 5=buy losers):"); print(g.round(0).T.to_string())
    hl=d.groupby(["etf","R","q"])[["into","rday","after"]].mean().unstack()
    for y in ["into","rday","after"]:
        x=(hl[y][5]-hl[y][1]).dropna(); print(f"   Q5-Q1 {y}: {x.mean():+.0f}bp (t {x.mean()/x.std()*np.sqrt(len(x)):+.2f})")
    for e_ in sorted(d.etf.unique()):
        x=(hl.loc[e_]["into"][5]-hl.loc[e_]["into"][1]).dropna(); y_=(hl.loc[e_]["after"][5]-hl.loc[e_]["after"][1]).dropna()
        print(f"     {e_}: into {x.mean():+.0f} (t {x.mean()/x.std()*np.sqrt(len(x)):+.1f}) after {y_.mean():+.0f} (t {y_.mean()/y_.std()*np.sqrt(len(y_)):+.1f})")

""")
md(r"""
**Interpretation.**
- **No reset pressure.** Pooled, laggards do not beat leaders into the reset (Q5−Q1 +16–17bp, t ≈ 0.4) or on the
  reset day (t ≤ 1.1), and there is no reversal after. Per-ETF results flip sign between eras (KRE +82bp in-sample,
  t 3.9, then −36bp; XRT and XBI change sign too).

**Verdict: dead.** The resets are anticipated, or other traders supply the liquidity.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
