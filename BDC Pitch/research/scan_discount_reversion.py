import numpy as np, pandas as pd, statsmodels.api as sm
D="/Users/hanselchen/Desktop/QFS/BDC Pitch/data/"
H=pd.read_csv(D+"nav_history.csv",parse_dates=["period","public"])
H=H[H.clean & (H.ticker!="GECC") & ~((H.ticker=="CSWC")&(H.period<"2016-01-01"))].sort_values(["ticker","period"])
p=pd.read_csv(D+"prices.csv",parse_dates=["date"])
C=p.pivot(index="date",columns="ticker",values="close"); A=p.pivot(index="date",columns="ticker",values="adj_close")
dates=C.index; bdcs=[t for t in H.ticker.unique() if t in C]
# daily last-public NAV (effective the trading day after public timestamp to be safe)
N=pd.DataFrame(index=dates,columns=bdcs,dtype=float)
for t,g in H[H.ticker.isin(bdcs)].groupby("ticker"):
    eff=dates[np.minimum(np.searchsorted(dates,g.public.dt.normalize(),side="right"),len(dates)-1)]
    s=pd.Series(g.nav_ps.values,index=eff); s=s[~s.index.duplicated(keep="last")]
    N[t]=s.reindex(dates).ffill()
    # drop NAV older than 150 days (missed filings)
    last=pd.Series(s.index,index=s.index).reindex(dates).ffill()
    N.loc[(dates.to_series()-last).dt.days>150,t]=np.nan
disc=np.log(C[bdcs]/N)
R=np.log(A[bdcs]).diff(); peer=R.mean(1)
mu=disc.rolling(252,min_periods=180).mean().shift(1); sd=disc.rolling(252,min_periods=180).std().shift(1)
Z=(disc-mu)/sd
cs=disc.sub(disc.median(1),axis=0)       # cross-sectional discount vs peers today
print("names/day",Z.notna().sum(1).resample("YE").mean().round(0).to_dict())
tt=lambda s: s.mean()/s.std()*np.sqrt(len(s))
for H_ in [5,20,60]:
    fwd=sum(R.shift(-k) for k in range(1,H_+1)).sub(sum(peer.shift(-k) for k in range(1,H_+1)),axis=0)
    for sig,lab in [(Z,"own z-score"),(cs,"discount vs peers")]:
        out=f"H{H_:2d} {lab:18s}"
        for per,m in [("IS 12-19",dates<"2020-01-01"),("OOS 20-",dates>="2020-01-01")]:
            # weekly (non-overlapping-ish) cross-sectional spearman, then t over weeks with HAC
            ix=dates[m][::5]
            rho=pd.Series({d:sig.loc[d].corr(fwd.loc[d],method="spearman") for d in ix if sig.loc[d].notna().sum()>=10 and fwd.loc[d].notna().sum()>=10}).dropna()
            f=sm.OLS(rho,np.ones(len(rho))).fit(cov_type="HAC",cov_kwds={"maxlags":max(1,H_//5)})
            out+=f" | {per} mean rho {rho.mean():+.3f} t {f.tvalues.iloc[0]:+.2f}"
        print(out)
Z.to_pickle("/private/tmp/claude-501/-Users-hanselchen-Desktop-QFS/0a137c4b-a128-4cf9-bc0c-83c602157027/scratchpad/bdc_Z.pkl"); disc.to_pickle("/private/tmp/claude-501/-Users-hanselchen-Desktop-QFS/0a137c4b-a128-4cf9-bc0c-83c602157027/scratchpad/bdc_disc.pkl")
print("\n--- robustness: skip one day; control for past relative returns (Fama-MacBeth, weekly) ---")
rel=R.sub(peer,axis=0)
past20=rel.rolling(20).sum(); past60=rel.rolling(60).sum(); past5=rel.rolling(5).sum()
for H_ in [20,60]:
    fwd=sum(rel.shift(-k) for k in range(2,H_+2))          # skip day t+1
    for per,m in [("IS 12-19",dates<"2020-01-01"),("OOS 20-",dates>="2020-01-01")]:
        ix=dates[m][::5]; co=[]
        for d in ix:
            df=pd.DataFrame({"z":Z.loc[d],"p5":past5.loc[d],"p20":past20.loc[d],"p60":past60.loc[d],"y":fwd.loc[d]}).dropna()
            if len(df)<12: continue
            X=(df[["z","p5","p20","p60"]]-df[["z","p5","p20","p60"]].mean())/df[["z","p5","p20","p60"]].std()
            co.append(sm.OLS(df.y,sm.add_constant(X)).fit().params)
        co=pd.DataFrame(co)
        ts={k: sm.OLS(co[k],np.ones(len(co))).fit(cov_type="HAC",cov_kwds={"maxlags":H_//5}).tvalues.iloc[0] for k in ["z","p5","p20","p60"]}
        print(f"H{H_} skip1 {per}: z {co.z.mean()*1e4:+.0f}bp/sd (t {ts['z']:+.2f}) | past5 t {ts['p5']:+.2f} | past20 t {ts['p20']:+.2f} | past60 t {ts['p60']:+.2f}")
print("\n--- univariate, skip one day (weekly cross-sectional spearman, HAC t) ---")
for H_ in [5,20,60]:
    fwd=sum(rel.shift(-k) for k in range(2,H_+2))
    for sig,lab in [(Z,"own discount z"),(past20,"past 20d rel return"),(past5,"past 5d rel return")]:
        out=f"H{H_:2d} {lab:20s}"
        for per,m in [("IS",dates<"2020-01-01"),("OOS",dates>="2020-01-01")]:
            ix=dates[m][::5]
            rho=pd.Series({d:sig.loc[d].corr(fwd.loc[d],method="spearman") for d in ix if sig.loc[d].notna().sum()>=10 and fwd.loc[d].notna().sum()>=10}).dropna()
            f=sm.OLS(rho,np.ones(len(rho))).fit(cov_type="HAC",cov_kwds={"maxlags":max(1,H_//5)})
            out+=f" | {per} rho {rho.mean():+.3f} t {f.tvalues.iloc[0]:+.2f}"
        print(out)
print("\n--- portfolios: weekly signal (Friday close), enter next close (skip a day), hold H days, overlapping tranches ---")
DV=(C[bdcs]*p.pivot(index="date",columns="ticker",values="volume")[bdcs]).rolling(20).median()
liq=DV.shift(1)>=1e6
bizd=np.log(A["BIZD"]).diff()
def port(sig,H_,q=0.2,cost_bp=15):
    wl=pd.DataFrame(0.0,index=dates,columns=bdcs); ws=wl.copy()
    reb=dates[4::5]
    for d in reb:
        s=sig.loc[d].where(liq.loc[d]).dropna()
        if len(s)<10: continue
        lo=s[s<=s.quantile(q)].index; hi=s[s>=s.quantile(1-q)].index
        i=dates.get_loc(d)
        sl=slice(i+2,i+2+H_)                       # skip day i+1
        wl.iloc[sl][lo]=0                          # placeholder
        wl.iloc[i+2:i+2+H_, [bdcs.index(x) for x in lo]] += 1/len(lo)/ (H_/5)
        ws.iloc[i+2:i+2+H_, [bdcs.index(x) for x in hi]] += 1/len(hi)/ (H_/5)
    rr=R.fillna(0)
    L=(wl*rr).sum(1); S=(ws*rr).sum(1)
    tl=wl.diff().abs().sum(1); ts=ws.diff().abs().sum(1)
    ls=L-S-(tl+ts)*cost_bp/1e4
    lb=L-wl.sum(1)*bizd.fillna(0)-tl*cost_bp/1e4-wl.diff().sum(1).abs()*2/1e4   # long cheap vs short BIZD (BIZD 2bp)
    return ls, lb, (tl+ts).mean()
for H_ in [20,60]:
    for cost in [0,15,30]:
        ls,lb,to=port(Z,H_,cost_bp=cost)
        out=f"z-sig H{H_} cost {cost:2d}bp/side:"
        for per,m in [("IS",dates<"2020-01-01"),("OOS",dates>="2020-01-01")]:
            for nm,s in [("L/S",ls),("long-cheap/short-BIZD",lb)]:
                x=s[m]; x=x[x.index>="2013-03-01"]; x=x[x!=0]
                f=sm.OLS(x,np.ones(len(x))).fit(cov_type="HAC",cov_kwds={"maxlags":10})
                out+=f" | {per} {nm} {x.mean()*252*100:5.1f}%/yr SR {x.mean()/x.std()*np.sqrt(252):4.2f} t {f.tvalues.iloc[0]:4.2f}"
        print(out, f"| daily turnover {to:.3f}")
print("\n--- variants (L/S, H20) ---")
comb=(Z.rank(axis=1,pct=True)+past20.rank(axis=1,pct=True))
for thr in [1e6,5e6]:
    liq=DV.shift(1)>=thr
    for sig,lab in [(Z,"z"),(past20,"past20"),(comb,"z+past20")]:
        for cost in [8,15]:
            ls,lb,to=port(sig,20,cost_bp=cost)
            out=f"DV>={thr/1e6:.0f}m {lab:9s} cost {cost:2d}:"
            for per,m in [("IS",dates<"2020-01-01"),("OOS",dates>="2020-01-01")]:
                x=ls[m]; x=x[(x.index>="2013-03-01")&(x!=0)]
                f=sm.OLS(x,np.ones(len(x))).fit(cov_type="HAC",cov_kwds={"maxlags":10})
                out+=f" | {per} {x.mean()*252*100:5.1f}%/yr SR {x.mean()/x.std()*np.sqrt(252):4.2f} t {f.tvalues.iloc[0]:4.2f}"
            print(out)
