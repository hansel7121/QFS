import numpy as np, pandas as pd, statsmodels.api as sm
D="/Users/hanselchen/Desktop/QFS/BDC Pitch/data/"
H=pd.read_csv(D+"nav_history.csv",parse_dates=["period","public"])
H=H[H.clean & (H.ticker!="GECC") & ~((H.ticker=="CSWC")&(H.period<"2016-01-01"))].sort_values(["ticker","period"])
p=pd.read_csv(D+"prices.csv",parse_dates=["date"])
C=p.pivot(index="date",columns="ticker",values="close"); A=p.pivot(index="date",columns="ticker",values="adj_close"); O=p.pivot(index="date",columns="ticker",values="open")
dates=C.index; bdcs=[t for t in H.ticker.unique() if t in C]
R=np.log(A).diff()
peer=R[bdcs].mean(1)                                   # equal-weight BDC return (available names)
# reaction day
pub=H.public
pos=np.searchsorted(dates, pub.dt.normalize(), side="left")
is_td=np.isin(pub.dt.normalize(), dates)
before_open=(pub.dt.hour*60+pub.dt.minute)<570
react=np.where(is_td & before_open, pos, np.where(is_td, pos+1, pos))   # after 9:30 on a trading day: next day (conservative: entry at that day's close)
H["ri"]=react; H=H[(H.ri>0)&(H.ri<len(dates)-70)]
H["nav_prev"]=H.groupby("ticker").nav_ps.shift(1); H["per_prev"]=H.groupby("ticker").period.shift(1)
H=H[(H.period-H.per_prev).dt.days.between(80,100)]
H["dnav"]=np.log(H.nav_ps/H.nav_prev)
def px(df,t,i): 
    try: return df[t].iat[i]
    except: return np.nan
H["ar"]=[ (R[t].iloc[i:i+1].sum()-peer.iloc[i:i+1].sum()) if t in R else np.nan for t,i in zip(H.ticker,H.ri)]
H["ar5"]=[ (R[t].iloc[i:i+5].sum()-peer.iloc[i:i+5].sum()) if t in R else np.nan for t,i in zip(H.ticker,H.ri)]
H["ar20"]=[ (R[t].iloc[i+1:i+21].sum()-peer.iloc[i+1:i+21].sum()) if t in R else np.nan for t,i in zip(H.ticker,H.ri)]
# pre-release information
bk=np.log(A["BKLN"]).diff()
pe_i=np.searchsorted(dates,H.period,side="right")-1; pp_i=np.searchsorted(dates,H.per_prev,side="right")-1
H["bkln_q"]=[bk.iloc[a+1:b+1].sum() for a,b in zip(pp_i,pe_i)]
H["own_q"]=[R[t].iloc[a+1:b+1].sum()-peer.iloc[a+1:b+1].sum() if t in R else np.nan for t,a,b in zip(H.ticker,pp_i,pe_i)]
H["own_pre"]=[R[t].iloc[a+1:r].sum()-peer.iloc[a+1:r].sum() if t in R else np.nan for t,a,r in zip(H.ticker,pp_i,H.ri)]  # quarter start -> day before reaction
H["disc_pre"]=[np.log(px(C,t,r-1)/n) for t,r,n in zip(H.ticker,H.ri,H.nav_prev)]
H=H.dropna(subset=["ar","dnav","disc_pre","own_pre"])
H["bkln_q1"]=H.groupby("ticker").bkln_q.shift(1)
H["dnav_cs"]=H.dnav-H.groupby("period").dnav.transform("median")
H["year"]=H.period.dt.year; IS=H.period<"2020-01-01"
tt=lambda s: s.mean()/s.std()*np.sqrt(len(s))
print("events",len(H),"IS",IS.sum(),"OOS",(~IS).sum())
print("\nA. Does the market react to the NAV news? corr(reaction-day abnormal return, NAV change minus quarter median)")
for lab,m in [("IS 2012-19",IS),("OOS 2020-",~IS)]:
    z=H[m]; print(lab,"spearman",round(z.ar.corr(z.dnav_cs,method="spearman"),3))
print("\nB. Is the NAV change predictable before release? cross-sectional (demeaned by quarter)")
for v in ["own_pre","disc_pre","own_q"]:
    for lab,m in [("IS",IS),("OOS",~IS)]:
        z=H[m]; x=z[v]-z.groupby("period")[v].transform("median")
        print(f"  {v:9s} {lab}: spearman with dnav_cs {x.corr(z.dnav_cs,method='spearman'):+.3f}")
print("\nC. Does pre-release info predict the reaction (alpha)? quarter-demeaned, FM by quarter")
for v in ["own_pre","disc_pre","own_q"]:
    for y in ["ar","ar5","ar20"]:
        out=f"  {v:9s} -> {y:5s}"
        for lab,m in [("IS",IS),("OOS",~IS)]:
            z=H[m].copy(); z["x"]=z[v]-z.groupby("period")[v].transform("mean")
            sl=z.groupby("period").apply(lambda g: np.polyfit(g.x/g.x.std(),g[y],1)[0] if len(g)>=8 and g.x.std()>0 else np.nan).dropna()
            out+=f" | {lab} slope {sl.mean()*1e4:6.1f}bp/sd t {tt(sl):5.2f} (q {len(sl)})"
        print(out)
H.to_pickle("/private/tmp/claude-501/-Users-hanselchen-Desktop-QFS/0a137c4b-a128-4cf9-bc0c-83c602157027/scratchpad/bdc_events.pkl")
