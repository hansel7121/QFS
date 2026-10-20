"""Generate eda.ipynb: crypto crashes with and without forced liquidations (open-interest drops), and what follows."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Liquidation Pitch EDA: do forced liquidations overshoot, and who provides the liquidity?

**The mechanism.** When a perpetual-futures crash triggers margin liquidations, positions are closed at market
regardless of price: forced selling (long liquidations) or buying (short squeezes). Forced flow should overshoot and
revert, rewarding whoever buys from liquidators. A big drop in open interest during the move marks liquidations,
because positions are closed rather than opened.

**Design.** Binance BTC/ETH USDT perpetuals, 5-minute prices (`Funding Pitch/data`) and open interest
(`collect_metrics.py`, from 2021-12). Event = 15-minute return in the in-sample 0.2% tail, de-clustered to one per 4
hours. "OI drop" = 15-minute OI change in the in-sample bottom 5%. The trade fades the move (buy after crashes, sell
after spikes). In-sample 2021-12 → 2023, out-of-sample 2024 →. Robustness: events merged across coins by 4-hour
block, entry delayed 5–10 minutes (in case the OI timestamp is end-of-bar), 8bp taker round trip.
""")
code(r"""
import pandas as pd, numpy as np
out=[]
for s in ["BTCUSDT","ETHUSDT"]:
    k=pd.read_csv(f"../Funding Pitch/data/klines_{s}.csv",parse_dates=["open_time"])
    px=k.set_index(k.open_time+pd.Timedelta(minutes=5)).close          # price at time t (close of bar ending t)
    m=pd.read_csv(f"data/metrics_{s}.csv",parse_dates=["create_time"]).set_index("create_time").sum_open_interest
    idx=m.index.intersection(px.index); P=np.log(px.reindex(idx)); OI=np.log(m.reindex(idx))
    full=pd.DataFrame({"p":np.log(px)}); full=full[full.index>=idx[0]]
    D=pd.DataFrame({"p":P,"oi":OI})
    D["r15"]=(D.p-D.p.shift(3))*1e4; D["doi15"]=(D.oi-D.oi.shift(3))*100
    for h,lab in [(12,"f1h"),(48,"f4h"),(288,"f24h")]:
        D[lab]=(D.p.shift(-h)-D.p)*1e4
    D["sym"]=s
    is_=D.index<"2024-01-01"
    lo=D.r15[is_].quantile(0.002); hi=D.r15[is_].quantile(0.998)
    D["down"]=D.r15<=lo; D["up"]=D.r15>=hi
    D["oi_q"]=D.doi15<=D.doi15[is_].quantile(0.05)        # big OI drop = forced closing
    # de-cluster: keep first event per 4h
    for c in ["down","up"]:
        ev=D[D[c]].index; keep=[]; last=None
        for t in ev:
            if last is None or (t-last)>pd.Timedelta(hours=4): keep.append(t); last=t
        D[c+"_ev"]=D.index.isin(keep)
    out.append(D)
D=pd.concat(out); D["era"]=np.where(D.index<"2024-01-01","IS 2021-23","OOS 2024-")
for c,sign in [("down_ev",1),("up_ev",-1)]:
    print("\n", "crash (long liquidations)" if c=="down_ev" else "spike (short squeezes)", "-> fade (buy after crash / sell after spike)")
    for era in ["IS 2021-23","OOS 2024-"]:
        for oi in [True,False]:
            d=D[(D[c])&(D.era==era)&(D.oi_q==oi)]
            print(f"  {era} OI-drop={oi!s:5s} n {len(d):3d}: " + " | ".join(f"{h} {sign*d[h].mean():+.0f}bp (t {sign*d[h].mean()/d[h].std()*np.sqrt(len(d)) if len(d)>2 else np.nan:+.1f})" for h in ["f1h","f4h","f24h"]))
print("\n==== robustness: buy after crash with NO big OI drop; entry delayed; time-clustered; net 8bp taker RT")
res=[]
for s,Ds in D.groupby("sym"):
    p=Ds.p
    for delay in [0,1,2]:
        for h,lab in [(12,"1h"),(48,"4h"),(288,"24h")]:
            Ds[f"d{delay}_{lab}"]=(p.shift(-(h+delay))-p.shift(-delay))*1e4
    res.append(Ds)
D2=pd.concat(res)
ev=D2[D2.down_ev & ~D2.oi_q].copy()
ev["t"]=ev.index.floor("4h")
for delay in [0,1,2]:
    for lab in ["1h","4h","24h"]:
        g=ev.groupby("t").agg(r=(f"d{delay}_{lab}","mean"),era=("era","first"))
        line=f"delay {delay*5:2d}min hold {lab:3s}: "
        for era in ["IS 2021-23","OOS 2024-"]:
            x=g[g.era==era].r-8
            line+=f"{era} {x.mean():+5.0f}bp t {x.mean()/x.std()*np.sqrt(len(x)):+.2f} n {len(x)} hit {(x>0).mean():.0%} | "
        print(line)
# all crashes regardless of OI, for comparison (4h, delay 1)
g=D2[D2.down_ev].assign(t=lambda x:x.index.floor("4h")).groupby("t").agg(r=("d1_4h","mean"),era=("era","first"))
print("all crashes 4h delay5:", {e:f"{(g[g.era==e].r-8).mean():+.0f} (t {(g[g.era==e].r-8).mean()/(g[g.era==e].r).std()*np.sqrt((g.era==e).sum()):+.2f})" for e in ["IS 2021-23","OOS 2024-"]})

""")
md(r"""
**Interpretation.**
- **The liquidation hypothesis fails.** Crashes *with* big OI drops (forced long liquidations) do not revert over 1–4
  hours in either sample. Short squeezes (spikes with OI drops) do not reverse either; out of sample they keep going
  for 24 hours.
- **Crashes *without* an OI drop do revert.** Buying them: 4 hours +34 to +47bp net in-sample (t 1.4–1.9), +56 to
  +68bp out of sample (t 2.1–2.6); 24 hours +76–81bp (t 1.5–1.6) and +126–138bp (t 2.8–3.0). Hit rates 60–69%. All
  crashes together earn nothing (+2 / +7bp), so the OI split is what matters.
- **Why that would be:** a crash where open interest holds up is driven by new shorts or spot selling, not
  margin-forced closes. That selling is discretionary and gets absorbed and reversed. A liquidation crash removes
  leveraged longs and the selling is finished, but it does not bounce, perhaps because it reflects real news that set
  off the leverage unwind.

**Verdict: a lead, not a pass.** It has the same sign in both periods and is significant out of sample after delay
and costs, but in-sample it is only t ≈ 1.5, and the effect sits in the group that was meant as the control. It
needs pre-registration on fresh data (events after October 2026) or another venue (Bybit/OKX open interest).
""")

md(r"""
## Out-of-sample check on other coins (rule frozen from BTC/ETH)
**What this does.** Applies the exact rule (own in-sample tail thresholds, 5-minute delay, 8bp cost, 4h/24h holds)
to SOL, XRP and DOGE, which were not used to find it.
""")
code(r"""import pandas as pd, numpy as np
# pre-registered from BTC/ETH: 15m return in own in-sample 0.2% tail, OI change NOT in own in-sample bottom 5%,
# one event per 4h, enter 5 min after, hold 4h and 24h, 8bp cost
rows=[]
for s in ["SOLUSDT","XRPUSDT","DOGEUSDT"]:
    k=pd.read_csv(f"../Funding Pitch/data/klines_{s}.csv",parse_dates=["open_time"])
    px=k.set_index(k.open_time+pd.Timedelta(minutes=5)).close
    m=pd.read_csv(f"data/metrics_{s}.csv",parse_dates=["create_time"]).set_index("create_time").sum_open_interest
    idx=m.index.intersection(px.index); D=pd.DataFrame({"p":np.log(px.reindex(idx)),"oi":np.log(m.reindex(idx).replace(0,np.nan))})
    D["r15"]=(D.p-D.p.shift(3))*1e4; D["doi15"]=(D.oi-D.oi.shift(3))*100
    is_=D.index<"2024-01-01"
    lo=D.r15[is_].quantile(0.002); oiq=D.doi15[is_].quantile(0.05)
    ev=D[(D.r15<=lo)].index; keep=[]; last=None
    for t in ev:
        if last is None or (t-last)>pd.Timedelta(hours=4): keep.append(t); last=t
    for t in keep:
        i=D.index.get_loc(t)
        if i+1+288>=len(D): continue
        rows.append({"sym":s,"t":t,"oi_drop":D.doi15.iloc[i]<=oiq,"r4h":(D.p.iloc[i+1+48]-D.p.iloc[i+1])*1e4,"r24h":(D.p.iloc[i+1+288]-D.p.iloc[i+1])*1e4})
E=pd.DataFrame(rows); E["era"]=np.where(E.t<"2024-01-01","2021-23","2024-")
E["blk"]=E.t.dt.floor("4h")
for oi in [False,True]:
    for era in ["2021-23","2024-"]:
        g=E[(E.oi_drop==oi)&(E.era==era)].groupby("blk")[["r4h","r24h"]].mean()-8
        print(f"OI-drop={oi!s:5s} {era}: n {len(g)} | 4h {g.r4h.mean():+.0f} (t {g.r4h.mean()/g.r4h.std()*np.sqrt(len(g)):+.2f}) | 24h {g.r24h.mean():+.0f} (t {g.r24h.mean()/g.r24h.std()*np.sqrt(len(g)):+.2f})")
""")
md(r"""
**Interpretation.** The altcoins do not confirm it. In 2021–23, crashes without OI drops earn −13bp over 4h (t −0.2).
In 2024– they pay (+63bp 4h, +189bp 24h), but so do crashes *with* OI drops (+136bp 24h). What survives is "buying
sharp dips paid in the 2024–26 crypto market", a regime, not a liquidation-specific mechanism.

**Final verdict: not a pitch.** The lead is downgraded.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
