"""Generate eda.ipynb: overnight vs intraday returns of cross-sectional strategies (Lou, Polk & Skouras 2019), tradability."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Overnight Pitch EDA: can you trade the overnight half of momentum?

**The paper.** Lou, Polk & Skouras, *"A Tug of War: Overnight Versus Intraday Expected Returns"* (**JFE** 2019):
momentum profits are earned entirely overnight and partly reversed intraday. Each stock's overnight and intraday
returns also persist. They attribute this to different clienteles trading at the open and during the day.

**Tradable angle.** Buying at the closing auction (MOC) and selling at the opening auction (MOO) pays no bid-ask
spread, only fees and commissions. So the overnight leg could be harvested on its own if costs are a fraction of a bp.

**Design.** Current S&P 500 members (**survivorship bias**, which favours momentum winners), 2004–2026, Yahoo daily
open/close (`collect_panel.py`). Overnight = previous close → open (dividend-adjusted); intraday = open → close.
Decile long-short, equal weight. Signals: 12-1 month momentum (monthly) and the past-month average overnight /
intraday return (persistence). In-sample 2005–16, out-of-sample 2017–.
""")
code(r"""
import pandas as pd, numpy as np
p=pd.read_csv("data/panel.csv",parse_dates=["date"])
O=p.pivot(index="date",columns="ticker",values="open"); C=p.pivot(index="date",columns="ticker",values="close"); A=p.pivot(index="date",columns="ticker",values="adj_close")
spy_on=None
f=A/C
ON=(O*f/A.shift(1)-1)*1e4; ID=(C/O-1)*1e4
bad=(ON.abs()>3000)|(ID.abs()>3000); ON=ON.mask(bad); ID=ID.mask(bad)
ON=ON.drop(columns="SPY"); ID=ID.drop(columns="SPY"); A2=A.drop(columns="SPY")
cal=ON.index
# signals at month end
me=pd.Series(cal,index=cal).groupby(cal.to_period("M")).max().values
mom=(A2.shift(21)/A2.shift(252)-1)
pon=ON.rolling(21,min_periods=15).mean()   # past 1-month avg overnight
pid=ID.rolling(21,min_periods=15).mean()
def ls(sig, comp, freq="M", q=0.1):
    # long top decile, short bottom, weights fixed for next period; returns daily series of comp
    out=pd.Series(0.0,index=cal)
    if freq=="M":
        dates=me
    else:
        dates=cal
    s=sig.loc[dates]
    rank=s.rank(axis=1,pct=True)
    w=((rank>1-q).astype(float).div((rank>1-q).sum(1),axis=0)-(rank<=q).astype(float).div((rank<=q).sum(1),axis=0))
    w=w.reindex(cal).ffill().shift(1)   # set at close of signal date, earn from next day
    return (w*comp).sum(1)
def rep(name, r):
    for per,m in [("IS 2005-16",(cal>="2005")&(cal<"2017")),("OOS 2017-",cal>="2017")]:
        x=r[m]; print(f"{name:32s} {per}: {x.mean()*252/100:+6.1f}%/yr Sharpe {x.mean()/x.std()*np.sqrt(252):+.2f} t {x.mean()/x.std()*np.sqrt(len(x)):+.2f}")
rep("mom L-S overnight", ls(mom,ON)); rep("mom L-S intraday", ls(mom,ID))
rep("past-ON L-S overnight (monthly)", ls(pon,ON)); rep("past-ON L-S intraday (monthly)", ls(pon,ID))
rep("past-ON L-S overnight (daily reb)", ls(pon,ON,"D"))
rep("past-ID L-S intraday (daily reb)", ls(pid,ID,"D"))
print("\nnet of auction costs (each leg trades MOC in and MOO out daily; c bp per side per leg; gross exposure 2):")
for nm,r in [("past-ON overnight",ls(pon,ON)),("mom overnight",ls(mom,ON))]:
    for c in [0.25,0.5,1.0]:
        rn=r-2*2*c
        x=rn[cal>="2017"]; print(f"{nm:18s} c={c}: OOS {x.mean()*252/100:+6.1f}%/yr Sharpe {x.mean()/x.std()*np.sqrt(252):+.2f}")
    cc=(ls(pon if nm.startswith('past') else mom,ON)+ls(pon if nm.startswith('past') else mom,ID))[cal>="2017"]
    print(f"   hold close-to-close instead (monthly turnover only): OOS {cc.mean()*252/100:+.1f}%/yr Sharpe {cc.mean()/cc.std()*np.sqrt(252):+.2f}")

""")
md(r"""
**Interpretation.**
- **The decomposition replicates.** Momentum earns overnight (+4.6%/yr in-sample, +13.3% out of sample) and loses
  intraday. Overnight-return persistence is very strong: past-month overnight return predicts the next month's overnight
  return with Sharpe 3.9 in-sample and 0.80 (t 2.5) out of sample. The intraday leg reverses it, so held close to
  close it earns nothing.
- **Costs decide it.** Harvesting only the overnight leg means a full round trip every day on both sides of the book.
  At 0.25bp per side (institutional auction fees): momentum-overnight Sharpe 0.71, persistence 0.58 out of sample.
  At a more realistic 0.5bp: 0.54 / 0.35. At 1bp: about zero.
- The momentum-overnight version was only t = 1.8 in-sample (gross), and the universe is survivorship-biased in its
  favour.

**Verdict: near miss.** The effect is real and big gross, but it only survives at institutional auction costs
(≤ 0.25bp per side). For a student/retail implementation it fails. Worth revisiting with a point-in-time universe
and explicit fee schedules.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
