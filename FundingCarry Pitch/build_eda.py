"""Generate eda.ipynb: perpetual-futures funding carry (long spot / short perp) on Binance, 2020-2026."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# FundingCarry Pitch EDA: how much does the perp funding carry pay after the cost of capital?

**The mechanism.** Perpetual futures pay funding every 8 hours from the crowded side to the other, by contract design.
Crypto demand for leverage is long-biased, so longs usually pay shorts. Long spot / short perp is delta-neutral and
collects it (the "cash-and-carry" / basis trade).

**Design.** Binance funding history and prices (`Funding Pitch/data`, spot from `CBPremium Pitch/data`). BTC: full P&L =
funding received + basis change, 8-hour steps, 10bp per entry/exit. All coins: funding yield, then excess over
3-month T-bills per unit of capital (spot notional + 20% perp margin = 1.2×). 2020–22 vs 2023–.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
perp=pd.read_csv("../Funding Pitch/data/klines_BTCUSDT.csv",parse_dates=["open_time"]).set_index("open_time").close
perp.index=perp.index+pd.Timedelta(minutes=5)                      # price at bar end
spot=pd.read_csv("../CBPremium Pitch/data/binance_BTCUSDT_1h.csv",parse_dates=["time"]).set_index("time").close
spot.index=spot.index+pd.Timedelta(hours=1)
fu=pd.read_csv("../Funding Pitch/data/funding_BTCUSDT.csv",parse_dates=["calc_time"]); fu["T"]=fu.calc_time.dt.round("h"); fu=fu.set_index("T").last_funding_rate*1e4
T=fu.index[(fu.index.hour%8==0)]
D=pd.DataFrame({"f":fu.reindex(T)}); D["perp"]=perp.reindex(T); D["spot"]=spot.reindex(T)
D=D.dropna()
D["basis"]=(D.perp/D.spot-1)*1e4
D["f_next"]=D.f.shift(-1)                                             # funding paid at the end of the next 8h period
D["dbasis"]=D.basis.shift(-1)-D.basis                                 # short perp / long spot loses when basis widens
D["pnl_hold"]=D.f_next-D.dbasis                                       # bp per 8h, per unit notional (one leg)
D=D.dropna()
for nm,sig in [("always",pd.Series(True,index=D.index)),("only if last funding > 0",D.f>0),("only if last funding > 1bp (~11%/yr)",D.f>1.0)]:
    pos=sig.astype(float); ch=pos.diff().abs().fillna(0)
    pnl=pos*D.pnl_hold-ch*10                                             # 10bp per entry/exit (2 legs, taker+spread)
    day=pnl.groupby(pnl.index.normalize()).sum()
    for per,m in [("IS 2020-2022",day.index<"2023"),("OOS 2023-",day.index>="2023")]:
        x=day[m]; print(f"{nm:38s} {per}: ann {x.mean()*365/100:+5.1f}% Sharpe {x.mean()/x.std()*np.sqrt(365):+.2f} t {x.mean()/x.std()*np.sqrt(len(x)):+.2f} in-market {pos[(pos.index<'2023')==(per.startswith('IS'))].mean():.0%} worst day {x.min():+.0f}bp")
import io, requests
print("\n==== excess over T-bills on capital used (spot notional + 20% perp margin = 1.2x), always-on carry, funding only (basis ignored for alts)")
tb=pd.read_csv(io.StringIO(requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3",timeout=60).text)); tb.columns=["date","r"]
tb["date"]=pd.to_datetime(tb.date); tb=pd.to_numeric(tb.set_index("date").r,errors="coerce").ffill()
for s in ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","DOGEUSDT"]:
    f=pd.read_csv(f"../Funding Pitch/data/funding_{s}.csv",parse_dates=["calc_time"]).set_index("calc_time").last_funding_rate*1e4
    day=f.groupby(f.index.normalize()).sum()                       # bp/day on notional
    rf=tb.reindex(day.index,method="ffill")/100*1e4/365            # bp/day
    ex=day/1.2-rf                                                  # per unit of capital
    for per,m in [("IS 2020-22",day.index<"2023"),("OOS 2023-",day.index>="2023")]:
        x=ex[m]; g=day[m]
        print(f"{s:9s} {per}: gross funding {g.mean()*365/100:+5.1f}%/yr | excess over T-bill per capital {x.mean()*365/100:+5.1f}%/yr t {x.mean()/x.std()*np.sqrt(len(x)):+.2f} | days with negative funding {(g<0).mean():.0%}")

""")
md(r"""
**Interpretation.**
- **The carry is real and extremely steady:** BTC always-on +17%/yr in 2020–22 (Sharpe 9, t 16) and +7%/yr since
  2023 (t 26), with basis changes included. Timing it on the last funding rate loses to switching costs.
- **After the cost of capital it has mostly gone:** in 2020–22 (T-bills near 0) the excess was +14–19%/yr; since 2023
  it is +1.2 to +2.3%/yr per unit of capital for BTC/ETH/XRP/DOGE (still statistically positive, t 4.8–9.7, because it
  barely moves), and negative for SOL.
- **Unpriced tails:** exchange default (FTX 2022), stablecoin depeg, and basis blowouts in crashes. Widely run
  (Ethena's USDe is built on this trade).

**Verdict: statistically clean but economically decayed.** About 1–2%/yr over T-bills for exchange and stablecoin
default risk is not a pitch.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
